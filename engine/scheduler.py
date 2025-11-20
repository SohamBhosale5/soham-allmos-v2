"""
Scheduler for continuous batching with prefill/decode separation.

Design Philosophy:
- Maximize GPU utilization through dynamic batching
- Separate prefill (first token) and decode (subsequent tokens) phases
- Support preemption when memory is constrained
- Integrate tightly with BlockManager for memory decisions

Optimization Notes (from benchmark report):
- Continuous batching: 10-50x throughput improvement
- Key insight: GPU can process multiple sequences in parallel
- Prefill/decode separation: Different compute patterns, batch separately
- Preemption: Gracefully handle memory pressure by pausing sequences

Architecture:
- Waiting queue: Sequences waiting to be scheduled
- Running queue: Sequences currently being processed
- Scheduler decides which sequences run in each step
"""
from collections import deque
from typing import List, Tuple

from config import Config
from engine.types import Scheduler as SchedulerABC
from engine.sequence import Sequence, SequenceStatus
from memory.block_manager import BlockManager


class Scheduler(SchedulerABC):
    """
    Continuous batching scheduler with prefill/decode separation.

    The scheduler maintains two queues:
    1. Waiting: New sequences waiting to start
    2. Running: Sequences currently being processed

    In each step, the scheduler:
    - Tries to schedule waiting sequences (prefill phase)
    - If no waiting sequences, schedules running sequences (decode phase)
    - Handles preemption if memory is insufficient
    """

    def __init__(self, config: Config):
        """
        Initialize scheduler.

        Args:
            config: System configuration with batch/memory limits
        """
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos_token_id = config.eos_token_id
        
        # Interleaving configuration
        self.enable_interleaving = config.enable_prefill_decode_interleaving
        self.prefill_token_budget_ratio = config.prefill_token_budget_ratio
        self.min_prefill_batch_size = config.min_prefill_batch_size
        self.min_decode_batch_size = config.min_decode_batch_size

        # Block manager for KV cache memory
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )

        # Sequence queues
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """
        Check if all sequences are complete.

        Returns:
            True if no sequences are waiting or running
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """
        Add a new sequence to the waiting queue.

        Args:
            seq: Sequence to add
        """
        self.waiting.append(seq)

    def schedule(self) -> Tuple[List[Sequence], List[Sequence]]:
        """
        Schedule sequences for the next execution step with prefill/decode interleaving.

        Strategy:
        1. If interleaving enabled: Schedule both prefill and decode in same step
           - Allocate token budget between prefill (30%) and decode (70%)
           - Schedule prefill sequences up to budget
           - Schedule decode sequences with remaining capacity
           
        2. If interleaving disabled: Use original behavior
           - Prioritize prefill, then decode

        Returns:
            Tuple of (prefill_sequences, decode_sequences)
        """
        prefill_seqs = []
        decode_seqs = []
        
        if not self.enable_interleaving:
            # Original behavior: schedule prefill OR decode
            scheduled_seqs = []
            num_seqs = 0
            num_batched_tokens = 0

            # Try prefill phase first
            while self.waiting and num_seqs < self.max_num_seqs:
                seq = self.waiting[0]
                tokens_needed = len(seq) - seq.num_cached_tokens

                if (num_batched_tokens + tokens_needed > self.max_num_batched_tokens or
                    not self.block_manager.can_allocate(seq)):
                    break

                num_seqs += 1
                self.block_manager.allocate(seq)
                num_batched_tokens += tokens_needed
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)

            if scheduled_seqs:
                return scheduled_seqs, []

            # Decode phase
            while self.running and num_seqs < self.max_num_seqs:
                seq = self.running.popleft()
                while not self.block_manager.can_append(seq):
                    if self.running:
                        victim = self.running.pop()
                        self.preempt(victim)
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)

            assert scheduled_seqs, "No sequences could be scheduled!"
            self.running.extendleft(reversed(scheduled_seqs))
            return [], scheduled_seqs

        # Interleaving enabled: schedule both prefill and decode
        prefill_token_budget = int(self.max_num_batched_tokens * self.prefill_token_budget_ratio)
        prefill_tokens_used = 0
        prefill_count = 0
        
        # Schedule prefill sequences (up to budget)
        while self.waiting and prefill_count < self.max_num_seqs:
            seq = self.waiting[0]
            tokens_needed = len(seq) - seq.num_cached_tokens

            # Check if we exceed prefill budget or total capacity
            if (prefill_tokens_used + tokens_needed > prefill_token_budget or
                prefill_count >= self.max_num_seqs or
                not self.block_manager.can_allocate(seq)):
                break

            # Schedule the prefill sequence
            self.block_manager.allocate(seq)
            prefill_tokens_used += tokens_needed
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            prefill_seqs.append(seq)
            prefill_count += 1

        # Schedule decode sequences (remaining capacity)
        decode_count = 0
        max_decode_seqs = self.max_num_seqs - len(prefill_seqs)
        
        # Only schedule decode if we have minimum batch size or no prefill
        decode_candidates = []
        temp_running = deque()
        
        while self.running and decode_count < max_decode_seqs:
            seq = self.running.popleft()
            
            # Check if we can append a token
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    self.preempt(seq)
                    break
            else:
                decode_candidates.append(seq)
                decode_count += 1

        # Only schedule decode if we meet minimum batch size or have no prefill
        if len(decode_candidates) >= self.min_decode_batch_size or (not prefill_seqs and decode_candidates):
            for seq in decode_candidates:
                self.block_manager.may_append(seq)
                decode_seqs.append(seq)
            # Put decode sequences back at front of running queue
            self.running.extendleft(reversed(decode_seqs))
        else:
            # Put candidates back if we didn't schedule them
            self.running.extendleft(reversed(decode_candidates))

        # Must have at least one sequence to run
        assert prefill_seqs or decode_seqs, "No sequences could be scheduled!"

        return prefill_seqs, decode_seqs

    def preempt(self, seq: Sequence) -> None:
        """
        Preempt a sequence by deallocating its memory and moving to waiting.

        The sequence will be rescheduled later when memory is available.

        Args:
            seq: Sequence to preempt
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: List[Sequence], token_ids: List[int]) -> None:
        """
        Process generated tokens and update sequence states.

        For each sequence:
        1. Append the generated token
        2. Check if generation is complete (EOS or max_tokens reached)
        3. If complete, deallocate memory and remove from running queue

        Args:
            seqs: Sequences that just generated tokens
            token_ids: Generated token IDs (one per sequence)
        """
        for seq, token_id in zip(seqs, token_ids):
            # Append the new token
            seq.append_token(token_id)

            # Check stopping criteria
            is_eos = (not seq.ignore_eos and token_id == self.eos_token_id)
            is_max_len = (seq.num_completion_tokens == seq.max_tokens)

            if is_eos or is_max_len:
                # Sequence is finished
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
