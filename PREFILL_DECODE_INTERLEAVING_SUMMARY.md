# Prefill/Decode Interleaving Implementation Summary

## Overview

This document summarizes the attempt to implement prefill/decode interleaving for improved GPU utilization in allmos_v2. The implementation was attempted but encountered critical issues that prevent it from working reliably.

## Motivation

**Goal**: Improve throughput by running prefill and decode phases simultaneously in the same step, rather than sequentially.

**Expected Benefits**:
- Better GPU utilization (reduce idle time between prefill and decode)
- Lower latency for both new requests (prefill) and ongoing requests (decode)
- Expected 10-20% throughput improvement

## Implementation Approach

### Design: Dual Execution

The implementation attempted to:
1. Schedule both prefill and decode sequences in the same step
2. Allocate token budget between prefill (30%) and decode (70%)
3. Run both phases in the same execution step but as separate forward passes

### Changes Made

1. **Configuration** (`config.py`):
   - Added `enable_prefill_decode_interleaving` flag
   - Added `prefill_token_budget_ratio` (default 0.3)
   - Added minimum batch size parameters

2. **Scheduler** (`engine/scheduler.py`):
   - Modified `schedule()` to return `(prefill_seqs, decode_seqs)` instead of `(seqs, is_prefill)`
   - Implemented budget allocation between prefill and decode
   - Attempted to schedule both types in same step

3. **Engine** (`engine/llm_engine.py`):
   - Modified `step()` to handle dual execution
   - Run prefill and decode as separate model forward passes

4. **Model Runner** (`engine/model_runner.py`):
   - Added context reset before/after runs to prevent contamination

## Critical Problems Encountered

### Problem 1: Global Context State Contamination

**Issue**: The attention context is stored in a **single global variable** (`_CONTEXT` in `utils/context.py`).

**Why it fails**:
- Prefill sets `is_prefill=True`, `cu_seqlens_q/k`, etc.
- Decode immediately sets `is_prefill=False`, `context_lens`, etc.
- CUDA operations are **asynchronous** - prefill kernels may still be running when decode starts
- Decode reads prefill's context state → wrong parameters passed to Flash Attention

**Error**: Flash Attention receives incorrect parameters, leading to crashes or incorrect results.

---

### Problem 2: Asynchronous CUDA Execution Without Synchronization

**Issue**: No GPU synchronization between prefill and decode phases.

**Why it fails**:
```python
if prefill_seqs:
    model_runner.call("run", prefill_seqs, True)  # Starts async GPU work
    # NO cuda.synchronize() here!

if decode_seqs:
    model_runner.call("run", decode_seqs, False)  # Starts immediately
    # Decode runs while prefill kernels still executing!
```

- PyTorch CUDA operations are **asynchronous by default**
- Decode starts while prefill kernels are still executing
- Both access the same **KV cache memory simultaneously**
- Results in **race conditions and memory corruption**

**Error**: `RuntimeError: CUDA error: an illegal memory access was encountered`

---

### Problem 3: CUDA Graph Replay Conflicts

**Issue**: Prefill uses eager mode while decode uses CUDA graphs.

**Why it fails**:
- **Prefill**: Eager mode - writes directly to KV cache (variable shapes)
- **Decode**: CUDA graph mode - uses pre-allocated `graph_vars` buffers (fixed shapes)
- CUDA graphs assume **exclusive access** during replay
- Prefill writes to KV cache while decode graph is reading → **memory conflicts**

**Error**: Illegal memory access errors during CUDA graph replay.

---

### Problem 4: Incompatible Flash Attention APIs

**Issue**: Prefill and decode use completely different Flash Attention APIs.

**Why it fails**:
- **Prefill**: `flash_attn_varlen_func()` - variable-length batches with `cu_seqlens_q/k`
- **Decode**: `flash_attn_with_kvcache()` - fixed-length batches with `cache_seqlens`
- Different memory layouts and parameter structures
- Running back-to-back without proper GPU synchronization → **internal Flash Attention state corruption**

**Error**: Flash Attention state corruption, incorrect attention computations.

---

### Problem 5: Tensor Shape Mismatches

**Issue**: Prefill and decode have different `slot_mapping` tensor shapes.

**Why it fails**:
- **Prefill**: `slot_mapping` shape `[total_tokens]` (e.g., `[491]` for all tokens across sequences)
- **Decode**: `slot_mapping` shape `[batch_size]` (e.g., `[32]` for one slot per sequence)
- Context contamination causes decode to use prefill's slot_mapping
- Leads to dimension mismatch errors

**Error**: `IndexError: index_copy_(): Number of indices (0) should be equal to source.size(dim) (491)`

---

## Root Cause Analysis

The fundamental issue is that **prefill and decode are fundamentally incompatible when run simultaneously** without proper isolation:

1. **No GPU Synchronization**: Async CUDA operations overlap
2. **Shared Global State**: Single context variable used by both phases
3. **Shared Memory**: Both access KV cache simultaneously
4. **Different Execution Modes**: Eager (prefill) vs CUDA graphs (decode)
5. **Incompatible APIs**: Different Flash Attention functions

## Attempted Fixes

Several fixes were attempted but insufficient:

1. ✅ **Context Reset**: Added `reset_context()` before/after model runs
   - **Result**: Helped but not sufficient - GPU operations still overlap

2. ✅ **Empty Slot Mapping Check**: Check if `slot_mapping` is empty before storing KV cache
   - **Result**: Prevented some crashes but didn't fix root cause

3. ❌ **Context Isolation**: Not implemented - would require major refactoring

4. ❌ **GPU Synchronization**: Not added - would add latency overhead

5. ❌ **Separate CUDA Streams**: Not implemented - complex to coordinate

## Current Status

**Implementation**: Complete but **disabled by default**

**Configuration**:
```python
# config.py
enable_prefill_decode_interleaving: bool = False  # Disabled until issues resolved
```

**Why Disabled**:
- CUDA illegal memory access errors when enabled
- State corruption between prefill and decode phases
- Requires fundamental architectural changes to fix

## What Would Be Required to Fix

To make interleaving work properly, you would need:

1. **GPU Synchronization**:
   ```python
   if prefill_seqs:
       model_runner.call("run", prefill_seqs, True)
       torch.cuda.synchronize()  # Wait for prefill to complete
   
   if decode_seqs:
       model_runner.call("run", decode_seqs, False)
   ```

2. **Context Isolation**: 
   - Separate context objects for prefill and decode
   - Or use thread-local storage
   - Or pass context as parameters instead of global

3. **Memory Isolation**:
   - Separate CUDA streams for prefill and decode
   - Or ensure exclusive access with synchronization
   - Or separate KV cache regions

4. **API Compatibility**:
   - Ensure Flash Attention state is properly isolated
   - Or use separate Flash Attention contexts
   - Or synchronize between Flash Attention calls

## Lessons Learned

1. **Async CUDA Operations**: Always synchronize GPU operations when shared state is involved
2. **Global State**: Global variables are problematic for concurrent execution
3. **CUDA Graphs**: Require exclusive access during replay - cannot mix with eager execution
4. **Flash Attention**: Different APIs have different memory requirements and state
5. **Testing**: Need to test with actual GPU workloads to catch these issues

## Performance Impact

**Baseline (Sequential)**: ~1,750 tok/s
**With Interleaving Disabled**: ~1,708 tok/s (slight regression due to code path changes)
**With Interleaving Enabled**: ❌ Crashes with CUDA errors

**Conclusion**: The sequential approach (prefill OR decode per step) is simpler and avoids these concurrency issues.

## Files Modified

1. `config.py` - Added interleaving configuration (disabled)
2. `engine/types.py` - Updated scheduler interface (reverted)
3. `engine/scheduler.py` - Implemented dual scheduling (reverted)
4. `engine/llm_engine.py` - Added dual execution support (reverted)
5. `memory/block_manager.py` - Fixed `may_append` logic
6. `layers/attention.py` - Added empty slot_mapping check
7. `engine/model_runner.py` - Added context reset calls

## Branch Information

**Branch**: `feature/prefill-decode-interleaving`
**Status**: Implementation complete but disabled
**Recommendation**: Keep for reference but do not enable in production

---

**Date**: November 2025
**Status**: Experimental - Not Production Ready

