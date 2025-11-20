#!/bin/bash
# Script to run benchmarks on GCP instance remotely

set -e

INSTANCE_NAME="instance-20251106-20251113-20251120-20251120-191939"
ZONE="${GCP_ZONE:-us-west1-a}"  # Default zone, can be overridden

echo "=========================================="
echo "Running Benchmarks on GCP Instance"
echo "=========================================="
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Ensure instance is running
INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$INSTANCE_STATUS" = "NOT_FOUND" ]; then
    echo "❌ Error: Instance $INSTANCE_NAME not found in zone $ZONE"
    exit 1
fi

if [ "$INSTANCE_STATUS" != "RUNNING" ]; then
    echo "Starting instance..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE"
    echo "Waiting for instance to be ready..."
    sleep 15
fi

# Run benchmarks
echo ""
echo "Running benchmarks..."
echo "This may take several minutes..."
echo ""

gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    cd ~/allmos_v2 && \
    echo 'Current directory:' && pwd && \
    echo 'Python version:' && python3 --version && \
    echo 'CUDA check:' && python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")' && \
    echo '' && \
    echo 'Starting benchmark...' && \
    python3 bench.py
"

BENCHMARK_EXIT_CODE=$?

echo ""
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ Benchmarks completed successfully!"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Benchmarks failed with exit code $BENCHMARK_EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if model is downloaded: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='ls -la ~/huggingface/Qwen3-0.6B/'"
    echo "2. Run system check: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='cd ~/allmos_v2 && python3 check_system.py'"
    echo "3. Check GPU: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command='nvidia-smi'"
fi

exit $BENCHMARK_EXIT_CODE

