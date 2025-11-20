#!/bin/bash
# Deployment script to transfer code to GCP instance and run setup

set -e

INSTANCE_NAME="instance-20251106-20251113-20251120-20251120-191939"
ZONE="${GCP_ZONE:-us-west1-a}"  # Default zone, can be overridden
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Deploying allmos_v2 to GCP Instance"
echo "=========================================="
echo "Instance: $INSTANCE_NAME"
echo "Zone: $ZONE"
echo "Project Directory: $PROJECT_DIR"
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if instance exists and is accessible
echo "Checking instance status..."
if ! gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" &> /dev/null; then
    echo "❌ Error: Instance $INSTANCE_NAME not found in zone $ZONE"
    echo "Please verify the instance name and zone"
    exit 1
fi

# Get instance status
INSTANCE_STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$ZONE" --format="value(status)")
echo "Instance status: $INSTANCE_STATUS"

if [ "$INSTANCE_STATUS" != "RUNNING" ]; then
    echo "Starting instance..."
    gcloud compute instances start "$INSTANCE_NAME" --zone="$ZONE"
    echo "Waiting for instance to be ready..."
    sleep 10
fi

# Create a temporary tarball excluding unnecessary files
echo ""
echo "Creating deployment package..."
TEMP_DIR=$(mktemp -d)
TARBALL="$TEMP_DIR/allmos_v2_deploy.tar.gz"

cd "$PROJECT_DIR"
tar --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.tar.gz' \
    -czf "$TARBALL" .

# Transfer tarball to instance
echo "Transferring code to instance..."
gcloud compute scp \
    "$TARBALL" \
    "$INSTANCE_NAME:~/allmos_v2_deploy.tar.gz" \
    --zone="$ZONE"

if [ $? -ne 0 ]; then
    echo "❌ Error transferring code"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Extract on remote instance
echo "Extracting code on remote instance..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    mkdir -p ~/allmos_v2 && \
    cd ~/allmos_v2 && \
    tar -xzf ~/allmos_v2_deploy.tar.gz && \
    rm ~/allmos_v2_deploy.tar.gz
"

if [ $? -eq 0 ]; then
    echo "✅ Code transferred successfully"
else
    echo "❌ Error extracting code"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Clean up local tarball
rm -rf "$TEMP_DIR"

# Make setup script executable and run it
echo ""
echo "Running setup on remote instance..."
gcloud compute ssh "$INSTANCE_NAME" --zone="$ZONE" --command="
    cd ~/allmos_v2 && \
    chmod +x setup_gcp_vm.sh && \
    bash setup_gcp_vm.sh
"

if [ $? -eq 0 ]; then
    echo "✅ Setup completed successfully"
else
    echo "❌ Error during setup"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Deployment complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. SSH to instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "2. Run benchmarks: cd ~/allmos_v2 && python bench.py"
echo ""
echo "Or run benchmarks remotely:"
echo "  bash run_benchmarks.sh"

