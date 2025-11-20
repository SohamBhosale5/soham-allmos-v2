#!/bin/bash
# Setup script for GCP VM instance
# This script installs all dependencies and prepares the environment

set -e

echo "=========================================="
echo "Setting up allmos_v2 on GCP VM"
echo "=========================================="

# Update system packages
echo ""
echo "Updating system packages..."
sudo apt-get update -qq

# Install Python 3.10+ if not present
if ! command -v python3 &> /dev/null || ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Installing Python 3.10+..."
    sudo apt-get install -y python3.10 python3.10-venv python3-pip
fi

# Check for NVIDIA GPU
echo ""
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA drivers detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  NVIDIA drivers not found. Installing..."
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "⚠️  Please reboot if drivers were just installed: sudo reboot"
fi

# Install CUDA toolkit if not present
echo ""
echo "Checking for CUDA toolkit..."
if ! command -v nvcc &> /dev/null; then
    echo "Attempting to install CUDA toolkit..."
    # Try different package names
    if sudo apt-get install -y cuda-toolkit-12-1 2>/dev/null; then
        CUDA_HOME=/usr/local/cuda-12.1
    elif sudo apt-get install -y cuda-toolkit-12 2>/dev/null; then
        CUDA_HOME=/usr/local/cuda-12
    elif sudo apt-get install -y cuda-toolkit 2>/dev/null; then
        # Find CUDA installation
        CUDA_HOME=$(ls -d /usr/local/cuda-* 2>/dev/null | head -n 1 || echo "/usr/local/cuda")
    else
        echo "⚠️  CUDA toolkit installation failed. PyTorch will use bundled CUDA libraries."
        echo "   This is acceptable for most use cases."
        CUDA_HOME=""
    fi
    
    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        # Set CUDA environment variables
        export CUDA_HOME
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        
        # Add to bashrc for persistence
        if ! grep -q "CUDA_HOME" ~/.bashrc; then
            echo "" >> ~/.bashrc
            echo "# CUDA Environment" >> ~/.bashrc
            echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
            echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
            echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        fi
        echo "✅ CUDA toolkit installed at $CUDA_HOME"
    fi
else
    echo "✅ CUDA toolkit found:"
    nvcc --version | head -n 1
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    export CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
fi

# Install build dependencies
echo ""
echo "Installing build dependencies..."
sudo apt-get install -y build-essential git

# Check Python version
echo ""
echo "Python version:"
python3 --version

# Install pip dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing core dependencies..."
pip3 install torch>=2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers>=4.51.0 xxhash>=3.0.0 safetensors>=0.4.0 tqdm>=4.65.0 numpy>=1.26.0

# Try to install flash-attn (optional, may fail on some systems)
echo ""
echo "Attempting to install flash-attention (optional)..."
if pip3 install flash-attn --no-build-isolation 2>&1 | tee /tmp/flash_attn_install.log; then
    echo "✅ flash-attention installed successfully"
else
    echo "⚠️  flash-attention installation failed (will use fallback)"
    echo "   This is acceptable - the system will use PyTorch fallback"
fi

# Try to install triton (optional)
echo ""
echo "Attempting to install triton (optional)..."
if pip3 install triton>=3.0.0 2>&1 | tee /tmp/triton_install.log; then
    echo "✅ triton installed successfully"
else
    echo "⚠️  triton installation failed (will use fallback)"
fi

# Check GLIBC version
echo ""
echo "Checking GLIBC version..."
if command -v ldd &> /dev/null; then
    GLIBC_VERSION=$(ldd --version | head -n 1 | grep -oP '[0-9]+\.[0-9]+')
    echo "GLIBC version: $GLIBC_VERSION"
    if python3 -c "major, minor = map(int, '$GLIBC_VERSION'.split('.')); exit(0 if major > 2 or (major == 2 and minor >= 32) else 1)" 2>/dev/null; then
        echo "✅ GLIBC >= 2.32 (flash-attn compatible)"
    else
        echo "⚠️  GLIBC < 2.32 (flash-attn may not work, will use fallback)"
    fi
fi

# Verify PyTorch can see GPU
echo ""
echo "Verifying PyTorch CUDA support..."
python3 << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
else:
    print("⚠️  CUDA not available - check NVIDIA drivers")
EOF

# Check if model exists, if not provide instructions
echo ""
echo "Checking for model files..."
MODEL_PATH="$HOME/huggingface/Qwen3-0.6B"
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
    echo "✅ Model found at $MODEL_PATH"
else
    echo "⚠️  Model not found at $MODEL_PATH"
    echo ""
    echo "To download the model, run:"
    echo "  pip3 install huggingface-hub"
    echo "  huggingface-cli download Qwen/Qwen3-0.6B --local-dir $MODEL_PATH"
    echo ""
    echo "Or set a different model path in bench.py"
fi

# Run system check
echo ""
echo "Running system check..."
if [ -f "check_system.py" ]; then
    python3 check_system.py || echo "⚠️  Some checks failed, but continuing..."
fi

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download model if needed (see above)"
echo "2. Run system check: python3 check_system.py"
echo "3. Run benchmarks: python3 bench.py"
echo ""

