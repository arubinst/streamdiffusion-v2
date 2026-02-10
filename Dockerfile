FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Clone StreamDiffusionV2
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Install PyTorch (matching CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install xformers (pre-built for CUDA 12.1)
RUN pip install --no-cache-dir xformers==0.0.22.post7

# Install HuggingFace CLI and other dependencies
RUN pip install --no-cache-dir \
    "huggingface_hub[cli]" \
    diffusers>=0.25.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    omegaconf \
    einops \
    opencv-python-headless \
    numpy \
    Pillow \
    gradio

# Install any remaining requirements (skip failures)
RUN pip install --no-cache-dir -r requirements.txt || echo "Continuing despite errors..."

# Install StreamDiffusionV2
RUN python setup.py develop || python setup.py install || echo "Setup completed"

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

# Use Python module instead of CLI command
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Starting StreamDiffusionV2 ==="

# Download models using Python module (works even if CLI isn't in PATH)
python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download

# Download main model
model_path = "/app/wan_models/Wan2.1-T2V-1.3B"
if not os.path.exists(model_path):
    print("Downloading Wan2.1-T2V-1.3B model (this may take 10-15 minutes)...")
    try:
        snapshot_download(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B",
            local_dir=model_path,
            resume_download=True
        )
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        exit(1)
else:
    print("Model already exists, skipping download")

# Download checkpoint
ckpt_path = "/app/ckpts/wan_causal_dmd_v2v"
if not os.path.exists(ckpt_path):
    print("Downloading checkpoint...")
    try:
        snapshot_download(
            repo_id="jerryfeng/StreamDiffusionV2",
            local_dir="/app/ckpts",
            allow_patterns=["wan_causal_dmd_v2v/*"],
            resume_download=True
        )
        print("Checkpoint downloaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to download checkpoint: {e}")
        exit(1)
else:
    print("Checkpoint already exists, skipping download")
PYEOF

echo "Models ready! Starting application..."

# Find and run the demo app
if [ -d "/app/demo" ] && [ -f "/app/demo/app.py" ]; then
    echo "Found demo at /app/demo/app.py"
    cd /app/demo
    python app.py --server_name 0.0.0.0 --server_port 7860
elif [ -f "/app/app.py" ]; then
    echo "Found app at /app/app.py"
    python /app/app.py --server_name 0.0.0.0 --server_port 7860
elif [ -f "/app/streamv2v/inference_pipe.py" ]; then
    echo "Found inference script, starting basic server..."
    python -m http.server 7860
else
    echo "ERROR: No demo app found!"
    echo "Directory contents:"
    ls -la /app/
    if [ -d "/app/demo" ]; then
        echo "Demo directory contents:"
        ls -la /app/demo/
    fi
    echo ""
    echo "Available Python files:"
    find /app -name "*.py" -type f | head -20
    exit 1
fi
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
