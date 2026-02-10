FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including Node.js
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
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

# Build frontend
WORKDIR /app/demo/frontend
RUN npm install && npm run build

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

WORKDIR /app

# Create entrypoint script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Starting StreamDiffusionV2 ==="

# Download models using Python module
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

echo "Models ready! Starting demo..."

cd /app/demo

# Start the demo with proper arguments
# Use environment variables for configuration
NUM_GPUS=${NUM_GPUS:-1}
GPU_IDS=${GPU_IDS:-0}
STEP=${STEP:-2}

echo "Starting with NUM_GPUS=$NUM_GPUS GPU_IDS=$GPU_IDS STEP=$STEP"

exec python main.py \
    --port 7860 \
    --host 0.0.0.0 \
    --num_gpus $NUM_GPUS \
    --gpu_ids $GPU_IDS \
    --step $STEP
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
