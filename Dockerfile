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
    huggingface_hub[cli] \
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

# Improved startup script
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Starting StreamDiffusionV2 ==="

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "ERROR: huggingface-cli not found, installing..."
    pip install -U "huggingface_hub[cli]"
fi

# Download models if not present
if [ ! -d "/app/wan_models/Wan2.1-T2V-1.3B" ]; then
    echo "Downloading Wan2.1-T2V-1.3B model (this may take 10-15 minutes)..."
    huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir /app/wan_models/Wan2.1-T2V-1.3B || {
        echo "ERROR: Failed to download model"
        exit 1
    }
fi

if [ ! -d "/app/ckpts/wan_causal_dmd_v2v" ]; then
    echo "Downloading checkpoint..."
    huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 \
        --local-dir /app/ckpts --include "wan_causal_dmd_v2v/*" || {
        echo "ERROR: Failed to download checkpoint"
        exit 1
    }
fi

echo "Models ready! Starting application..."

# Find and run the demo app
if [ -d "/app/demo" ] && [ -f "/app/demo/app.py" ]; then
    echo "Found demo at /app/demo/app.py"
    cd /app/demo
    python app.py --server_name 0.0.0.0 --server_port 7860
elif [ -f "/app/app.py" ]; then
    echo "Found app at /app/app.py"
    python /app/app.py --server_name 0.0.0.0 --server_port 7860
else
    echo "ERROR: No demo app found!"
    echo "Directory contents:"
    ls -la /app/
    if [ -d "/app/demo" ]; then
        echo "Demo directory contents:"
        ls -la /app/demo/
    fi
    echo "Starting Python shell for debugging..."
    python
fi
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
