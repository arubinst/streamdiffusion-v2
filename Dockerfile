FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Node.js 18 and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set Python as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Clone repo
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install xformers
RUN pip install --no-cache-dir xformers==0.0.22.post7 --no-deps

# Install Python dependencies
RUN pip install --no-cache-dir \
    "huggingface_hub[cli]" \
    diffusers \
    transformers \
    accelerate \
    omegaconf \
    einops \
    opencv-python-headless \
    numpy \
    Pillow \
    gradio && \
    pip install --no-cache-dir -r requirements.txt || true && \
    python setup.py develop || true

# Build frontend
WORKDIR /app/demo/frontend
RUN npm install && npm run build

# Clean up to save space
RUN pip cache purge && \
    npm cache clean --force && \
    rm -rf /root/.cache /tmp/* /var/tmp/*

WORKDIR /app

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

# Entrypoint
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e
echo "=== Starting StreamDiffusionV2 ==="
python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download
model_path = "/app/wan_models/Wan2.1-T2V-1.3B"
if not os.path.exists(model_path):
    print("Downloading model...")
    snapshot_download(repo_id="Wan-AI/Wan2.1-T2V-1.3B", local_dir=model_path, resume_download=True)
ckpt_path = "/app/ckpts/wan_causal_dmd_v2v"
if not os.path.exists(ckpt_path):
    print("Downloading checkpoint...")
    snapshot_download(repo_id="jerryfeng/StreamDiffusionV2", local_dir="/app/ckpts", allow_patterns=["wan_causal_dmd_v2v/*"], resume_download=True)
PYEOF
echo "Starting demo..."
cd /app/demo
exec python main.py --port 7860 --host 0.0.0.0 --num_gpus ${NUM_GPUS:-1} --gpu_ids ${GPU_IDS:-0} --step ${STEP:-2}
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
