# ============================================================================
# Stage 1: Build frontend
# ============================================================================
FROM node:18-slim AS frontend-builder

WORKDIR /build

# Clone repo just to get frontend
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/chenfengxu714/StreamDiffusionV2.git . && \
    cd demo/frontend && \
    npm install && \
    npm run build

# ============================================================================
# Stage 2: Final runtime image
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies (no build tools!)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

# Clone StreamDiffusionV2
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Copy pre-built frontend from builder stage
COPY --from=frontend-builder /build/demo/frontend/dist /app/demo/frontend/dist

# Install PyTorch (smaller runtime version)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install pre-built xformers wheel
RUN pip install --no-cache-dir xformers==0.0.22.post7 --no-deps

# Install remaining dependencies
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

# Clean up to save space
RUN pip cache purge && \
    rm -rf /root/.cache && \
    apt-get clean

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

# Entrypoint script
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
