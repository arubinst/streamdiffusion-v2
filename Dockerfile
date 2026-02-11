FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Node.js and git
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone repo
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Pin NumPy to 1.x (critical fix for PyTorch compatibility)
RUN pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        xformers \
        markdown2 \
        easydict \
        "huggingface_hub[cli]" && \
    python setup.py develop

# Build frontend
RUN cd demo/frontend && npm install && npm run build && npm cache clean --force

# Clean up
RUN pip cache purge && rm -rf /root/.cache /tmp/*

# Create entrypoint
RUN cat > /app/entrypoint.sh << 'EOF'
#!/bin/bash
set -e
echo "=== Starting StreamDiffusionV2 ==="
python3 << 'PYEOF'
import os
from huggingface_hub import snapshot_download

model_path = "/app/wan_models/Wan2.1-T2V-1.3B"
if not os.path.exists(model_path):
    print("Downloading model (10-15 min)...")
    snapshot_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        local_dir=model_path,
        resume_download=True
    )
    print("Model downloaded!")
else:
    print("Model already exists")

ckpt_path = "/app/ckpts/wan_causal_dmd_v2v"
if not os.path.exists(ckpt_path):
    print("Downloading checkpoint...")
    snapshot_download(
        repo_id="jerryfeng/StreamDiffusionV2",
        local_dir="/app/ckpts",
        allow_patterns=["wan_causal_dmd_v2v/*"],
        resume_download=True
    )
    print("Checkpoint downloaded!")
else:
    print("Checkpoint already exists")
PYEOF

echo "Starting demo on 0.0.0.0:7860..."
cd /app/demo
exec python main.py \
    --port 7860 \
    --host 0.0.0.0 \
    --num_gpus ${NUM_GPUS:-1} \
    --gpu_ids ${GPU_IDS:-0} \
    --step ${STEP:-2}
EOF

RUN chmod +x /app/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/entrypoint.sh"]
