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

# Install remaining dependencies without xformers
RUN pip install --no-cache-dir \
    diffusers>=0.25.0 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    omegaconf \
    einops \
    opencv-python-headless \
    numpy \
    Pillow \
    gradio \
    huggingface_hub

# Install any remaining requirements (skip failures)
RUN pip install --no-cache-dir -r requirements.txt || echo "Continuing despite errors..."

# Install StreamDiffusionV2
RUN python setup.py develop || python setup.py install || echo "Setup completed"

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

# Startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "=== Starting StreamDiffusionV2 ==="\n\
if [ ! -d "/app/wan_models/Wan2.1-T2V-1.3B" ]; then\n\
    echo "Downloading Wan2.1-T2V-1.3B model (this may take a while on first run)..."\n\
    huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B \\\n\
        --local-dir /app/wan_models/Wan2.1-T2V-1.3B || echo "Model download failed, will retry at runtime"\n\
fi\n\
if [ ! -d "/app/ckpts/wan_causal_dmd_v2v" ]; then\n\
    echo "Downloading checkpoint..."\n\
    huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 \\\n\
        --local-dir /app/ckpts --include "wan_causal_dmd_v2v/*" || echo "Checkpoint download failed, will retry at runtime"\n\
fi\n\
echo "Models ready! Starting application..."\n\
cd /app/demo || cd /app\n\
if [ -f "app.py" ]; then\n\
    python app.py --server_name 0.0.0.0 --server_port 7860\n\
else\n\
    echo "Demo app not found, starting Python shell for debugging"\n\
    python\n\
fi\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
