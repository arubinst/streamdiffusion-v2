FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Clone StreamDiffusionV2
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Install PyTorch
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install common dependencies
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    xformers \
    omegaconf \
    einops \
    imageio \
    imageio-ffmpeg \
    opencv-python-headless \
    numpy \
    pillow \
    gradio \
    huggingface_hub

# Try requirements.txt (continue even if it fails)
RUN pip install --no-cache-dir -r requirements.txt || echo "Some requirements failed, continuing..."

# Install the package
RUN python setup.py develop || python setup.py install

# Create model directories
RUN mkdir -p /app/wan_models /app/ckpts

# Create startup script that downloads models
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Checking for models..."\n\
if [ ! -d "/app/wan_models/Wan2.1-T2V-1.3B" ]; then\n\
    echo "Downloading models..."\n\
    huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B \\\n\
        --local-dir /app/wan_models/Wan2.1-T2V-1.3B\n\
fi\n\
if [ ! -d "/app/ckpts/wan_causal_dmd_v2v" ]; then\n\
    huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 \\\n\
        --local-dir /app/ckpts --include "wan_causal_dmd_v2v/*"\n\
fi\n\
echo "Models ready!"\n\
cd /app/demo\n\
python app.py --server_name 0.0.0.0 --server_port 7860\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
