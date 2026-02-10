FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during build
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
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Clone StreamDiffusionV2
RUN git clone https://github.com/chenfengxu714/StreamDiffusionV2.git /app

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install StreamDiffusionV2
RUN python setup.py develop

# Download models (1.3B - faster for demo)
RUN pip install --no-cache-dir huggingface_hub && \
    huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir /app/wan_models/Wan2.1-T2V-1.3B && \
    huggingface-cli download --resume-download jerryfeng/StreamDiffusionV2 \
        --local-dir /app/ckpts --include "wan_causal_dmd_v2v/*"

# Expose port for web demo
EXPOSE 7860

WORKDIR /app/demo
CMD ["python", "app.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
