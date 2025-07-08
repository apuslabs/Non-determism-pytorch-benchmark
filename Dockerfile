FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV CUDA_LAUNCH_BLOCKING=1
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
ENV PYTHONHASHSEED=0

WORKDIR /workspace

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Copy requirements file with exact versions
COPY requirements.txt .

# Install dependencies from fixed requirements
RUN python3 -m pip install -r requirements.txt

# Copy the rest of the application
COPY . .