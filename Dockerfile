FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV CUDA_LAUNCH_BLOCKING=1
ENV CUBLAS_WORKSPACE_CONFIG=:16:8
ENV PYTHONHASHSEED=0

WORKDIR /workspace

COPY . .

# Install uv package manager
RUN apt-get update && apt-get install -y python3 python3-pip curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    . $HOME/.local/bin/env

# Install dependencies strictly according to uv.lock
RUN $HOME/.local/bin/uv sync --frozen
