FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PYTHONUNBUFFERED=1

# [WandB Optimization] 
ENV WANDB_PROJECT="REFLEX-Project"
ENV WANDB_DIR="/workspace/wandb"
ENV WANDB_CACHE_DIR="/workspace/.cache/wandb"

# Export MuJoCo and NVIDIA paths for mujoco-py rendering
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin:/usr/lib/nvidia

# Install system dependencies
# Combined GUI, OpenGL, OpenCV, and C++ Build tools (cmake, build-essential)
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libgl1-mesa-glx libosmesa6-dev libglew-dev \
    libglfw3 libglfw3-dev libgles2-mesa-dev \
    libglib2.0-0 libxrender1 libxext6 \
    git wget curl patchelf cmake build-essential x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Download and install MuJoCo 2.1.0 (Required for robomimic image extraction)
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xzf mujoco.tar.gz -C /root/.mujoco/ \
    && rm mujoco.tar.gz

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy and install requirements
COPY requirements.txt .

# Install dependencies (cython must be installed first for mujoco-py build)
RUN pip3 install "cython<3.0.0" \
    && pip3 install -r requirements.txt

# Default command
CMD ["/bin/bash"]
