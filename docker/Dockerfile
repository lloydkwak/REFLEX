FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libgl1-mesa-glx libosmesa6-dev libglew-dev \
    libglfw3 libgles2-mesa-dev \
    git wget x11-apps \
    && rm -rf /var/lib/apt/lists/*
    libglib2.0-0 \
    libgl1-mesa-glx \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Default command
CMD ["/bin/bash"]
