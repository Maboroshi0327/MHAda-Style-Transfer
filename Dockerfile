FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

# Install Miniconda
RUN apt update && apt upgrade -y && apt install -y wget \
    && mkdir -p /root/miniconda3 \
    && wget -c --tries=10 https://repo.anaconda.com/miniconda/Miniconda3-py312_25.3.1-1-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh \
    && bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 \
    && rm /root/miniconda3/miniconda.sh \
    && source /root/miniconda3/bin/activate \
    && conda init --all

# Install PyTorch
WORKDIR /root/miniconda3/bin
RUN ./conda install -y python=3.12 \
    && ./pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN ./pip install scipy==1.15.1 opencv-contrib-python==4.11.0.86 seaborn==0.13.2 imageio==2.37.0 imageio-ffmpeg==0.6.0

# Create directories
RUN mkdir -p /root/project \
    && mkdir -p /root/datasets
WORKDIR /root/project
