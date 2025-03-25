FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

RUN apt update && apt upgrade -y && apt install -y \
    # Install dependencies
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Create directories
    && mkdir -p ~/project \
    && mkdir -p ~/datasets \
    # Install Miniconda
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm ~/miniconda3/miniconda.sh \
    && source ~/miniconda3/bin/activate \
    && conda init --all \
    # Install PyTorch
    && conda install -y python=3.12 \
    && pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 \
    && pip install scipy==1.15.1 opencv-contrib-python==4.11.0.86
