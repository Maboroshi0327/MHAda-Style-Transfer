FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

# Install Miniconda
RUN apt update && apt upgrade -y && apt install -y wget \
    && mkdir -p /root/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh \
    && bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 \
    && rm /root/miniconda3/miniconda.sh \
    && source /root/miniconda3/bin/activate \
    && conda init --all

# Install PyTorch
WORKDIR /root/miniconda3/bin
RUN ./conda install -y python=3.12 \
    && ./pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install CUDA
WORKDIR /tmp
RUN apt update && apt install -y build-essential libxml2 \
    && wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run \
    && sh ./cuda_12.4.1_550.54.15_linux.run --silent --toolkit \
    && rm ./cuda_12.4.1_550.54.15_linux.run
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=""
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install dependencies
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python dependencies
WORKDIR /root/miniconda3/bin
RUN ./pip install scipy==1.15.1 opencv-contrib-python==4.11.0.86 seaborn==0.13.2 \
    tensorboardX==2.6.2.2 setproctitle==1.3.5 colorama==0.4.6 imageio==2.37.0

# Create directories
RUN mkdir -p /root/project \
    && mkdir -p /root/datasets
WORKDIR /root/project