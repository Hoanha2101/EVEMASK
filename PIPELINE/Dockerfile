# Use NVIDIA's official TensorRT 8.6.1 image with CUDA 11.8 and Python 3.9
FROM nvcr.io/nvidia/tensorrt:22.12-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev \
    zlib1g-dev \
    libgdbm-compat-dev \
    libnss3-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    # libqtgui4 \
    # libqtwebkit4 \
    # libqt4-test \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-cffi \
    libcairo2-dev \
    libpango1.0-dev \
    libglib2.0-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio \
    ffmpeg \
    libsm6 \
    libxext6 \
    # Add network tools for debugging
    net-tools \
    iputils-ping \
    netcat \
    iptables \
    # Add multicast support
    iproute2 \
    # Add additional system libraries that were in requirements
    # Note: These are runtime libraries, not development packages
    libbz2-1.0 \
    libexpat1 \
    libffi7 \
    libssl1.1 \
    libsqlite3-0 \
    libtk8.6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Cleanup any files used by apt-get update to free up space
RUN apt-get clean

# Upgrade pip and setuptools
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 support
RUN pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other CUDA-related packages
RUN pip install pycuda==2024.1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-docker.txt .

# Install Python dependencies (excluding torch, torchvision, torchaudio, tensorrt, pycuda as they're already installed)
RUN pip install -r requirements-docker.txt

# Copy the entire project
COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p weights/yolo weights/pytorch weights/onnx weights/trtPlans recognizeData videos

# Set permissions
RUN chmod +x main.py monitor.py export.py getpytorch.py build.py

# Create network configuration script for Windows
RUN echo '#!/bin/bash\n\
# Network configuration for Windows Docker\n\
echo "Configuring network for Windows Docker..."\n\
\n\
# Enable multicast routing\n\
echo 1 > /proc/sys/net/ipv4/ip_forward\n\
\n\
# Add multicast routes if needed\n\
# route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0 2>/dev/null || true\n\
\n\
# Set network interface to promiscuous mode for multicast\n\
# ip link set eth0 promisc on 2>/dev/null || true\n\
\n\
echo "Network configuration completed"\n\
' > /app/configure_network.sh && chmod +x /app/configure_network.sh

# Expose ports for streaming (if needed)
EXPOSE 30122 30133

# Set default command
CMD ["python", "main.py"] 
