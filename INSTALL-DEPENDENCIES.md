# Installation Guide

This document provides a clean setup guide for running the test case described in the README on Ubuntu 24.04 (Noble) with NVIDIA GPU support. It covers host prerequisites, GPU drivers, container tooling, Docker Compose with GPU support, and developer dependencies.

## Hardware Requirements
- One or more NVIDIA GPUs with at least Turing architecture (Ampere or higher recommended).
- At best, 3-4 times the RAM of the available VRAM for optimal performance.

## Software Basis
The test case is based on Docker Compose for orchestrating multiple vLLM inference servers with CUDA checkpointing, enabling efficient model switching without full reloads.

> Requires: sudo privileges and a machine with a supported NVIDIA GPU.

## 1) Base packages and Docker

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Log out/in (or `newgrp docker`) so the Docker group is effective.

## 2) NVIDIA driver + CUDA Toolkit 13

Clean old drivers, install current driver + toolkit, and disable nouveau:

```bash
sudo apt remove --purge "*cuda*" "*nvidia*"
sudo apt autoremove -y

sudo apt update && sudo apt full-upgrade -y
sudo apt install -y cmake build-essential linux-headers-$(uname -r)

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

sudo apt install -y cuda-drivers cuda-toolkit-13-0

echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo apt install -y nvtop pciutils screen curl git-lfs jq
sudo reboot
```

After reboot verify the GPU:

```bash
nvidia-smi
```

Make the CUDA toolchain available on PATH and LD_LIBRARY_PATH:

```bash
sudo ln -s /usr/local/cuda-13.0 /usr/local/cuda

sudo tee /etc/profile.d/cuda.sh >/dev/null <<'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

sudo chmod +x /etc/profile.d/cuda.sh
sudo reboot
```

Validate:

```bash
which nvcc
nvcc --version
```

## 3) NVIDIA Container Toolkit

Install the runtime so containerd/Docker can launch GPU workloads:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 4) CUDA checkpoint utility

```bash
cd ~
git clone https://github.com/NVIDIA/cuda-checkpoint.git
cd cuda-checkpoint
sudo cp bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/
sudo chmod +x /usr/local/bin/cuda-checkpoint
cuda-checkpoint --help
```

## 5) CRIU (checkpoint/restore)

Build CRIU from source to match the kernel:

```bash
sudo apt update
sudo apt install -y build-essential git pkg-config python3 protobuf-c-compiler libprotobuf-c-dev protobuf-compiler \
  libprotobuf-dev libnl-3-dev libnl-route-3-dev libcap-dev libaio-dev libseccomp-dev libpixman-1-dev asciidoc xmlto \
  libnftnl-dev libdrm-dev libjson-c-dev libc6-dev libbsd-dev gcc make kmod libsystemd-dev libnet1-dev \
  libgnutls28-dev libnftables-dev python3-protobuf uuid-dev python3-yaml

cd ~
git clone https://github.com/checkpoint-restore/criu.git
cd criu

git fetch --tags
LATEST_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))
git checkout $LATEST_TAG

make clean
make -j$(nproc)
sudo make install
sudo ldconfig

criu --version
sudo criu check --all
```