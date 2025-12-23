# NVIDIA Runtime Image with CUDA + nvidia-smi
FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System packages incl. Python3 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-venv \
      curl \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink so that "python" exists
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]