# vllm/Dockerfile

# vLLM-Version beim Build steuerbar
ARG VLLM_VERSION=v0.10.2
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Tools zum Bauen von cuda-checkpoint
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# cuda-checkpoint bauen und installieren
RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint \
    && cd /tmp/cuda-checkpoint \
    && cp bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/ \
    && chmod +x /usr/local/bin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint

WORKDIR /app

# Python-Dependencies für den Checkpoint-Service
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Checkpoint-Service + Entrypoint
COPY checkpoint_service.py /app/checkpoint_service.py
COPY entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

ENV CHECKPOINT_SERVICE_PORT=9000

EXPOSE 8000 9000

ENTRYPOINT ["/app/entrypoint.sh"]