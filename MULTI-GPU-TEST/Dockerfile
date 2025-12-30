# MULTI-GPU-TEST/Dockerfile
# Standalone image: vLLM OpenAI server + cuda-checkpoint helper

ARG VLLM_VERSION=v0.11.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cuda-checkpoint binary
RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint \
    && cp /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/ \
    && chmod +x /usr/local/bin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint

# Helper to toggle *all* CUDA-controllable vLLM worker processes
COPY scripts/cuda-toggle-all.sh /usr/local/bin/cuda-toggle-all
RUN chmod +x /usr/local/bin/cuda-toggle-all
