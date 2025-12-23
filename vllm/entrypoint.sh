#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_SERVICE_PORT:=9000}"

echo "[entrypoint] Starting vLLM server with args: $*"

# WICHTIG: python3 statt python
python3 -m vllm.entrypoints.openai.api_server "$@" &
VLLM_PID=$!
echo "${VLLM_PID}" > /tmp/vllm.pid
echo "[entrypoint] vLLM PID is ${VLLM_PID}"

echo "[entrypoint] Starting Checkpoint Service on port ${CHECKPOINT_SERVICE_PORT}..."
exec uvicorn checkpoint_service:app \
  --host 0.0.0.0 \
  --port "${CHECKPOINT_SERVICE_PORT}"