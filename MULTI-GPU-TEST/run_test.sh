#!/usr/bin/env bash
set -euo pipefail

# End-to-end test:
# 1) docker compose up
# 2) wait for /v1/models
# 3) curl chat
# 4) cuda-checkpoint suspend (checkpointed)
# 5) cuda-checkpoint resume (running)
# 6) curl chat again

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="${ROOT_DIR}/MULTI-GPU-TEST"

COMPOSE=(docker compose -f "${TEST_DIR}/docker-compose.yml")
API_BASE="${API_BASE:-http://localhost:8010}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-qwen3-8b-awq}"
MODEL_NAME="${MODEL_NAME:-qwen3-8b-awq}"

WAIT_SECS="${WAIT_SECS:-3600}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

cleanup() {
  if [[ "${CLEANUP:-0}" == "1" ]]; then
    "${COMPOSE[@]}" down
  fi
}
trap cleanup EXIT

wait_ready() {
  local deadline=$(( $(date +%s) + WAIT_SECS ))
  local start_ts
  start_ts="$(date +%s)"
  local loops=0

  local models_tmp
  models_tmp="$(mktemp -t vllm_models.XXXXXX.json)"
  trap 'rm -f "${models_tmp}"' RETURN

  while true; do
    loops=$((loops + 1))

    # Fail fast if the container crashed during startup (e.g., OOM).
    if (( loops % 5 == 0 )); then
      status="$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
      if [[ -n "${status}" && "${status}" != "running" ]]; then
        echo "[error] container '${CONTAINER_NAME}' is not running (status=${status})" >&2
        "${COMPOSE[@]}" logs --tail 200 "${CONTAINER_NAME}" 2>/dev/null || true
        return 2
      fi
    fi

    if curl -sf "${API_BASE}/v1/models" > "${models_tmp}"; then
      if python3 - <<PY
import json,sys
p="${models_tmp}"
obj=json.load(open(p))
ids=[m.get('id') for m in obj.get('data',[]) if isinstance(m,dict)]
print("\n".join(ids))
if "${MODEL_NAME}" in ids:
    sys.exit(0)
sys.exit(1)
PY
      then
        echo "[ok] vLLM is ready and model '${MODEL_NAME}' is listed"
        return 0
      fi
    fi

    # Periodic status so it's obvious we're making progress.
    if (( loops % 15 == 0 )); then
      local now_ts elapsed
      now_ts="$(date +%s)"
      elapsed=$(( now_ts - start_ts ))
      echo "[wait] still starting... elapsed=${elapsed}s (timeout=${WAIT_SECS}s)"
      "${COMPOSE[@]}" logs --tail 25 "${CONTAINER_NAME}" 2>/dev/null || true
    fi

    if (( $(date +%s) >= deadline )); then
      echo "[error] timed out waiting for vLLM readiness at ${API_BASE}" >&2
      return 1
    fi
    sleep "${POLL_INTERVAL}"
  done
}

chat_once() {
  local prompt="$1"
  curl -sf "${API_BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(python3 - <<PY
import json
req={
  'model': '${MODEL_NAME}',
  'messages': [{'role':'user','content': '''${prompt}'''}],
  'temperature': 0.2,
  'max_tokens': 64,
  'stream': False,
}
print(json.dumps(req))
PY
)" \
    > /tmp/vllm_chat.json

  python3 - <<PY
import json
obj=json.load(open('/tmp/vllm_chat.json'))
choices=obj.get('choices')
assert isinstance(choices, list) and len(choices) > 0, obj
msg=choices[0].get('message',{})
content=msg.get('content','')
print(content)
PY
}

echo "[step] starting vLLM container"
"${COMPOSE[@]}" up --build -d

echo "[step] waiting for readiness"
wait_ready

echo "[step] curl test (before suspend)"
chat_once "Say hello in one short sentence."

echo "[step] suspend (checkpoint) all CUDA worker PIDs"
docker exec -t "${CONTAINER_NAME}" cuda-toggle-all checkpointed

echo "[step] resume all CUDA worker PIDs"
docker exec -t "${CONTAINER_NAME}" cuda-toggle-all running

echo "[step] curl test (after resume)"
chat_once "Confirm you are running after resume in one sentence."

echo "[ok] multi-gpu suspend/resume test completed"
