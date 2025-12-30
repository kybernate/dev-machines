# Copy/paste curl commands (vLLM @ :8010)

## 0) Start the vLLM container (Docker Compose)

From the repo root:

```bash
mkdir -p ../.cache
docker compose up -d --build
```

Optional: follow logs while it loads the model:

```bash
docker compose logs -f vllm-qwen3-8b-awq
```

## 1) Ask the model its name (chat completion)

Copy/paste this as-is:

```bash
curl -sS http://localhost:8010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b-awq",
    "messages": [
      {"role": "user", "content": "What is your model name? Reply with only the model name."}
    ],
    "temperature": 0.0,
    "max_tokens": 32,
    "stream": false
  }'
```

## 2) (Optional) List served model IDs

```bash
curl -sS http://localhost:8010/v1/models
```

## 3) Checkpoint suspend (CUDA -> checkpointed)

This toggles all CUDA-controllable vLLM worker PIDs to the `checkpointed` state:

```bash
docker exec -t vllm-qwen3-8b-awq cuda-toggle-all checkpointed
```

## 4) Checkpoint resume (CUDA -> running)

This toggles all CUDA-controllable vLLM worker PIDs back to the `running` state:

```bash
docker exec -t vllm-qwen3-8b-awq cuda-toggle-all running
```

Notes:
- Don’t add a trailing `]` at the end of the command.
- If you want pretty output and have `jq` installed: append `| jq` to either command.
