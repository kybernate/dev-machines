# MULTI-GPU-TEST — Standalone vLLM + CUDA Checkpoint Suspend/Resume (2 GPUs)

This folder describes a **standalone** multi‑GPU test that:
1. spins up a single vLLM container via Docker Compose,
2. loads **Qwen3 8B AWQ**,
3. stores all model/cache data in the project root **`.cache/`** directory,
4. validates inference with `curl`,
5. **suspends** the running model via **`cuda-checkpoint`** (toggling *all* CUDA worker PIDs, i.e., covering both GPUs),
6. **resumes** the model,
7. validates inference again with `curl`.

The test can borrow ideas from the repository (e.g. how `cuda-checkpoint` is installed in [`vllm/Dockerfile`](../vllm/Dockerfile)), but it has **no integration** with the controller/frontend stack (no dependency on [`controller/app.py`](../controller/app.py), no dashboard, no checkpoint service). The **only shared integration** is the root `.cache/` volume.

---

## Prerequisites (host)

- Docker + Docker Compose
- NVIDIA Driver + NVIDIA Container Toolkit (so containers can access GPUs)
- 2 GPUs visible to Docker (`nvidia-smi` works on host)
- Enough VRAM/RAM for Qwen3 8B AWQ on tensor parallel 2
- Project root `.cache/` directory (this repo creates it at `../.cache/`)

If it doesn’t exist yet:

```sh
mkdir -p ../.cache
```

---

## Task 1 — Create a docker compose file with a vLLM container

Create `MULTI-GPU-TEST/docker-compose.yml` with **one** service. This example exposes vLLM on `http://localhost:8010`.

```yml
services:
  vllm-qwen3-8b-awq:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        VLLM_VERSION: ${VLLM_VERSION:-v0.11.0}
    image: vllm-qwen3-8b-awq:local
    container_name: vllm-qwen3-8b-awq
    ipc: host
    gpus: all
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    environment:
      - HF_HOME=/root/.cache/huggingface
      - TRANSFORMERS_CACHE=/root/.cache/huggingface
      - XDG_CACHE_HOME=/root/.cache
      # cuda-checkpoint restore currently does not support CUDA IPC/UVM allocations.
      # vLLM tensor-parallel uses NCCL, which may rely on CUDA IPC / P2P; these knobs
      # keep NCCL on non-P2P paths so suspend/resume works.
      - NCCL_P2P_DISABLE=1
      - NCCL_SHM_DISABLE=1
    volumes:
      # IMPORTANT: store model weights + HF cache in the project root .cache/
      - ../.cache:/root/.cache
    ports:
      - "8010:8000"
    command: >
      --model Qwen/Qwen3-8B-AWQ
      --served-model-name qwen3-8b-awq
      --quantization awq
      --dtype half
      --max-model-len 8192
      --gpu-memory-utilization 0.80
      --tensor-parallel-size 2
      --enforce-eager
      --host 0.0.0.0
      --port 8000
```

Notes:
- `../.cache:/root/.cache` is the **only** intended linkage to the repo.
- `--tensor-parallel-size 2` ensures vLLM uses both GPUs (2-way TP).
- `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` are required here to make `cuda-checkpoint` resume succeed with vLLM’s multi-process tensor parallel.

---

## Task 2 — The vLLM container should load the Qwen3 8B AWQ Model

The compose `command:` above uses:

- `--model Qwen/Qwen3-8B-AWQ`
- `--served-model-name qwen3-8b-awq`

So all inference calls will reference model id `qwen3-8b-awq`.

---

## Task 3 — The model should be stored in the project root `.cache/` directory

The compose file mounts:

- project root: `../.cache`
- container: `/root/.cache`

This covers Hugging Face downloads and vLLM caches under `/root/.cache/**`.

---

## Task 4 — No more integration except shared `.cache/`

This test does **not**:
- use the controller (no [`controller/app.py`](../controller/app.py)),
- use the checkpoint service (no [`vllm/checkpoint_service.py`](../vllm/checkpoint_service.py)),
- use the frontend.

Checkpointing is done manually via `docker exec` + `cuda-checkpoint`.

---

## Task 5 — Spin up the vLLM container

Create `MULTI-GPU-TEST/Dockerfile` that installs `cuda-checkpoint` (mirrors the approach from [`vllm/Dockerfile`](../vllm/Dockerfile)).

```dockerfile
ARG VLLM_VERSION=v0.11.0
FROM vllm/vllm-openai:${VLLM_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cuda-checkpoint binary (as used elsewhere in this repo)
RUN git clone https://github.com/NVIDIA/cuda-checkpoint.git /tmp/cuda-checkpoint \
    && cp /tmp/cuda-checkpoint/bin/x86_64_Linux/cuda-checkpoint /usr/local/bin/ \
    && chmod +x /usr/local/bin/cuda-checkpoint \
    && rm -rf /tmp/cuda-checkpoint
```

Then start:

```sh
cd MULTI-GPU-TEST
docker compose up --build -d
docker compose logs -f
```

Wait until the vLLM logs show it is serving requests.

### Quick automation (recommended)
Run the full sequence (up → curl → suspend → resume → curl):

```sh
cd MULTI-GPU-TEST
chmod +x run_test.sh
./run_test.sh
```

To automatically tear down the container at the end:

```sh
CLEANUP=1 ./run_test.sh
```

---

## Task 6 — Test the running model with a curl call

### 6.1 List models
```sh
curl -s http://localhost:8010/v1/models | jq
```

You should see `qwen3-8b-awq` listed.

### 6.2 Run a chat completion (non-streaming)
```sh
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-awq",
    "messages": [{"role":"user","content":"Say hello in one short sentence."}],
    "temperature": 0.2,
    "max_tokens": 64,
    "stream": false
  }' | jq
```

---

## Task 7 — Suspend the model with CUDA checkpoints (incorporate both GPUs)

### What “incorporate both GPUs” means here
With tensor parallel `2`, vLLM typically uses **multiple worker processes** that collectively occupy both GPUs. To suspend “both GPUs”, you must toggle **all CUDA-controllable worker PIDs**, not just a single PID.

`cuda-checkpoint` operates on **PIDs**, so the procedure is:
1. find the vLLM server PID,
2. find all descendant PIDs,
3. filter to PIDs where `cuda-checkpoint --get-state` returns `running` or `checkpointed`,
4. toggle all of them (so every worker transitions to `checkpointed`).

### How suspend/resume works (in detail)

This test uses **NVIDIA `cuda-checkpoint`** to checkpoint/restore the *CUDA state* of the vLLM worker processes.

#### Key idea: CUDA state is toggled per-PID

- `cuda-checkpoint` does **not** checkpoint the whole container.
- It operates on a **single Linux PID at a time**.
- In a 2‑GPU tensor-parallel vLLM setup (`--tensor-parallel-size 2`), vLLM typically runs multiple CUDA worker processes (e.g. `VLLM::Worker_TP0` and `VLLM::Worker_TP1`). To “incorporate both GPUs”, you must toggle **all** CUDA-controllable worker PIDs.

#### What `cuda-checkpoint` actually does

At a high level, a transition to `checkpointed`:

- waits for in-flight CUDA work to finish,
- saves CUDA GPU state and device memory back to host-managed storage,
- releases the process’ GPU resources.

Important: `cuda-checkpoint` **does not stop CPU threads**. The Python/worker processes keep running, but any CUDA calls may block until CUDA is resumed.

This is why the GPU looks “empty” after suspend:

- `nvidia-smi --query-compute-apps=...` shows **no compute processes**,
- per-GPU VRAM drops to ~0 MiB (often a few MiB remain due to display/driver overhead).

When resuming back to `running`, `cuda-checkpoint`:

- re-acquires GPUs,
- restores CUDA objects/contexts,
- copies device memory back to GPU,
- unblocks CUDA calls.

#### How `cuda-toggle-all` finds the right PIDs

The helper `cuda-toggle-all` (baked into the image as `/usr/local/bin/cuda-toggle-all`) does the PID bookkeeping so you don’t have to.

1) Find the vLLM API server PID

- It runs `pgrep -f vllm.entrypoints.openai.api_server` and picks the first match.

2) Build the full process tree

- It reads `ps -e -o pid=,ppid=` and builds a PPID → children map.
- It then does a BFS from the API server PID to include the server and **all descendants**.

3) Filter to CUDA-controllable PIDs

- For each PID in that tree, it runs `cuda-checkpoint --get-state --pid <pid>`.
- Only PIDs that return `running` or `checkpointed` are considered “CUDA-controllable”.

4) Toggle + verify until converged

- For each CUDA-controllable PID that is not yet in the target state, it runs `cuda-checkpoint --toggle --pid <pid>`.
- It repeats this for up to `CUDA_TOGGLE_VERIFY_ATTEMPTS` attempts (default 50), sleeping `CUDA_TOGGLE_VERIFY_SLEEP_S` seconds between attempts (default 0.2), until **all** targeted PIDs report the desired state.

This “verify loop” is important because in multi-process setups, workers may not all transition instantaneously.

#### Why the compose config includes extra permissions

`cuda-checkpoint` relies on driver/kernel operations that are commonly blocked by container sandboxes.
That’s why the standalone compose includes:

- `cap_add: [SYS_PTRACE]`
- `security_opt: [seccomp=unconfined]`

Without these, suspend/resume can fail even though inference works.

#### Why the NCCL env vars are required for reliable resume

vLLM tensor-parallel uses NCCL under the hood, and some NCCL transports can rely on CUDA IPC / shared-memory paths.
`cuda-checkpoint` restore has limitations around CUDA IPC/UVM allocations; when those are used, resume can fail with errors like:

- `OS call failed or operation not supported on this OS`

To avoid those paths, the compose sets:

- `NCCL_P2P_DISABLE=1`
- `NCCL_SHM_DISABLE=1`

These keep NCCL on safer transports for this checkpoint/restore demo, making “resume” actually work for both TP workers.

### Suspend command (runs inside the container)
This repo’s implementation bakes a helper into the image: `cuda-toggle-all`.
It locates the vLLM process tree and toggles *all* CUDA-controllable worker PIDs.

Suspend (checkpoint):

```sh
docker exec -it vllm-qwen3-8b-awq cuda-toggle-all checkpointed
```

If you prefer the fully expanded inline version, you can still use the command below.

```sh
docker exec -it vllm-qwen3-8b-awq bash -lc '
set -euo pipefail

echo "[suspend] locating vLLM main pid..."
MAIN_PID="$(pgrep -f "vllm.entrypoints.openai.api_server" | head -n1 || true)"
if [ -z "${MAIN_PID}" ]; then
  echo "[suspend] ERROR: vLLM main pid not found"
  exit 1
fi
echo "[suspend] MAIN_PID=${MAIN_PID}"

# Collect process tree: MAIN_PID + all descendants (PPID walk via ps)
ALL_PIDS="$(python3 - <<PY
import subprocess
main=int("'"${MAIN_PID}"'")
out=subprocess.check_output(["ps","-e","-o","pid=,ppid="], text=True).strip().splitlines()
ppid_map={}
for line in out:
    pid_s, ppid_s = line.split()
    ppid_map.setdefault(int(ppid_s), []).append(int(pid_s))
seen=set()
q=[main]
while q:
    p=q.pop(0)
    if p in seen: 
        continue
    seen.add(p)
    q.extend(ppid_map.get(p, []))
print(" ".join(str(x) for x in sorted(seen)))
PY
)"

echo "[suspend] candidate pids: ${ALL_PIDS}"

CUDA_PIDS=()
for pid in ${ALL_PIDS}; do
  st="$(cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null | tr -d "\r" | tr "[:upper:]" "[:lower:]" || true)"
  if [ "${st}" = "running" ] || [ "${st}" = "checkpointed" ]; then
    CUDA_PIDS+=("${pid}")
  fi
done

if [ "${#CUDA_PIDS[@]}" -eq 0 ]; then
  echo "[suspend] ERROR: no CUDA-controllable PIDs found"
  exit 2
fi

echo "[suspend] cuda pids: ${CUDA_PIDS[*]}"
echo "[suspend] current states:"
for pid in "${CUDA_PIDS[@]}"; do
  echo "  pid=${pid} state=$(cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null || true)"
done

echo "[suspend] toggling all CUDA PIDs (TP=2 => covers both GPUs by toggling all workers)..."
for pid in "${CUDA_PIDS[@]}"; do
  # toggle: running <-> checkpointed
  cuda-checkpoint --toggle --pid "${pid}" >/dev/null 2>&1 || true
done

echo "[suspend] verifying:"
for pid in "${CUDA_PIDS[@]}"; do
  echo "  pid=${pid} state=$(cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null || true)"
done
'
```

Expected: all relevant worker PIDs show `checkpointed`.

---

## Task 8 — Resume the model again

Resuming is the same operation: **toggle all CUDA worker PIDs** back to `running`.

```sh
docker exec -it vllm-qwen3-8b-awq bash -lc '
set -euo pipefail

MAIN_PID="$(pgrep -f "vllm.entrypoints.openai.api_server" | head -n1 || true)"
if [ -z "${MAIN_PID}" ]; then
  echo "[resume] ERROR: vLLM main pid not found"
  exit 1
fi

ALL_PIDS="$(python3 - <<PY
import subprocess
main=int("'"${MAIN_PID}"'")
out=subprocess.check_output(["ps","-e","-o","pid=,ppid="], text=True).strip().splitlines()
ppid_map={}
for line in out:
    pid_s, ppid_s = line.split()
    ppid_map.setdefault(int(ppid_s), []).append(int(pid_s))
seen=set()
q=[main]
while q:
    p=q.pop(0)
    if p in seen:
        continue
    seen.add(p)
    q.extend(ppid_map.get(p, []))
print(" ".join(str(x) for x in sorted(seen)))
PY
)"

CUDA_PIDS=()
for pid in ${ALL_PIDS}; do
  st="$(cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null | tr -d "\r" | tr "[:upper:]" "[:lower:]" || true)"
  if [ "${st}" = "running" ] || [ "${st}" = "checkpointed" ]; then
    CUDA_PIDS+=("${pid}")
  fi
done

if [ "${#CUDA_PIDS[@]}" -eq 0 ]; then
  echo "[resume] ERROR: no CUDA-controllable PIDs found"
  exit 2
fi

echo "[resume] toggling all CUDA PIDs back to running..."
for pid in "${CUDA_PIDS[@]}"; do
  cuda-checkpoint --toggle --pid "${pid}" >/dev/null 2>&1 || true
done

echo "[resume] verifying:"
for pid in "${CUDA_PIDS[@]}"; do
  echo "  pid=${pid} state=$(cuda-checkpoint --get-state --pid "${pid}" 2>/dev/null || true)"
done
'
```

Expected: states return to `running`.

---

## Task 9 — Run the test call again via curl

```sh
curl -s http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-awq",
    "messages": [{"role":"user","content":"Confirm you are running after resume in one sentence."}],
    "temperature": 0.2,
    "max_tokens": 64,
    "stream": false
  }' | jq
```

If resume succeeded, the model should respond normally.

---

## Cleanup

```sh
cd MULTI-GPU-TEST
docker compose down -v
```

The project root cache directory `../.cache/` is **not** deleted by this (by design).