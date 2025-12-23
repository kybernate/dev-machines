from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import subprocess
import os
import httpx

app = FastAPI(
    title="vLLM Checkpoint Service",
    version="0.1.0",
    description="Controls cuda-checkpoint for vLLM (Suspend/Resume of GPU states).",
)

VLLM_PID_FILE = Path("/tmp/vllm.pid")
VLLM_API_URL = "http://127.0.0.1:8000"  # vLLM runs in the container on port 8000

class CheckpointRequest(BaseModel):
    checkpoint_id: Optional[str] = None  # reserved for later design

class StatusResponse(BaseModel):
    state: str
    cuda_state: str
    has_toggled_once: bool
    initialized: bool

class ServiceState(BaseModel):
    # "logical" high-level state, based on CUDA state
    state: str = "INIT"          # INIT, RUNNING, SUSPENDED, UNKNOWN, ERROR
    has_toggled_once: bool = False
    initialized: bool = False    # was /ready successfully initialized once?

state = ServiceState()

# Cache for the actual GPU-using PID
GPU_PID: Optional[int] = None


# --- Helper functions for PID/GPU detection -------------------------------------------

def get_main_pid() -> int:
    """
    Reads the PID of the vLLM API process from /tmp/vllm.pid.
    """
    if not VLLM_PID_FILE.exists():
        raise RuntimeError("vLLM PID file not found")
    pid_str = VLLM_PID_FILE.read_text().strip()
    if not pid_str.isdigit():
        raise RuntimeError(f"Invalid PID content in {VLLM_PID_FILE}: {pid_str}")
    return int(pid_str)


def read_cgroup(pid: int) -> str:
    """
    Returns the cgroup line(s) of a process as a string.
    We use this to find processes that run in the same container.
    """
    cgroup_file = Path(f"/proc/{pid}/cgroup")
    try:
        return cgroup_file.read_text()
    except FileNotFoundError:
        return ""


def process_uses_gpu(pid: int) -> bool:
    """
    Checks if the process has an open file descriptor on /dev/nvidia*.
    """
    fd_dir = Path(f"/proc/{pid}/fd")
    if not fd_dir.exists():
        return False

    try:
        for fd in fd_dir.iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                return True
    except FileNotFoundError:
        return False
    return False


def find_gpu_pid() -> int:
    """
    Tries to find the "GPU process" that belongs to the vLLM container.
    Strategy:
      - main_pid = PID from /tmp/vllm.pid
      - go through all PIDs in /proc
      - only consider processes in the same cgroup as main_pid
      - among these, look for a process that uses /dev/nvidia*
      - Fallback: main_pid
    """
    main_pid = get_main_pid()
    main_cgroup = read_cgroup(main_pid)

    if not main_cgroup:
        print("[checkpoint] Warning: could not read cgroup of main PID, falling back to main_pid")
        return main_pid

    gpu_candidate = None

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == main_pid:
            continue  # skip the API server, it is rarely the GPU process

        # only consider processes in the same container
        if read_cgroup(pid) != main_cgroup:
            continue

        if process_uses_gpu(pid):
            gpu_candidate = pid
            print(f"[checkpoint] Discovered GPU PID: {gpu_candidate}")
            break

    if gpu_candidate is None:
        print("[checkpoint] Warning: no separate GPU PID found, falling back to main PID")
        gpu_candidate = main_pid

    return gpu_candidate


def get_gpu_pid() -> int:
    """
    Returns the cached GPU PID or searches for it anew if necessary.
    """
    global GPU_PID
    if GPU_PID is not None:
        return GPU_PID
    GPU_PID = find_gpu_pid()
    return GPU_PID


def is_vllm_ready() -> bool:
    """
    Checks if vLLM in the container is ready.
    Prefers /v1/models because that is a "real" API check.
    """
    try:
        resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=1.0)
        if resp.status_code != 200:
            return False
        # Optional: check model list
        data = resp.json()
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return True
        return True  # 200 is enough for us as a minimal criterion
    except Exception as e:
        print(f"[ready] vLLM not ready yet: {e}")
        return False


# --- CUDA-Checkpoint-Wrapper ------------------------------------------------------------

def cuda_get_state() -> str:
    """
    Queries the CUDA checkpoint state of the GPU process from the driver.
    Expected output: "running" or "checkpointed".
    """
    pid = get_gpu_pid()
    try:
        result = subprocess.run(
            ["cuda-checkpoint", "--get-state", "--pid", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        out = result.stdout.strip()
        print(f"[checkpoint] get-state stdout (pid {pid}):", repr(out))
        return out
    except subprocess.CalledProcessError as e:
        msg = f"cuda-checkpoint --get-state failed for pid {pid}: {e.stderr}"
        print(msg)
        state.state = "ERROR"
        raise HTTPException(status_code=500, detail=msg)


def cuda_toggle() -> str:
    """
    Performs a toggle and then returns the new CUDA state.
    """
    pid = get_gpu_pid()
    try:
        result = subprocess.run(
            ["cuda-checkpoint", "--toggle", "--pid", str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[checkpoint] toggle stdout (pid {pid}):", repr(result.stdout.strip()))
    except subprocess.CalledProcessError as e:
        msg = f"cuda-checkpoint --toggle failed for pid {pid}: {e.stderr}"
        print(msg)
        state.state = "ERROR"
        raise HTTPException(status_code=500, detail=msg)

    # After successful toggle, get the new state
    new_state = cuda_get_state()
    return new_state


def map_cuda_to_logical(cuda_state: str) -> str:
    """
    Translates the CUDA state ("running", "checkpointed", ...) into our
    logical state for the API.
    """
    s = cuda_state.lower()
    if s == "running":
        return "RUNNING"
    if s == "checkpointed":
        return "SUSPENDED"
    # fallback
    return "UNKNOWN"


# --- API-Endpunkte ----------------------------------------------------------------------

@app.get("/healthz", tags=["meta"])
def healthz():
    # Only: is the service process alive?
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready():
    """
    "Ready" = this container is basically operational:
      - vLLM API is reachable (model loaded),
      - GPU PID found and CUDA API accessible,
      - initial one-time initialization possibly done.
    From Docker Compose's perspective, this endpoint should
    always return 200 after the first successful initialization,
    regardless of whether the CUDA state is currently running or checkpointed.
    """
    # 1) vLLM must be ready (at least once)
    if not is_vllm_ready():
        raise HTTPException(status_code=503, detail="vLLM API not ready")

    # 2) Main PID must exist
    if not VLLM_PID_FILE.exists():
        raise HTTPException(status_code=503, detail="vLLM PID file missing")

    # 3) Determine GPU PID & CUDA state
    cuda_state = cuda_get_state()
    logical = map_cuda_to_logical(cuda_state)

    # 4) Initial "setup" (one-time)
    if not state.initialized:
        # We have a working setup here:
        # vLLM responds, GPU PID valid, get-state works.
        # Optional: one-time base checkpoint, if we want to ensure
        # that after /ready a valid snapshot exists.
        if logical == "RUNNING":
            print("[ready] Initial base checkpoint: toggling CUDA state once")
            new_cuda_state = cuda_toggle()
            cuda_state = new_cuda_state
            logical = map_cuda_to_logical(new_cuda_state)
            state.has_toggled_once = True
        else:
            # If we are already "SUSPENDED", there was possibly a manual toggle before.
            state.has_toggled_once = (logical == "SUSPENDED")

        state.state = logical
        state.initialized = True
        print(f"[ready] Initialization complete. state={state.state}, cuda_state={cuda_state}")

    # 5) From here on, the service is considered "ready", regardless of whether running or suspended.
    return {
        "ready": True,
        "state": state.state,
        "cuda_state": cuda_state,
        "has_toggled_once": state.has_toggled_once,
        "initialized": state.initialized,
    }


@app.get("/status", response_model=StatusResponse, tags=["checkpoint"])
def status():
    """
    Returns the current CUDA/logic status.
    Here the controller can distinguish "active/passive".
    """
    cuda_state = cuda_get_state()
    logical = map_cuda_to_logical(cuda_state)
    if logical != "UNKNOWN":
        state.state = logical
    return StatusResponse(
        state=state.state,
        cuda_state=cuda_state,
        has_toggled_once=state.has_toggled_once,
        initialized=state.initialized,
    )


@app.post("/checkpoint/dump", tags=["checkpoint"])
def dump(req: CheckpointRequest):
    """
    Semantics: Bring model into "checkpointed" state.
    Idempotent:
      - if already checkpointed → no toggle
      - if running → toggle → checkpointed
    """
    cuda_state = cuda_get_state()
    s = cuda_state.lower()

    if s == "checkpointed":
        # already in desired state
        state.state = "SUSPENDED"
        return {
            "status": "ok",
            "toggled": False,
            "state": state.state,
            "cuda_state": cuda_state,
        }

    if s == "running":
        new_cuda_state = cuda_toggle()
        logical = map_cuda_to_logical(new_cuda_state)
        state.has_toggled_once = True
        state.state = logical
        return {
            "status": "ok",
            "toggled": True,
            "state": state.state,
            "cuda_state": new_cuda_state,
        }

    raise HTTPException(
        status_code=409,
        detail=f"Cannot dump from CUDA state '{cuda_state}'",
    )


@app.post("/checkpoint/restore", tags=["checkpoint"])
def restore(req: CheckpointRequest):
    """
    Semantics: Bring model into "running" state.
    Idempotent:
      - if already running → no toggle
      - if checkpointed → toggle → running
    """
    cuda_state = cuda_get_state()
    s = cuda_state.lower()

    if s == "running":
        # already in desired state
        state.state = "RUNNING"
        return {
            "status": "ok",
            "toggled": False,
            "state": state.state,
            "cuda_state": cuda_state,
        }

    if s == "checkpointed":
        new_cuda_state = cuda_toggle()
        logical = map_cuda_to_logical(new_cuda_state)
        state.has_toggled_once = True
        state.state = logical
        return {
            "status": "ok",
            "toggled": True,
            "state": state.state,
            "cuda_state": new_cuda_state,
        }

    raise HTTPException(
        status_code=409,
        detail=f"Cannot restore from CUDA state '{cuda_state}'",
    )