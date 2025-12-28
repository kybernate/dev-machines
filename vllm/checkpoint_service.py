from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
import subprocess
import os
import httpx
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

app = FastAPI(
    title="vLLM Checkpoint Service",
    version="0.3.0",
    description="Multi-GPU Checkpoint Controller for vLLM",
)

VLLM_PID_FILE = Path("/tmp/vllm.pid")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://127.0.0.1:8000")

# Toggle behavior (useful for tensor-parallel multi-GPU setups)
CUDA_TOGGLE_PARALLEL = os.getenv("CUDA_TOGGLE_PARALLEL", "1").lower() in ("1", "true", "yes")
CUDA_TOGGLE_MAX_WORKERS = int(os.getenv("CUDA_TOGGLE_MAX_WORKERS", "0"))  # 0 => auto
CUDA_TOGGLE_VERIFY_ATTEMPTS = int(os.getenv("CUDA_TOGGLE_VERIFY_ATTEMPTS", "50"))
CUDA_TOGGLE_VERIFY_SLEEP_S = float(os.getenv("CUDA_TOGGLE_VERIFY_SLEEP_S", "0.2"))

class StatusResponse(BaseModel):
    state: str
    cuda_states: dict
    cuda_state: Optional[str] = None
    has_toggled_once: bool
    initialized: bool
    active_pids: List[int]

class ServiceState:
    state: str = "INIT"
    has_toggled_once: bool = False
    initialized: bool = False

service_state = ServiceState()
GPU_PIDS: List[int] = []
LAST_CUDA_PIDS: List[int] = []

# --- PID & GPU Detection -----------------------------------------------------

def get_main_pid() -> int:
    if not VLLM_PID_FILE.exists():
        raise RuntimeError("vLLM PID file not found")
    return int(VLLM_PID_FILE.read_text().strip())

def read_cgroup(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cgroup").read_text()
    except: return ""

def process_uses_gpu(pid: int) -> bool:
    """Prüft, ob der Prozess NVIDIA-Devices offen hat."""
    try:
        fd_dir = Path(f"/proc/{pid}/fd")
        if not fd_dir.exists():
            return False
        for fd in fd_dir.iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                return True
    except: pass
    return False

def find_all_vllm_pids() -> List[int]:
    """Findet vLLM-Prozesse in diesem Container via Prozess-Baum.

    Wichtig: In Docker ist /proc typischerweise bereits auf den Container-PID-NS
    beschränkt. Ein strikter Vergleich der kompletten cgroup-Datei ist für
    Tensor-Parallel oft zu fragil (Sub-cgroups), daher nutzen wir PPID/Child-Tree.
    """
    try:
        main_pid = get_main_pid()
    except Exception:
        return []

    def get_ppid(pid: int) -> Optional[int]:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            # Format: pid (comm) state ppid ...
            after_comm = stat.rsplit(")", 1)[1].strip()
            parts = after_comm.split()
            return int(parts[1])  # state is parts[0], ppid is parts[1]
        except Exception:
            return None

    # Build PPID -> children mapping for all visible processes
    children: dict[int, List[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        ppid = get_ppid(pid)
        if ppid is None:
            continue
        children.setdefault(ppid, []).append(pid)

    # BFS from main_pid
    seen: set[int] = set()
    queue: List[int] = [main_pid]
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        queue.extend(children.get(pid, []))

    return sorted(seen)


def _nvidia_smi_compute_pids() -> List[int]:
    """Liest aktive Compute PIDs aus nvidia-smi.

    Rückgabewert ist leer, wenn nvidia-smi nicht verfügbar ist oder nichts läuft.
    """
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode != 0:
            return []
        pids: List[int] = []
        for line in res.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pids.append(int(line))
            except ValueError:
                continue
        return sorted(set(pids))
    except Exception:
        return []


def find_vllm_gpu_pids() -> List[int]:
    """Findet die GPU-Workload-PIDs von vLLM.

    Für Tensor-Parallel gibt es mehrere Worker-Prozesse. Wir nehmen die vLLM
    Prozess-Tree-PIDs und schneiden sie optional mit den PIDs aus nvidia-smi.
    """
    vllm_tree_pids = set(find_all_vllm_pids())
    if not vllm_tree_pids:
        return []

    compute_pids = set(_nvidia_smi_compute_pids())
    if compute_pids:
        # Achtung: In Standard-Docker-Setups zeigt nvidia-smi oft Host-PIDs,
        # während /proc Container-PIDs zeigt. Dann ist die Schnittmenge leer.
        intersected = sorted(vllm_tree_pids.intersection(compute_pids))
        if intersected:
            gpu_pids = intersected
        else:
            # Fallback: /proc fd scan (unabhängig von PID-Namespaces)
            gpu_pids = sorted([pid for pid in vllm_tree_pids if process_uses_gpu(pid)])
    else:
        # Fallback: /proc fd scan (langsamer, aber unabhängig von nvidia-smi)
        gpu_pids = sorted([pid for pid in vllm_tree_pids if process_uses_gpu(pid)])

    return gpu_pids

# --- CUDA Checkpoint Logic ---------------------------------------------------

def get_cuda_states(pids: List[int]) -> dict:
    """Prüft für jede PID den CUDA-Status. Filtert Nicht-CUDA-Prozesse."""
    states = {}
    for pid in pids:
        try:
            res = subprocess.run(
                ["cuda-checkpoint", "--get-state", "--pid", str(pid)],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0:
                states[pid] = res.stdout.strip().lower()
            else:
                # Kann Non-CUDA sein, aber auch Rechte/Tool/Timing-Probleme.
                # Nicht als Erfolg werten.
                states[pid] = "unknown"
        except:
            states[pid] = "error"
    return states


def find_vllm_cuda_pids() -> List[int]:
    """Find vLLM worker PIDs that cuda-checkpoint can control.

    Important: After checkpointing, processes may no longer have /dev/nvidia*
    file descriptors, so /proc fd-based detection can go empty even though
    cuda-checkpoint can still get/toggle state.
    """
    global LAST_CUDA_PIDS

    candidates = find_all_vllm_pids()
    if not candidates:
        return []

    states = get_cuda_states(candidates)
    cuda_pids = [pid for pid, st in states.items() if st in ["running", "checkpointed"]]

    if cuda_pids:
        LAST_CUDA_PIDS = sorted(cuda_pids)
        return LAST_CUDA_PIDS

    # Fallback: if we previously found CUDA PIDs, keep using them.
    # (Useful if the process tree is stable but state probing is momentarily flaky.)
    return list(LAST_CUDA_PIDS)

def toggle_cuda_processes(target_state: str):
    """
    Toggelt alle Prozesse, die tatsächlich CUDA nutzen.
    target_state: 'suspend' (wird zu checkpointed) oder 'resume' (wird zu running)
    """
    # Wichtig: Wir wollen wirklich die CUDA-kontrollierbaren Worker togglen.
    pids = find_vllm_cuda_pids()
    if not pids:
        print("[checkpoint] No vLLM CUDA PIDs found; refusing to report success.")
        return False

    states = get_cuda_states(pids)

    desired = "checkpointed" if target_state == "suspend" else "running"
    toggled_any = False
    ok = True

    print(f"[checkpoint] Target: {target_state}. CUDA PIDs: {pids}. States: {states}")

    to_toggle = [pid for pid in pids if states.get(pid, "unknown") != desired]
    if not to_toggle:
        # Already in desired state.
        return True

    def _toggle_one(pid: int):
        try:
            res = subprocess.run(
                ["cuda-checkpoint", "--toggle", "--pid", str(pid)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return pid, res.returncode, (res.stdout or ""), (res.stderr or "")
        except Exception as e:
            return pid, -1, "", str(e)

    # Toggle concurrently (important for multi-GPU / tensor-parallel)
    if CUDA_TOGGLE_PARALLEL and len(to_toggle) > 1:
        max_workers = CUDA_TOGGLE_MAX_WORKERS or min(len(to_toggle), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_toggle_one, pid) for pid in to_toggle]
            results = [f.result() for f in as_completed(futures)]
    else:
        results = [_toggle_one(pid) for pid in to_toggle]

    # If any toggle command errored, log it (final success is still determined by states).
    for pid, rc, out, err in results:
        if rc != 0:
            print(f"[checkpoint] cuda-checkpoint toggle failed for PID {pid}: rc={rc} err={err.strip()} out={out.strip()}")
            ok = False
        else:
            toggled_any = True

    # Verification: all PIDs should reach desired within a bounded time.
    verified_all = False
    for _ in range(max(1, CUDA_TOGGLE_VERIFY_ATTEMPTS)):
        time.sleep(CUDA_TOGGLE_VERIFY_SLEEP_S)
        new_states = get_cuda_states(pids)
        if all(new_states.get(pid, "unknown") == desired for pid in pids):
            verified_all = True
            ok = True  # if the state matches, treat it as success
            break

    if not verified_all:
        final_states = get_cuda_states(pids)
        print(f"[checkpoint] Not all PIDs reached desired state '{desired}'. Final states: {final_states}")
        return False

    # If we got here, all PIDs are in desired state.
    return True


def _logical_from_cuda_states(states: dict) -> str:
    worker_states = [s for s in states.values() if s not in ["unknown", "error"]]
    if "running" in worker_states:
        return "RUNNING"
    if "checkpointed" in worker_states:
        return "SUSPENDED"
    return "INIT"


def _aggregate_cuda_state(states: dict) -> str:
    """Backwards-compatible single cuda_state string.

    The controller/UI expects a single string: "running" or "checkpointed".
    For multi-process setups we collapse as:
      - if any running => running
      - else if any checkpointed => checkpointed
      - else => unknown
    """
    vals = [str(v).lower() for v in states.values()]
    if any(v == "running" for v in vals):
        return "running"
    if any(v == "checkpointed" for v in vals):
        return "checkpointed"
    return "unknown"


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# --- API Endpoints -----------------------------------------------------------

@app.get("/ready")
async def ready():
    # Prüfe ob vLLM API da ist
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_API_URL}/v1/models", timeout=2.0)
            if resp.status_code != 200: raise Exception()
    except:
        raise HTTPException(status_code=503, detail="vLLM not ready")

    # Original semantics: once initialized successfully, /ready should stay 200
    # even if we cannot currently probe cuda-checkpoint state.
    pids = find_all_vllm_pids()
    cuda_pids = find_vllm_cuda_pids()
    states = get_cuda_states(cuda_pids) if cuda_pids else {}

    if service_state.initialized:
        cuda_state = _aggregate_cuda_state(states)
        return {
            "ready": True,
            "status": "ready",
            "state": service_state.state,
            "cuda_state": cuda_state,
            "has_toggled_once": service_state.has_toggled_once,
            "initialized": service_state.initialized,
            "vllm_tree_pids": pids,
            "cuda_pids": cuda_pids,
            "cuda_states": states,
        }

    if not cuda_pids:
        raise HTTPException(status_code=503, detail="CUDA workload PIDs not detected yet")

    if not any(st in ["running", "checkpointed"] for st in states.values()):
        raise HTTPException(status_code=503, detail="cuda-checkpoint state not available yet")

    # Original behavior (v0.1.0): after the first successful initialization,
    # do an initial toggle when the workload is RUNNING, to create a base
    # checkpoint and end up in SUSPENDED.
    if not service_state.initialized:
        logical = _logical_from_cuda_states(states)
        print(f"[ready] First init. vLLM tree PIDs {pids}. CUDA PIDs {cuda_pids}. States: {states}. logical={logical}")

        if logical == "RUNNING":
            print("[ready] Initial base checkpoint: toggling CUDA state once")
            success = toggle_cuda_processes("suspend")
            service_state.has_toggled_once = bool(success)
            # refresh states after toggle
            states = get_cuda_states(find_vllm_cuda_pids())
            logical = _logical_from_cuda_states(states)
        else:
            service_state.has_toggled_once = (logical == "SUSPENDED")

        service_state.state = logical
        service_state.initialized = True
        print(f"[ready] Initialization complete. state={service_state.state}")

    cuda_state = _aggregate_cuda_state(states)

    return {
        "ready": True,
        "status": "ready",
        "state": service_state.state,
        "cuda_state": cuda_state,
        "has_toggled_once": service_state.has_toggled_once,
        "initialized": service_state.initialized,
        "vllm_tree_pids": pids,
        "cuda_pids": cuda_pids,
        "cuda_states": states,
    }

@app.get("/status", response_model=StatusResponse)
async def status():
    pids = find_all_vllm_pids()
    cuda_pids = find_vllm_cuda_pids()
    states = get_cuda_states(cuda_pids) if cuda_pids else {}

    # Bestimme globalen Status anhand der Worker
    logical = _logical_from_cuda_states(states)
    cuda_state = _aggregate_cuda_state(states)

    return StatusResponse(
        state=logical,
        cuda_states=states,
        cuda_state=cuda_state,
        has_toggled_once=service_state.has_toggled_once,
        initialized=service_state.initialized,
        active_pids=cuda_pids
    )

@app.post("/checkpoint/dump")
async def dump():
    # Backwards-compatible response expected by controller:
    # {status, toggled, state, cuda_state}
    cuda_pids = find_vllm_cuda_pids()
    if not cuda_pids:
        raise HTTPException(status_code=503, detail="No CUDA PIDs found")
    before = get_cuda_states(cuda_pids)
    before_cuda_state = _aggregate_cuda_state(before)

    if before_cuda_state == "checkpointed":
        service_state.state = "SUSPENDED"
        return {
            "status": "ok",
            "toggled": False,
            "state": service_state.state,
            "cuda_state": before_cuda_state,
            "cuda_states": before,
            "active_pids": cuda_pids,
        }

    success = toggle_cuda_processes("suspend")
    after_pids = find_vllm_cuda_pids()
    after = get_cuda_states(after_pids) if after_pids else {}
    after_cuda_state = _aggregate_cuda_state(after)

    if success and after_cuda_state == "checkpointed":
        service_state.state = "SUSPENDED"
        service_state.has_toggled_once = True
        return {
            "status": "ok",
            "toggled": True,
            "state": service_state.state,
            "cuda_state": after_cuda_state,
            "cuda_states": after,
            "active_pids": after_pids,
        }

    raise HTTPException(status_code=500, detail="Failed to suspend CUDA processes")

@app.post("/checkpoint/restore")
async def restore():
    cuda_pids = find_vllm_cuda_pids()
    if not cuda_pids:
        raise HTTPException(status_code=503, detail="No CUDA PIDs found")
    before = get_cuda_states(cuda_pids)
    before_cuda_state = _aggregate_cuda_state(before)

    if before_cuda_state == "running":
        service_state.state = "RUNNING"
        return {
            "status": "ok",
            "toggled": False,
            "state": service_state.state,
            "cuda_state": before_cuda_state,
            "cuda_states": before,
            "active_pids": cuda_pids,
        }

    success = toggle_cuda_processes("resume")
    after_pids = find_vllm_cuda_pids()
    after = get_cuda_states(after_pids) if after_pids else {}
    after_cuda_state = _aggregate_cuda_state(after)

    if success and after_cuda_state == "running":
        service_state.state = "RUNNING"
        service_state.has_toggled_once = True
        return {
            "status": "ok",
            "toggled": True,
            "state": service_state.state,
            "cuda_state": after_cuda_state,
            "cuda_states": after,
            "active_pids": after_pids,
        }

    raise HTTPException(status_code=500, detail="Failed to resume CUDA processes")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)