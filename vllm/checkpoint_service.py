from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional, List
import subprocess
import os
import httpx

app = FastAPI(
    title="vLLM Checkpoint Service",
    version="0.3.0",
    description="Multi-GPU Checkpoint Controller for vLLM",
)

VLLM_PID_FILE = Path("/tmp/vllm.pid")
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://127.0.0.1:8000")

class StatusResponse(BaseModel):
    state: str
    cuda_states: dict
    has_toggled_once: bool
    initialized: bool
    active_pids: List[int]

class ServiceState:
    state: str = "INIT"
    has_toggled_once: bool = False
    initialized: bool = False

service_state = ServiceState()
GPU_PIDS: List[int] = []

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
        for fd in fd_dir.iterdir():
            if "/dev/nvidia" in os.readlink(fd): return True
    except: pass
    return False

def find_all_vllm_pids() -> List[int]:
    """Findet alle Prozesse im selben Container-Context."""
    try:
        main_pid = get_main_pid()
        main_cgroup = read_cgroup(main_pid)
        pids = [main_pid]
        for entry in os.listdir("/proc"):
            if entry.isdigit() and int(entry) != main_pid:
                if read_cgroup(int(entry)) == main_cgroup:
                    pids.append(int(entry))
        return sorted(pids)
    except: return []

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
                # API-Server liefert hier oft Fehler, da kein CUDA-Context
                states[pid] = "non-cuda"
        except:
            states[pid] = "error"
    return states

def toggle_cuda_processes(target_state: str):
    """
    Toggelt alle Prozesse, die tatsächlich CUDA nutzen.
    target_state: 'suspend' (wird zu checkpointed) oder 'resume' (wird zu running)
    """
    pids = find_all_vllm_pids()
    states = get_cuda_states(pids)
    
    # Filter: Nur PIDs, die NICHT 'non-cuda' sind
    cuda_pids = [pid for pid, st in states.items() if st not in ["non-cuda", "error"]]
    
    print(f"[checkpoint] Target: {target_state}. Action on PIDs: {cuda_pids}")
    
    results = []
    for pid in cuda_pids:
        current_st = states[pid]
        # Nur toggeln, wenn nicht schon im Zielzustand
        if (target_state == "suspend" and current_st == "running") or \
           (target_state == "resume" and current_st == "checkpointed"):
            try:
                subprocess.run(["cuda-checkpoint", "--toggle", "--pid", str(pid)], check=True)
                results.append(True)
            except Exception as e:
                print(f"[checkpoint] Failed to toggle PID {pid}: {e}")
                results.append(False)
    
    return all(results) if results else True

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

    pids = find_all_vllm_pids()
    states = get_cuda_states(pids)
    
    if not service_state.initialized:
        service_state.initialized = True
        service_state.state = "RUNNING"
        print(f"[ready] vLLM detected with PIDs {pids}. CUDA processes: {[p for p, s in states.items() if s != 'non-cuda']}")

    return {"status": "ready", "pids": states}

@app.get("/status", response_model=StatusResponse)
async def status():
    pids = find_all_vllm_pids()
    states = get_cuda_states(pids)
    
    # Bestimme globalen Status anhand der Worker
    worker_states = [s for s in states.values() if s != "non-cuda"]
    if "running" in worker_states: logical = "RUNNING"
    elif "checkpointed" in worker_states: logical = "SUSPENDED"
    else: logical = "INIT"

    return StatusResponse(
        state=logical,
        cuda_states=states,
        has_toggled_once=service_state.has_toggled_once,
        initialized=service_state.initialized,
        active_pids=pids
    )

@app.post("/checkpoint/dump")
async def dump():
    success = toggle_cuda_processes("suspend")
    if success:
        service_state.state = "SUSPENDED"
        service_state.has_toggled_once = True
        return {"message": "GPU state suspended (checkpointed)"}
    raise HTTPException(status_code=500, detail="Failed to suspend some GPU processes")

@app.post("/checkpoint/restore")
async def restore():
    success = toggle_cuda_processes("resume")
    if success:
        service_state.state = "RUNNING"
        return {"message": "GPU state resumed (running)"}
    raise HTTPException(status_code=500, detail="Failed to resume some GPU processes")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)