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
    description="Steuert cuda-checkpoint für vLLM (Suspend/Resume des GPU-States).",
)

VLLM_PID_FILE = Path("/tmp/vllm.pid")
VLLM_API_URL = "http://127.0.0.1:8000"  # vLLM läuft im Container auf Port 8000

class CheckpointRequest(BaseModel):
    checkpoint_id: Optional[str] = None  # reserviert fürs spätere Design

class StatusResponse(BaseModel):
    state: str
    cuda_state: str
    has_toggled_once: bool
    initialized: bool

class ServiceState(BaseModel):
    # „logischer“ High-Level-State, angelehnt an CUDA-State
    state: str = "INIT"          # INIT, RUNNING, SUSPENDED, UNKNOWN, ERROR
    has_toggled_once: bool = False
    initialized: bool = False    # wurde /ready einmal erfolgreich „durchinitialisiert“?

state = ServiceState()

# Cache für den tatsächlich GPU-nutzenden PID
GPU_PID: Optional[int] = None


# --- Hilfsfunktionen für PID-/GPU-Ermittlung -------------------------------------------

def get_main_pid() -> int:
    """
    Liest die PID des vLLM-API-Prozesses aus /tmp/vllm.pid.
    """
    if not VLLM_PID_FILE.exists():
        raise RuntimeError("vLLM PID file not found")
    pid_str = VLLM_PID_FILE.read_text().strip()
    if not pid_str.isdigit():
        raise RuntimeError(f"Invalid PID content in {VLLM_PID_FILE}: {pid_str}")
    return int(pid_str)


def read_cgroup(pid: int) -> str:
    """
    Liefert die cgroup-Zeile(n) eines Prozesses als String.
    Nutzen wir, um Prozesse zu finden, die im gleichen Container laufen.
    """
    cgroup_file = Path(f"/proc/{pid}/cgroup")
    try:
        return cgroup_file.read_text()
    except FileNotFoundError:
        return ""


def process_uses_gpu(pid: int) -> bool:
    """
    Prüft, ob der Prozess ein File-Descriptor auf /dev/nvidia* offen hat.
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
    Versucht, den „GPU-Prozess“ zu finden, der zum vLLM-Container gehört.
    Strategie:
      - main_pid = PID aus /tmp/vllm.pid
      - alle PIDs in /proc durchgehen
      - nur Prozesse in gleicher cgroup wie main_pid betrachten
      - unter diesen nach einem Prozess suchen, der /dev/nvidia* nutzt
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
            continue  # den API-Server überspringen, der ist selten der GPU-Prozess

        # nur Prozesse im gleichen Container betrachten
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
    Liefert den gecachten GPU-PID oder sucht ihn bei Bedarf neu.
    """
    global GPU_PID
    if GPU_PID is not None:
        return GPU_PID
    GPU_PID = find_gpu_pid()
    return GPU_PID


def is_vllm_ready() -> bool:
    """
    Prüft, ob vLLM im Container bereit ist.
    Bevorzugt /v1/models, weil das ein „echter“ API-Check ist.
    """
    try:
        resp = httpx.get(f"{VLLM_API_URL}/v1/models", timeout=1.0)
        if resp.status_code != 200:
            return False
        # Optional: Modellliste prüfen
        data = resp.json()
        if "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
            return True
        return True  # 200 reicht uns als Minimal-Kriterium
    except Exception as e:
        print(f"[ready] vLLM not ready yet: {e}")
        return False


# --- CUDA-Checkpoint-Wrapper ------------------------------------------------------------

def cuda_get_state() -> str:
    """
    Fragt den CUDA-Checkpoint-State des GPU-Prozesses beim Treiber ab.
    Erwartete Ausgabe: „running“ oder „checkpointed“.
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
    Führt einen Toggle aus und gibt anschließend den neuen CUDA-State zurück.
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

    # Nach erfolgreichem Toggle den neuen State holen
    new_state = cuda_get_state()
    return new_state


def map_cuda_to_logical(cuda_state: str) -> str:
    """
    Übersetzt den CUDA-State („running“, „checkpointed“, …) in unseren
    logischen State für das API.
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
    # Nur: lebt der Service-Prozess?
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
def ready():
    """
    „Ready“ = dieser Container ist grundsätzlich einsatzbereit:
      - vLLM-API ist erreichbar (Modell geladen),
      - GPU-PID gefunden und CUDA-API ansprechbar,
      - initiale einmalige Initialisierung evtl. erledigt.
    Aus Sicht von Docker Compose soll dieser Endpoint
    nach der ersten erfolgreichen Initialisierung immer 200 liefern,
    egal ob der CUDA-State gerade running oder checkpointed ist.
    """
    # 1) vLLM muss (mindestens einmal) ready sein
    if not is_vllm_ready():
        raise HTTPException(status_code=503, detail="vLLM API not ready")

    # 2) Main-PID muss existieren
    if not VLLM_PID_FILE.exists():
        raise HTTPException(status_code=503, detail="vLLM PID file missing")

    # 3) GPU-PID & CUDA-State ermitteln
    cuda_state = cuda_get_state()
    logical = map_cuda_to_logical(cuda_state)

    # 4) Initiale „Einrichtung“ (einmalig)
    if not state.initialized:
        # Wir haben hier ein funktionierendes Setup:
        # vLLM antwortet, GPU-PID valid, get-state funktioniert.
        # Optional: einmaliger Base-Checkpoint, wenn wir sicherstellen wollen,
        # dass nach /ready ein gültiger Snapshot existiert.
        if logical == "RUNNING":
            print("[ready] Initial base checkpoint: toggling CUDA state once")
            new_cuda_state = cuda_toggle()
            cuda_state = new_cuda_state
            logical = map_cuda_to_logical(new_cuda_state)
            state.has_toggled_once = True
        else:
            # Wenn wir schon „SUSPENDED“ sind, war evtl. vorher manuell ein Toggle da.
            state.has_toggled_once = (logical == "SUSPENDED")

        state.state = logical
        state.initialized = True
        print(f"[ready] Initialization complete. state={state.state}, cuda_state={cuda_state}")

    # 5) Ab hier gilt der Service als „ready“, egal ob running oder suspended.
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
    Liefert den aktuellen CUDA-/Logik-Status.
    Hier kann der Controller „active/passive“ unterscheiden.
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
    Semantik: Modell in „checkpointed“ Zustand bringen.
    Idempotent:
      - wenn schon checkpointed → kein Toggle
      - wenn running → toggle → checkpointed
    """
    cuda_state = cuda_get_state()
    s = cuda_state.lower()

    if s == "checkpointed":
        # schon im gewünschten Zustand
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
    Semantik: Modell in „running“ Zustand bringen.
    Idempotent:
      - wenn schon running → kein Toggle
      - wenn checkpointed → toggle → running
    """
    cuda_state = cuda_get_state()
    s = cuda_state.lower()

    if s == "running":
        # schon im gewünschten Zustand
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