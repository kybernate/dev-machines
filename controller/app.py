import os
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import subprocess
import psutil
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("controller")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="vLLM Multi-Model Controller",
    version="0.3.0",
    description=(
        "OpenAI-compatible proxy with window-based scheduling across multiple "
        "vLLM backends, autodiscovery via /v1/models, including streaming."
    ),
)

# CORS for the frontend (e.g., http://localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok for dev; restrict later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# List of backend containers, comma-separated:
#   BACKEND_CONTAINERS="vllm-model-a,vllm-model-b"
BACKEND_CONTAINERS = os.getenv("BACKEND_CONTAINERS", "vllm-model-a,vllm-model-b")

# Window parameters
MAX_REQUESTS_PER_WINDOW = int(os.getenv("MAX_REQUESTS_PER_WINDOW", "8"))
MAX_WINDOW_DURATION_MS = int(os.getenv("MAX_WINDOW_DURATION_MS", "500"))
IDLE_SLEEP_MS = int(os.getenv("IDLE_SLEEP_MS", "20"))


@dataclass
class BackendConfig:
    name: str
    vllm_base_url: str
    checkpoint_base_url: str
    models: Set[str] = field(default_factory=set)


# Backends are autodiscovered at startup
backends: Dict[str, BackendConfig] = {}

# Map from Model-ID → Backend-Key
model_alias_to_backend: Dict[str, str] = {}

# One queue per backend
queues: Dict[str, "asyncio.Queue[QueuedRequest]"] = {}

# Global HTTP client
client: Optional[httpx.AsyncClient] = None

# Which backend is currently active?
active_backend: Optional[str] = None

scheduler_started = False


@dataclass
class QueuedRequest:
    backend_key: str
    body: Dict[str, Any]
    future: Optional[asyncio.Future]  # for non-stream
    is_stream: bool = False
    stream_queue: Optional["asyncio.Queue[Optional[bytes]]"] = None


# -----------------------------------------------------------------------------
# Pydantic models for OpenAPI documentation
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Role of the message in the chat."
    )
    content: str = Field(..., description="Content of the message.")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID as reported by /v1/models.")
    messages: List[ChatMessage] = Field(
        ..., description="Conversation as a list of messages."
    )
    stream: Optional[bool] = Field(
        False,
        description="If true, a streaming response in OpenAI format is expected.",
    )
    temperature: Optional[float] = Field(
        0.7, description="Sampling-Temperature."
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens in the response."
    )


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID as reported by /v1/models.")
    prompt: str = Field(..., description="Prompt text.")
    stream: Optional[bool] = Field(
        False,
        description="If true, a streaming response in OpenAI format is expected.",
    )
    temperature: Optional[float] = Field(0.7)
    max_tokens: Optional[int] = Field(None)


class EmbeddingsRequest(BaseModel):
    model: str = Field(..., description="Embedding model ID.")
    input: List[str] = Field(..., description="List of texts to be embedded.")


class ChatCompletionChoiceMock(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponseMock(BaseModel):
    id: str = "chatcmpl-mock"
    object: str = "chat.completion"
    model: str
    choices: List[ChatCompletionChoiceMock]


class CompletionChoiceMock(BaseModel):
    index: int
    text: str
    finish_reason: Optional[str] = None


class CompletionResponseMock(BaseModel):
    id: str = "cmpl-mock"
    object: str = "text_completion"
    model: str
    choices: List[CompletionChoiceMock]


class EmbeddingDataMock(BaseModel):
    index: int
    embedding: List[float]


class EmbeddingsResponseMock(BaseModel):
    object: str = "list"
    data: List[EmbeddingDataMock]
    model: str


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_backend_containers() -> List[str]:
    raw = BACKEND_CONTAINERS.strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


async def fetch_models_from_backend(cfg: BackendConfig) -> List[str]:
    """
    Fetches the list of models from a backend via /v1/models.
    """
    url = f"{cfg.vllm_base_url}/v1/models"
    try:
        resp = await client.get(url, timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        models: List[str] = []
        for m in data.get("data", []):
            mid = m.get("id")
            if isinstance(mid, str):
                models.append(mid)
        return models
    except httpx.HTTPError as e:
        logger.error("Failed to fetch models from %s: %s", cfg.name, e)
        return []


async def discover_backends():
    """
    Dynamically builds the backend list based on the container list,
    fetches /v1/models from each backend and creates:
      - backends[name]
      - queues[name]
      - model_alias_to_backend
    """
    global backends, queues, model_alias_to_backend

    containers = parse_backend_containers()
    if not containers:
        logger.warning("No BACKEND_CONTAINERS configured, controller will be idle.")
        return

    logger.info("Discovering backends from containers: %s", containers)

    for name in containers:
        vllm_base_url = f"http://{name}:8000"
        checkpoint_base_url = f"http://{name}:9000"
        cfg = BackendConfig(
            name=name,
            vllm_base_url=vllm_base_url,
            checkpoint_base_url=checkpoint_base_url,
        )

        models = await fetch_models_from_backend(cfg)
        cfg.models.update(models)
        backends[name] = cfg
        queues[name] = asyncio.Queue()

        for m in models:
            if m not in model_alias_to_backend:
                model_alias_to_backend[m] = name

    logger.info("Discovered %d backends", len(backends))
    for name, cfg in backends.items():
        logger.info("Backend %s: models=%s", name, sorted(cfg.models))
    logger.info("Model alias mapping: %s", model_alias_to_backend)


def resolve_backend_key(requested_model: str) -> Optional[str]:
    if not requested_model:
        return None
    return model_alias_to_backend.get(requested_model)


def pick_next_backend() -> Optional[str]:
    """
    Simple Policy:
      - if active_backend has pending requests -> stay with it
      - otherwise: first backend with non-empty queue
    """
    global active_backend
    if active_backend and active_backend in queues and not queues[active_backend].empty():
        return active_backend

    for key, q in queues.items():
        if not q.empty():
            return key
    return None


async def call_checkpoint_api(backend_key: str, action: str) -> Dict[str, Any]:
    """
    Calls /checkpoint/dump or /checkpoint/restore on the checkpoint service of a backend.
    """
    assert action in ("dump", "restore"), "invalid checkpoint action"
    cfg = backends[backend_key]
    url = f"{cfg.checkpoint_base_url}/checkpoint/{action}"
    try:
        resp = await client.post(url, json={}, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        logger.info(
            "Checkpoint %s on %s: state=%s cuda_state=%s",
            action,
            backend_key,
            data.get("state"),
            data.get("cuda_state"),
        )
        return data
    except httpx.HTTPError as e:
        logger.error("Failed to call checkpoint %s on %s: %s", action, backend_key, e)
        raise HTTPException(
            status_code=502,
            detail=f"checkpoint {action} failed for {backend_key}: {e}",
        )


async def ensure_backend_running(backend_key: str) -> None:
    await call_checkpoint_api(backend_key, "restore")


async def ensure_backend_suspended(backend_key: str) -> None:
    await call_checkpoint_api(backend_key, "dump")


async def process_backend_request(backend_key: str, qreq: QueuedRequest, endpoint: str) -> None:
    """
    Performs the HTTP call to the vLLM backend:
      - for non-stream: complete JSON response
      - for stream: chunked streaming via stream_queue
    """
    cfg = backends[backend_key]

    if endpoint == "chat":
        url = f"{cfg.vllm_base_url}/v1/chat/completions"
    elif endpoint == "completions":
        url = f"{cfg.vllm_base_url}/v1/completions"
    elif endpoint == "embeddings":
        url = f"{cfg.vllm_base_url}/v1/embeddings"
    else:
        # should not happen
        err = HTTPException(status_code=500, detail="Unknown endpoint type")
        if qreq.future and not qreq.future.done():
            qreq.future.set_exception(err)
        return

    is_stream = bool(qreq.is_stream)

    # Embeddings typically do not stream, we always handle them as non-stream
    if endpoint == "embeddings":
        is_stream = False

    try:
        if not is_stream:
            # Non-Streaming: usual JSON proxy
            resp = await client.post(url, json=qreq.body, timeout=None)
            resp.raise_for_status()
            data = resp.json()
            if qreq.future and not qreq.future.done():
                qreq.future.set_result(data)
        else:
            # Streaming: we stream the bytes 1:1 through
            if qreq.stream_queue is None:
                logger.error("stream_queue missing for streaming request")
                if qreq.future and not qreq.future.done():
                    qreq.future.set_exception(
                        HTTPException(status_code=500, detail="Internal streaming error")
                    )
                return

            async with client.stream("POST", url, json=qreq.body, timeout=None) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    # pass chunk unchanged to the client
                    await qreq.stream_queue.put(chunk)

            # Ende markieren
            await qreq.stream_queue.put(None)

    except httpx.HTTPStatusError as e:
        logger.error("Backend %s returned HTTP error: %s", backend_key, e)
        if is_stream and qreq.stream_queue is not None:
            # Stream error as JSON chunk and end
            error_json = (
                f'data: {{"error": "Backend {backend_key} HTTP {e.response.status_code}"}}\n\n'
            ).encode("utf-8")
            await qreq.stream_queue.put(error_json)
            await qreq.stream_queue.put(b"data: [DONE]\n\n")
            await qreq.stream_queue.put(None)
        elif qreq.future and not qreq.future.done():
            qreq.future.set_exception(
                HTTPException(
                    status_code=e.response.status_code,
                    detail=f"Backend {backend_key} error: {e.response.text}",
                )
            )
    except httpx.HTTPError as e:
        logger.error("Backend %s request failed: %s", backend_key, e)
        if is_stream and qreq.stream_queue is not None:
            error_json = (
                f'data: {{"error": "Backend {backend_key} unreachable"}}\n\n'
            ).encode("utf-8")
            await qreq.stream_queue.put(error_json)
            await qreq.stream_queue.put(b"data: [DONE]\n\n")
            await qreq.stream_queue.put(None)
        elif qreq.future and not qreq.future.done():
            qreq.future.set_exception(
                HTTPException(
                    status_code=502,
                    detail=f"Backend {backend_key} unreachable: {e}",
                )
            )
    except Exception as e:
        logger.exception(
            "Unexpected error processing request on %s: %s", backend_key, e
        )
        if is_stream and qreq.stream_queue is not None:
            error_json = (
                f'data: {{"error": "Internal controller error"}}\n\n'
            ).encode("utf-8")
            await qreq.stream_queue.put(error_json)
            await qreq.stream_queue.put(b"data: [DONE]\n\n")
            await qreq.stream_queue.put(None)
        elif qreq.future and not qreq.future.done():
            qreq.future.set_exception(
                HTTPException(status_code=500, detail="Internal controller error")
            )


# -----------------------------------------------------------------------------
# Scheduler-Loop (Window-based)
# -----------------------------------------------------------------------------

async def scheduler_loop():
    global active_backend

    logger.info(
        "Starting scheduler loop: MAX_REQUESTS_PER_WINDOW=%d, MAX_WINDOW_DURATION_MS=%d",
        MAX_REQUESTS_PER_WINDOW,
        MAX_WINDOW_DURATION_MS,
    )

    while True:
        try:
            if not backends:
                await asyncio.sleep(0.5)
                continue

            backend_to_serve = pick_next_backend()
            if backend_to_serve is None:
                await asyncio.sleep(IDLE_SLEEP_MS / 1000.0)
                continue

            if active_backend != backend_to_serve:
                # suspend old backend
                if active_backend is not None and active_backend in backends:
                    logger.info("Suspending backend %s", active_backend)
                    await ensure_backend_suspended(active_backend)
                # restore new backend
                logger.info("Restoring backend %s", backend_to_serve)
                await ensure_backend_running(backend_to_serve)
                active_backend = backend_to_serve
                logger.info("Active backend is now %s", active_backend)

            start = time.monotonic()
            served = 0
            q = queues[backend_to_serve]

            # Window for this backend
            while True:
                elapsed_ms = (time.monotonic() - start) * 1000.0
                if served >= MAX_REQUESTS_PER_WINDOW or elapsed_ms >= MAX_WINDOW_DURATION_MS:
                    break

                try:
                    # We pack the endpoint type into qreq.body["_endpoint_type"]
                    qreq: QueuedRequest = q.get_nowait()
                except asyncio.QueueEmpty:
                    break

                endpoint_type = qreq.body.pop("_endpoint_type", "chat")
                await process_backend_request(backend_to_serve, qreq, endpoint_type)
                served += 1

            await asyncio.sleep(0)

        except Exception as e:
            logger.exception("Error in scheduler loop: %s", e)
            await asyncio.sleep(1.0)


# -----------------------------------------------------------------------------
# FastAPI Lifecycle
# -----------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global client, scheduler_started
    client = httpx.AsyncClient(timeout=None)

    await discover_backends()

    if not scheduler_started:
        asyncio.create_task(scheduler_loop())
        scheduler_started = True
        logger.info("Scheduler task started.")


@app.on_event("shutdown")
async def on_shutdown():
    global client
    if client is not None:
        await client.aclose()


# -----------------------------------------------------------------------------
# API-Endpunkte: Meta / Status
# -----------------------------------------------------------------------------

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/status")
async def controller_status():
    """
    Provides:
      - active backend
      - queue size per backend
      - checkpoint status from /status of the checkpoint service
      - derived model status
    """
    backend_status: Dict[str, Any] = {}

    for name, cfg in backends.items():
        qsize = queues[name].qsize() if name in queues else 0
        chk_status: Dict[str, Any] = {}
        try:
            resp = await client.get(f"{cfg.checkpoint_base_url}/status", timeout=2.0)
            resp.raise_for_status()
            chk_status = resp.json()
        except Exception as e:
            chk_status = {"error": str(e)}

        backend_status[name] = {
            "vllm_base_url": cfg.vllm_base_url,
            "checkpoint_base_url": cfg.checkpoint_base_url,
            "queue_size": qsize,
            "checkpoint_status": chk_status,
        }

    # Modellstatus ableiten
    models_info: Dict[str, Any] = {}
    for mid, backend_key in model_alias_to_backend.items():
        bstat = backend_status.get(backend_key, {})
        chk = bstat.get("checkpoint_status", {})
        cuda_state = str(chk.get("cuda_state", "")).lower()
        if backend_key == active_backend:
            if cuda_state == "running":
                mstate = "ACTIVE"
            elif cuda_state == "checkpointed":
                # should rarely happen, then bug
                mstate = "INCONSISTENT"
            else:
                mstate = "UNKNOWN"
        else:
            if cuda_state == "checkpointed":
                mstate = "SUSPENDED"
            elif cuda_state == "running":
                mstate = "ACTIVE_BUT_NOT_SELECTED"
            else:
                mstate = "UNKNOWN"

        models_info[mid] = {
            "backend": backend_key,
            "status": mstate,
            "cuda_state": cuda_state,
        }

    return {
        "active_backend": active_backend,
        "backends": backend_status,
        "models": models_info,
    }

def get_gpu_memory_summary() -> Dict[str, int]:
    """
    Reads VRAM usage across all visible GPUs using nvidia-smi.
    Expects the controller container to have GPU access (runtime: nvidia).
    """
    total_mb = 0
    used_mb = 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                t = int(parts[0])
                u = int(parts[1])
                total_mb += t
                used_mb += u
    except Exception as e:
        logger.error("Failed to read GPU memory via nvidia-smi: %s", e)
    return {"total_mb": total_mb, "used_mb": used_mb}


def get_ram_summary() -> Dict[str, int]:
    """
    Total RAM vs. used (from the host's perspective, as far as psutil sees in the container).
    """
    vm = psutil.virtual_memory()
    total_mb = int(vm.total / (1024 * 1024))
    used_mb = int((vm.total - vm.available) / (1024 * 1024))
    return {"total_mb": total_mb, "used_mb": used_mb}


@app.get("/metrics/resources")
async def metrics_resources():
    """
    Provides aggregated VRAM and RAM metrics for the dashboard.
    """
    gpu = get_gpu_memory_summary()
    ram = get_ram_summary()
    return {
        "gpu": gpu,
        "ram": ram,
    }


# -----------------------------------------------------------------------------
# API-Endpunkte: OpenAI-kompatibel
# -----------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    """
    Aggregates all models from all backends.
    The IDs are exactly those reported by vLLM in /v1/models.
    """
    models: List[Dict[str, Any]] = []
    for mid, backend_key in model_alias_to_backend.items():
        models.append(
            {
                "id": mid,
                "object": "model",
                "owned_by": backend_key,
            }
        )
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    backend_key = resolve_backend_key(model_id)
    if backend_key is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "id": model_id,
        "object": "model",
        "owned_by": backend_key,
    }


@app.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponseMock,
    responses={200: {"description": "Chat completion or streaming response"}},
)
async def chat_completions(body: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completion endpoint.
    Supports stream=true (Server-Sent Events like OpenAI/vLLM).
    """
    requested_model = body.model
    backend_key = resolve_backend_key(requested_model)
    if backend_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{requested_model}'. "
                f"Available models: {list(model_alias_to_backend.keys())}"
            ),
        )

    is_stream = bool(body.stream)

    if not is_stream:
        # Non-streaming path with Future
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        payload = body.model_dump()
        payload["_endpoint_type"] = "chat"
        qreq = QueuedRequest(
            backend_key=backend_key,
            body=payload,
            future=fut,
            is_stream=False,
            stream_queue=None,
        )
        await queues[backend_key].put(qreq)

        try:
            result = await fut
            return JSONResponse(content=result)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.exception("Error waiting for chat completion result: %s", e)
            raise HTTPException(
                status_code=500, detail="Controller failed to process request"
            )

    # Streaming path
    stream_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    payload = body.model_dump()
    payload["_endpoint_type"] = "chat"
    qreq = QueuedRequest(
        backend_key=backend_key,
        body=payload,
        future=None,
        is_stream=True,
        stream_queue=stream_queue,
    )
    await queues[backend_key].put(qreq)

    async def event_stream():
        try:
            while True:
                chunk = await stream_queue.get()
                if chunk is None:
                    break
                # chunk is already what vLLM generates (SSE format)
                yield chunk
        except Exception as e:
            logger.exception("Error in streaming generator: %s", e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post(
    "/v1/completions",
    response_model=CompletionResponseMock,
    responses={200: {"description": "Legacy completion or streaming response"}},
)
async def completions(body: CompletionRequest, request: Request):
    """
    OpenAI-compatible /v1/completions endpoint.
    Passed 1:1 to the backend.
    """
    requested_model = body.model
    backend_key = resolve_backend_key(requested_model)
    if backend_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{requested_model}'. "
                f"Available models: {list(model_alias_to_backend.keys())}"
            ),
        )

    is_stream = bool(body.stream)

    if not is_stream:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        payload = body.model_dump()
        payload["_endpoint_type"] = "completions"
        qreq = QueuedRequest(
            backend_key=backend_key,
            body=payload,
            future=fut,
            is_stream=False,
            stream_queue=None,
        )
        await queues[backend_key].put(qreq)

        try:
            result = await fut
            return JSONResponse(content=result)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.exception("Error waiting for completion result: %s", e)
            raise HTTPException(
                status_code=500, detail="Controller failed to process request"
            )

    # Streaming path
    stream_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    payload = body.model_dump()
    payload["_endpoint_type"] = "completions"
    qreq = QueuedRequest(
        backend_key=backend_key,
        body=payload,
        future=None,
        is_stream=True,
        stream_queue=stream_queue,
    )
    await queues[backend_key].put(qreq)

    async def event_stream():
        try:
            while True:
                chunk = await stream_queue.get()
                if chunk is None:
                    break
                yield chunk
        except Exception as e:
            logger.exception("Error in streaming generator (completions): %s", e)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post(
    "/v1/embeddings",
    response_model=EmbeddingsResponseMock,
    responses={200: {"description": "Embedding response (non-stream)"}},
)
async def embeddings(body: EmbeddingsRequest, request: Request):
    """
    OpenAI-compatible /v1/embeddings endpoint.
    Proxied directly as JSON response (no streaming).
    """
    requested_model = body.model
    backend_key = resolve_backend_key(requested_model)
    if backend_key is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown model '{requested_model}'. "
                f"Available models: {list(model_alias_to_backend.keys())}"
            ),
        )

    loop = asyncio.get_event_loop()
    fut: asyncio.Future = loop.create_future()
    payload = body.model_dump()
    payload["_endpoint_type"] = "embeddings"
    qreq = QueuedRequest(
        backend_key=backend_key,
        body=payload,
        future=fut,
        is_stream=False,
        stream_queue=None,
    )
    await queues[backend_key].put(qreq)

    try:
        result = await fut
        return JSONResponse(content=result)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Error waiting for embeddings result: %s", e)
        raise HTTPException(
            status_code=500, detail="Controller failed to process request"
        )