# Docker Compose Setup for vLLM Checkpointing

This directory contains a Docker Compose-based setup for running multiple vLLM (Very Large Language Model) inference servers with advanced GPU state management via CUDA checkpointing. The system allows efficient switching between different models by suspending and resuming GPU memory states, enabling low-latency model switching without full reloads. The primary purpose is to showcase overutilization of existing GPUs with multiple vLLM containers, introducing a fast switching mechanism by suspending one model and resuming another based on request queues—avoiding the need for dedicated GPUs per workload.

## Architecture Overview

The setup consists of four main services orchestrated via Docker Compose:

### 1. vLLM Backends (`vllm-model-a` and `vllm-model-b`)
- Each backend runs a separate vLLM server instance with a quantized model (AWQ INT4), configured via environment variables for optimal performance on the target hardware.
- Exposed ports: 8001/9001 for A, 8002/9002 for B (vLLM API on 800x, Checkpoint API on 900x).

### 2. Controller (`controller`)
- A FastAPI-based proxy that provides an OpenAI-compatible API ([`/v1/chat/completions`](controller/app.py ), [`/v1/completions`](controller/app.py ), [`/v1/embeddings`](controller/app.py )).
- Manages multiple backends dynamically via autodiscovery.
- Implements **window-based scheduling**: Processes a batch of requests (up to 8 per 500ms window) for the active backend, then switches if needed.
- Handles checkpointing: Suspends the current backend and restores the next one when switching models.
- Tracks queue sizes, backend health, and model status.
- Provides metrics endpoints for GPU/RAM usage.
- Exposed on port 8000.

### 3. Frontend (`frontend`)
- A simple web dashboard built with HTML/CSS/JavaScript (Chart.js for metrics).
- Displays real-time charts for GPU VRAM and RAM usage.
- Shows request queues and model statuses.
- Includes a chat interface to interact with models via the Controller API.
- Served via Nginx on port 3000.

### 4. Shared Components
- **CUDA Checkpointing**: Uses NVIDIA's `cuda-checkpoint` binary to save/restore GPU states, allowing instant model switching without reloading weights.
- **Health Checks**: Each vLLM service has a readiness endpoint ([`/ready`](vllm/checkpoint_service.py )) that ensures vLLM is loaded and CUDA state is initialized.
- **Caching**: Volumes for Hugging Face and vLLM caches to persist model downloads.

## Hardware Parameterization
The setup is parameterized to adapt to different hardware configurations, focusing on overutilization via checkpointing. Configurations for common machine types are provided in the `.env.template` file.

Available configs:
- `Turing 1GPU 16GB VRAM 64GB RAM`: For desktops with Turing GPU, 16GB VRAM, 64GB RAM (uses VLLM v0.10.2).
- `Ampere 1GPU 12GB VRAM 64GB RAM`: For desktops with Ampere GPU, 12GB VRAM, 64GB RAM (uses VLLM v0.11.0).
- `Ampere 2GPU 24GB VRAM 128GB RAM`: For desktops with 2x Ampere GPUs, 24GB total VRAM, 128GB RAM (uses VLLM v0.11.0).

Each config includes tuned settings and can be customized by editing the uncommented values.

## How It Works

1. **Startup**:
   - Docker Compose builds and starts the backends sequentially (A then B).
   - Each backend initializes vLLM and performs an initial CUDA toggle to create a base checkpoint.
   - Controller discovers backends via environment variables and fetches model lists.

2. **Request Handling**:
   - Client requests hit the Controller's OpenAI API.
   - Controller resolves the model to a backend, queues the request, and schedules it.
   - If switching backends, it suspends the current one (checkpoint GPU state) and restores the target.
   - Requests are processed in windows: up to 8 requests per 500ms for the active backend.

3. **Checkpointing**:
   - The Checkpoint Service ([`vllm/checkpoint_service.py`](vllm/checkpoint_service.py )) monitors vLLM's PID, finds the GPU-using process, and calls `cuda-checkpoint` to get/set state.
   - States: "running" (active) or "checkpointed" (suspended).

4. **Frontend Interaction**:
   - Dashboard polls Controller for metrics, queues, and statuses every 500ms.
   - Chat sends streaming requests to [`/v1/chat/completions`](controller/app.py ).

## Prerequisites

- Docker and Docker Compose.
- NVIDIA GPU with CUDA support (runtime: nvidia).
- Sufficient GPU memory (memory utilization is auto-adjusted based on total VRAM to enable overutilization via checkpoints).
- `cuda-checkpoint` tool (built from NVIDIA's GitHub repo in the Dockerfile).

## Usage

1. **Select Configuration**:
   Copy `.env.template` to `.env` and uncomment the section for your machine config. Edit values as needed.
   ```bash
   cp .env.template .env
   # Edit .env: Uncomment the desired config section
   ```

2. **Build and Run**:
   ```bash
   docker-compose up --build
   ```

2. **Access**:
   - Frontend Dashboard: http://localhost:3000
   - Controller API: http://localhost:8000 (e.g., [`/v1/models`](controller/app.py ), [`/status`](vllm/checkpoint_service.py ))
   - Direct Backend APIs: http://localhost:8001 (A), http://localhost:8002 (B)

3. **Environment Variables** (in [`docker-compose.yml`](docker-compose.yml )):
   - `VLLM_VERSION`: vLLM version (default: v0.10.2).
   - Model-specific settings for Model A:
     - `VLLM_USE_V1_A`: Whether to use vLLM API v1 for Model A (0 for v0, 1 for v1).
     - `QUANTIZATION_A`: Quantization method for Model A (default: awq).
     - `DTYPE_A`: Data type for Model A (default: half).
     - `MAX_MODEL_LEN_A`: Maximum sequence length for Model A (default: 8192).
     - `EXTRA_ARGS_A`: Additional vLLM arguments for Model A (optional, e.g., "--trust-remote-code").
   - Model-specific settings for Model B:
     - `VLLM_USE_V1_B`: Whether to use vLLM API v1 for Model B (0 for v0, 1 for v1).
     - `QUANTIZATION_B`: Quantization method for Model B (default: awq).
     - `DTYPE_B`: Data type for Model B (default: half).
     - `MAX_MODEL_LEN_B`: Maximum sequence length for Model B (default: 8192).
     - `EXTRA_ARGS_B`: Additional vLLM arguments for Model B (optional).
   - `GPU_MEM_UTIL`: GPU memory utilization ratio (0.0-1.0).
   - [`BACKEND_CONTAINERS`](controller/app.py ): Comma-separated list of backend names.
   - [`MAX_REQUESTS_PER_WINDOW`](controller/app.py ), [`MAX_WINDOW_DURATION_MS`](controller/app.py ), [`IDLE_SLEEP_MS`](controller/app.py ): Scheduling params.

4. **Testing**:
   - Use the frontend chat or curl the Controller:
     ```bash
     curl -X POST http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{"model": "llama3.2-3b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
     ```

## Key Files

- [`docker-compose.yml`](docker-compose.yml ): Orchestrates all services.
- [`.env.template`](.env.template ): Template with all machine configs as commented sections.
- [`controller/Dockerfile`](controller/Dockerfile ): Builds vLLM image with `cuda-checkpoint`.
- [`vllm/checkpoint_service.py`](vllm/checkpoint_service.py ): FastAPI app for checkpoint management.
- [`vllm/entrypoint.sh`](vllm/entrypoint.sh ): Starts vLLM and Checkpoint Service.
- [`controller/app.py`](controller/app.py ): Main controller logic with scheduling and proxying.
- [`controller/Dockerfile`](controller/Dockerfile ): Builds controller image with CUDA runtime.
- [`frontend/index.html`](frontend/index.html ): Dashboard UI.

## Notes

- Model switching incurs a brief delay for checkpointing (~seconds), enabling overutilization without dedicated GPUs per model.
- Designed for development/testing to showcase checkpointing benefits; production may need scaling adjustments.
- GPUs are shared across containers—do not dedicate specific GPUs to workloads, as the focus is on efficient switching via queues.
- VLLM versions are architecture-specific: v0.10.2 for Turing, v0.11.0 for Ampere (update configs accordingly for new architectures).
- Add custom configs in `configs/` for new machine types by copying and modifying an existing .env.
- Logs are available via `docker-compose logs`.