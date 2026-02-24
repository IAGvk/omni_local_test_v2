FROM python:3.11-slim

WORKDIR /app

# ─── System deps ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# ─── Python deps ─────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Install qwen-omni-utils from source ─────────────────────────────────────
RUN pip install --no-cache-dir \
    git+https://github.com/QwenLM/qwen-omni-utils.git

# ─── Copy source ─────────────────────────────────────────────────────────────
COPY . .

# ─── Runtime directories (weights and audio are volume-mounted) ──────────────
RUN mkdir -p audio/input audio/output models

# ─────────────────────────────────────────────────────────────────────────────
#  Default CMD: run unit smoke tests (no model weights needed)
#  Override at 'docker run' time for other uses — see examples below.
# ─────────────────────────────────────────────────────────────────────────────
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# ─── Usage examples ───────────────────────────────────────────────────────────
#
# Build:
#   docker build -t omni-v2 .
#
# Run smoke tests (no model, no GPU):
#   docker run --rm omni-v2
#
# Run the FastAPI server (MPS not available in Docker — use CUDA or CPU):
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/audio:/app/audio \
#     -e OMNI_PROFILE=cuda_prod \
#     omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# Run with CUDA GPU:
#   docker run --rm --gpus all -p 8000:8000 \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/audio:/app/audio \
#     -e OMNI_PROFILE=cuda_prod \
#     omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000
#
# CLI one-shot (with mounted weights):
#   docker run --rm \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/audio:/app/audio \
#     omni-v2 python -m cli.main --test-tone
