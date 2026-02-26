"""
main.py — FastAPI application factory.

Startup sequence:
  1. Validate that model weights exist on disk
  2. Load the model into ModelRegistry (blocks until weights are in memory)
  3. Register REST routes and WebSocket endpoint

Run the server:
    # From v2/
    uvicorn app.main:app --host 0.0.0.0 --port 8000

    # Or via Python (reads OMNI_API_HOST / OMNI_API_PORT / OMNI_API_RELOAD):
    python -m app.main

    # Docker:
    docker run -p 8000:8000 \\
      -v $(pwd)/models:/app/models \\
      -e OMNI_PROFILE=mac_mps \\
      omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000

    # With CUDA on a GPU server:
    docker run -p 8000:8000 --gpus all \\
      -v $(pwd)/models:/app/models \\
      -e OMNI_PROFILE=cuda_prod \\
      omni-v2 uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.rest.router import router
from app.api.ws.stream import ws_router
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.services.registry import ModelRegistry

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Everything before 'yield' runs at startup; after 'yield' at shutdown.
    """
    cfg = get_settings()

    # ── Validate model weights before attempting to load ──────────────────────
    if not cfg.model_local_path.exists():
        logger.error(
            "Model weights not found at: %s\n"
            "  Run:  python scripts/download_model.py\n"
            "  Or set OMNI_MODELS_DIR / OMNI_MODEL_SUBDIR to the correct path.",
            cfg.model_local_path,
        )
        sys.exit(1)

    # ── Load model (blocks until weights are in device memory) ────────────────
    registry = ModelRegistry(settings=cfg)
    registry.load()
    app.state.registry = registry

    logger.info(
        "🚀  Omni API ready  profile=%s  model=%s",
        cfg.profile.value,
        cfg.model_local_path.name,
    )

    yield  # ← server is running here

    # ── Shutdown: release device memory ───────────────────────────────────────
    registry.unload()
    logger.info("Server shutdown complete.")


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI instance.
    Separated from the module-level 'app' so it can be called in tests
    with different settings without affecting the singleton.
    """
    application = FastAPI(
        title       = "Omni Audio API",
        description = (
            "Qwen2.5-Omni audio-in / audio-out inference service.\n\n"
            "**POST /infer** — upload an audio file, receive text + audio.\n"
            "**WS /ws/stream** — WebSocket for real-time audio streaming."
        ),
        version  = "2.0.0",
        lifespan = lifespan,
    )

    # Allow the Streamlit UI (default port 8501) and any local origin to reach
    # the API. Tighten allow_origins in production.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_origin_regex=r"http://localhost:\d+",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)
    application.include_router(ws_router)

    return application


# Module-level app instance used by uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = get_settings()
    uvicorn.run(
        "app.main:app",
        host   = cfg.api_host,
        port   = cfg.api_port,
        reload = cfg.api_reload,
    )
