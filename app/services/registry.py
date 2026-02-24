"""
registry.py — ModelRegistry: model lifecycle manager for long-running processes.

Used by the FastAPI server to:
  • Load the model ONCE at startup (not per-request)
  • Share the loaded ModelBackend across all requests
  • Release model memory cleanly at shutdown

The CLI does NOT use this — it creates and destroys its own backend
per invocation, which is the right behaviour for a short-lived process.

In a multi-model future this becomes a dict[str → ModelBackend] keyed
by profile or model name, with logic to swap the active backend.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.adapters.model.qwen_omni import QwenOmniBackend
from app.core.config import Settings, get_settings
from app.domain.ports import ModelBackend

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton-style holder for a loaded ModelBackend.

    Lifecycle:
        registry = ModelRegistry()
        registry.load()          # at server startup — blocks until weights loaded
        backend  = registry.get()  # during requests — instant, no I/O
        registry.unload()        # at server shutdown
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg     = settings or get_settings()
        self._backend: Optional[ModelBackend] = None

    def load(self) -> None:
        """
        Instantiate and initialise the ModelBackend.
        Idempotent — safe to call multiple times.
        Blocks until model weights are fully loaded into device memory.
        """
        if self._backend is not None:
            logger.debug("ModelRegistry: already loaded, skipping.")
            return

        logger.info(
            "ModelRegistry: loading %s (profile=%s)",
            self._cfg.model_local_path.name,
            self._cfg.profile.value,
        )
        self._backend = QwenOmniBackend(settings=self._cfg)
        self._backend.initialize()
        logger.info("ModelRegistry: backend ready.")

    def get(self) -> ModelBackend:
        """
        Return the loaded backend.
        Raises RuntimeError if load() has not been called.
        """
        if self._backend is None:
            raise RuntimeError(
                "ModelRegistry: backend not loaded. "
                "Call ModelRegistry.load() before serving requests."
            )
        return self._backend

    def unload(self) -> None:
        """Release model memory. Called at server shutdown."""
        if self._backend is not None:
            self._backend.cleanup()
            self._backend = None
            logger.info("ModelRegistry: backend unloaded.")

    @property
    def is_ready(self) -> bool:
        """True if the backend is loaded and ready for inference."""
        return self._backend is not None and self._backend.is_ready
