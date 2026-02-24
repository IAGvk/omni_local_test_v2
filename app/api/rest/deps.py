"""
deps.py — FastAPI dependency injection helpers.

Provides get_inference_service() which wires ModelRegistry → InferenceService
with the appropriate adapters. Both the REST router and the WebSocket handler
import from here to keep the wiring in one place.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.adapters.audio.local_io import LocalAudioIO
from app.services.inference import InferenceService
from app.services.registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    """Pull the ModelRegistry stored in app.state at server startup."""
    return request.app.state.registry


def get_inference_service(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> InferenceService:
    """
    Build an InferenceService for a single request.

    The ModelBackend is shared (loaded once in ModelRegistry).
    LocalAudioIO is stateless — safe to create per-request.
    """
    return InferenceService(
        model    = registry.get(),
        audio_io = LocalAudioIO(),
    )
