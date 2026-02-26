"""
schemas.py — Pydantic models for the API boundary (HTTP + WebSocket).

These shapes cross the external boundary — they are NOT the internal
domain models (see ports.py). Keep them flat and serialisation-friendly.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── REST: POST /infer ─────────────────────────────────────────────────────────

class InferResponse(BaseModel):
    """
    Response from POST /infer.

    audio_path is the filename (not the full path) relative to the output
    directory. Callers can fetch it via GET /audio/{filename}.
    """
    text: str
    audio_path: Optional[str] = None    # None when return_audio=False
    latency_s: float
    model: str                          # e.g. "Qwen2.5-Omni-3B"


# ── REST: GET /health ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str  = "ok"
    model_ready: bool
    model_path: str
    profile: str
    model_supports_multi_turn: bool  = True
    max_audio_minutes: float         = 20.0
    context_window_tokens: int       = 32768


# ── REST: /conversation/* ─────────────────────────────────────────────────────

class ConversationTurnSchema(BaseModel):
    """Summary of one completed turn, returned by /conversation endpoints."""
    turn_index: int
    audio_duration_s: float
    audio_tokens: int
    response_preview: str            # first 120 chars of model response


class ConversationSummaryResponse(BaseModel):
    """Full session summary: all turns + context budget status."""
    conversation_id: str
    turns: list[ConversationTurnSchema]
    used_audio_minutes: float
    max_audio_minutes: float
    context_used_pct: float          # 0.0 – 1.0


# ── WebSocket: /ws/stream ─────────────────────────────────────────────────────

class WSInferRequest(BaseModel):
    """
    JSON control message the client sends before audio bytes.

    Protocol:
      1. Client → JSON text frame: WSInferRequest
      2. Client → binary frame:   raw PCM float32 bytes at sample_rate
      3. Server → JSON text frame: WSChunk  (one or more, last has is_final=True)
      4. Server → binary frame:   raw PCM float32 bytes (if return_audio=True)
    """
    action: str           = "infer"
    system_prompt: Optional[str] = None
    sample_rate: int      = 16000
    return_audio: bool    = True


class WSChunk(BaseModel):
    """
    Streamed response chunk over WebSocket.

    Text is accumulated across chunks. Audio is sent as a separate
    binary frame immediately after the final chunk.
    """
    text: str    = ""
    is_final: bool = False


class WSError(BaseModel):
    """Error frame sent over WebSocket when inference fails."""
    error: str
