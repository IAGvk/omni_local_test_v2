"""
router.py — REST API routes.

Endpoints:
  POST /infer              Upload audio → text + audio response
  GET  /health             Liveness + model readiness check
  GET  /audio/{filename}   Serve a generated audio file
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.rest.deps import get_inference_service, get_registry
from app.domain.schemas import HealthResponse, InferResponse
from app.services.inference import InferenceService
from app.services.registry import ModelRegistry

router = APIRouter()


# ── GET /health ───────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Liveness and model readiness check",
)
async def health(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> HealthResponse:
    from app.core.config import get_settings
    cfg = get_settings()
    return HealthResponse(
        status      = "ok",
        model_ready = registry.is_ready,
        model_path  = str(cfg.model_local_path),
        profile     = cfg.profile.value,
    )


# ── POST /infer ───────────────────────────────────────────────────────────────

@router.post(
    "/infer",
    response_model=InferResponse,
    tags=["inference"],
    summary="Audio file → text + audio response",
)
async def infer(
    audio: UploadFile = File(
        ...,
        description="Audio file (WAV, FLAC, MP3). Mono or stereo, any sample rate.",
    ),
    system_prompt: Optional[str] = Form(
        default=None,
        description="Override the default system prompt.",
    ),
    return_audio: bool = Form(
        default=True,
        description="Set False for text-only (faster, no audio generation).",
    ),
    service: InferenceService = Depends(get_inference_service),
) -> InferResponse:
    """
    Upload an audio file and receive the model's text and audio response.

    The audio file is saved to a temporary path, processed, then deleted.
    The response audio (if requested) is saved in audio/output/ and
    accessible via GET /audio/{filename}.
    """
    # Validate content type loosely — multipart uploads vary in MIME
    ct = audio.content_type or ""
    if ct and not any(t in ct for t in ("audio", "octet-stream", "video", "application")):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {ct}. Send an audio file.",
        )

    # Write upload to temp file so LocalAudioIO.read() can handle it
    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)

    try:
        result = service.infer_from_file(
            input_path    = tmp_path,
            system_prompt = system_prompt,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Return only the filename (not the full server path) for audio_path
    audio_filename = None
    if result["audio_path"]:
        audio_filename = Path(result["audio_path"]).name

    return InferResponse(
        text       = result["text"],
        audio_path = audio_filename,
        latency_s  = result["latency_s"],
        model      = result["model"],
    )


# ── GET /audio/{filename} ─────────────────────────────────────────────────────

@router.get(
    "/audio/{filename}",
    tags=["inference"],
    summary="Download a generated audio response file",
)
async def serve_audio(filename: str):
    """
    Serve a generated WAV file from the output directory.

    The filename is returned in the audio_path field of InferResponse.
    Example: GET /audio/response_1740000000.wav
    """
    from app.core.config import get_settings
    cfg  = get_settings()
    path = cfg.audio_output_dir / filename

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {filename}")

    return FileResponse(str(path), media_type="audio/wav", filename=filename)
