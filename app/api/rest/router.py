"""
router.py — REST API routes.

Endpoints:
  POST /infer              Upload audio → text + audio response (blocking)
  POST /infer/submit       Upload audio → job_id immediately (non-blocking)
  GET  /infer/poll/{id}    Check job status / retrieve result
  GET  /health             Liveness + model readiness check
  GET  /audio/{filename}   Serve a generated audio file
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.rest.deps import get_inference_service, get_registry
from app.domain.schemas import HealthResponse, InferResponse
from app.services.inference import InferenceService
from app.services.registry import ModelRegistry

router = APIRouter()

# ── In-memory job store (single-user local deployment) ────────────────────────
# Each entry: {"status": "pending" | "done" | "error", "result": dict, "error": str}
_jobs: dict[str, dict] = {}


def _run_inference_job(
    service:       InferenceService,
    tmp_path:      Path,
    system_prompt: Optional[str],
    return_audio:  bool,
) -> dict:
    """
    Synchronous worker — runs in a thread pool via asyncio.to_thread.
    Always deletes tmp_path on completion or error.
    """
    try:
        result = service.infer_from_file(
            input_path    = tmp_path,
            system_prompt = system_prompt,
        )
        audio_filename = None
        if result["audio_path"] and return_audio:
            audio_filename = Path(result["audio_path"]).name
        return {
            "text"      : result["text"],
            "audio_path": audio_filename,
            "latency_s" : result["latency_s"],
            "model"     : result["model"],
        }
    finally:
        tmp_path.unlink(missing_ok=True)


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

    # Return only the filename (not the full server path) for audio_path.
    # If the caller set return_audio=False, suppress the audio path so the
    # client knows not to expect audio (the file may still exist on disk
    # if the server config has OMNI_RETURN_AUDIO=true).
    audio_filename = None
    if result["audio_path"] and return_audio:
        audio_filename = Path(result["audio_path"]).name

    return InferResponse(
        text       = result["text"],
        audio_path = audio_filename,
        latency_s  = result["latency_s"],
        model      = result["model"],
    )


# ── POST /infer/submit ────────────────────────────────────────────────────────

@router.post(
    "/infer/submit",
    tags=["inference"],
    summary="Submit audio for async inference — returns job_id immediately",
)
async def infer_submit(
    audio: UploadFile = File(
        ...,
        description="Audio file (WAV, FLAC, MP3). Mono or stereo, any sample rate.",
    ),
    system_prompt: Optional[str] = Form(default=None),
    return_audio:  bool          = Form(default=True),
    service: InferenceService    = Depends(get_inference_service),
) -> dict:
    """
    Non-blocking version of POST /infer.

    Returns {"job_id": "...", "status": "pending"} immediately.
    Poll GET /infer/poll/{job_id} until status is "done" or "error".
    """
    ct = audio.content_type or ""
    if ct and not any(t in ct for t in ("audio", "octet-stream", "video", "application")):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {ct}")

    suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = Path(tmp.name)

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "pending"}

    async def _run() -> None:
        try:
            result = await asyncio.to_thread(
                _run_inference_job, service, tmp_path, system_prompt, return_audio
            )
            _jobs[job_id] = {"status": "done", "result": result}
        except Exception as exc:  # noqa: BLE001
            _jobs[job_id] = {"status": "error", "error": str(exc)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "pending"}


# ── GET /infer/poll/{job_id} ──────────────────────────────────────────────────

@router.get(
    "/infer/poll/{job_id}",
    tags=["inference"],
    summary="Poll for the result of a submitted inference job",
)
async def infer_poll(job_id: str) -> dict:
    """
    Returns one of:
      {"status": "pending"}
      {"status": "done",  "result": {text, audio_path, latency_s, model}}
      {"status": "error", "error": "...message..."}
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job



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
