"""
router.py — REST API routes.

Endpoints:
  POST /infer                            Upload audio → text + audio (blocking)
  POST /infer/submit                     Submit async job → job_id immediately
  GET  /infer/poll/{job_id}              Poll job status / result
  GET  /health                           Liveness + model readiness
  GET  /audio/{filename}                 Serve a generated audio file
  POST /conversation/new                 Create a multi-turn session
  GET  /conversation/{id}                Get session summary + context stats
  DELETE /conversation/{id}/turn/{idx}   User-initiated turn prune
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
from app.domain.schemas import (
    ConversationSummaryResponse,
    ConversationTurnSchema,
    HealthResponse,
    InferResponse,
)
from app.services.inference import InferenceService
from app.services.registry import ModelRegistry
from app.services.session import SessionManager

router = APIRouter()

# ── In-memory job store (single-user / small-team local deployment) ────────────
# Each entry: {"status": "pending"|"done"|"error", "result": dict, "error": str}
# Not persistent — lost on server restart.  Upgrade to Redis for production.
_jobs: dict[str, dict] = {}


def _run_inference_job(
    service:            InferenceService,
    audio_path:         Path,
    system_prompt:      Optional[str],
    return_audio:       bool,
    history:            list,    # list[HistoryTurn] — [] for single-turn
    delete_audio_after: bool,    # True for temp files, False for session files
) -> dict:
    """
    Synchronous worker — runs in a thread pool via asyncio.to_thread.

    delete_audio_after=True   : single-turn flow (temp file is cleaned up)
    delete_audio_after=False  : multi-turn flow  (session file must stay
                                on disk for future turns to reference it)
    """
    try:
        result = service.infer_from_file(
            input_path    = audio_path,
            system_prompt = system_prompt,
            history       = history,
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
        if delete_audio_after:
            audio_path.unlink(missing_ok=True)


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
    session_mgr = SessionManager(settings=cfg)
    return HealthResponse(
        status                    = "ok",
        model_ready               = registry.is_ready,
        model_path                = str(cfg.model_local_path),
        profile                   = cfg.profile.value,
        model_supports_multi_turn = cfg.model_supports_multi_turn,
        max_audio_minutes         = round(session_mgr.max_audio_minutes(), 2),
        context_window_tokens     = cfg.context_window_tokens,
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
    system_prompt:   Optional[str] = Form(default=None),
    return_audio:    bool          = Form(default=True),
    multi_turn:      bool          = Form(
        default=False,
        description="Enable multi-turn mode. Requires conversation_id on subsequent turns.",
    ),
    conversation_id: Optional[str] = Form(
        default=None,
        description="Existing conversation_id for multi-turn continuation. "
                    "Omit (or leave blank) to start a new conversation.",
    ),
    service: InferenceService = Depends(get_inference_service),
) -> dict:
    """
    Non-blocking inference submission.

    Single-turn (default):  audio is written to a temp file, deleted after inference.
    Multi-turn:             audio is saved to the session directory, kept for future
                            turns. Pass conversation_id to continue an existing session;
                            omit it (or pass blank) to start a new session.

    Returns {"job_id": "...", "status": "pending", "conversation_id": "..."|null}
    Poll GET /infer/poll/{job_id} until status is "done" or "error".
    """
    ct = audio.content_type or ""
    if ct and not any(t in ct for t in ("audio", "octet-stream", "video", "application")):
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {ct}")

    audio_bytes = await audio.read()
    suffix      = Path(audio.filename or "input.wav").suffix or ".wav"

    # ── Resolve audio path and history ───────────────────────────────────
    history:            list  = []
    audio_path:         Path
    delete_audio_after: bool  = True
    active_conv_id:     Optional[str] = None
    turn_index:         int   = 0
    audio_duration_s:   float = 0.0

    if multi_turn:
        from app.core.config import get_settings
        session_mgr = SessionManager(settings=get_settings())

        # Create or continue session
        conv_id = (conversation_id or "").strip() or None
        if conv_id is None:
            conv_id = session_mgr.create_session()
        elif session_mgr.get_session(conv_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation not found: {conv_id}. "
                        "Start a new session by omitting conversation_id.",
            )

        audio_path, turn_index, audio_duration_s = session_mgr.save_input_audio(
            conv_id, audio_bytes
        )
        history             = session_mgr.get_history(conv_id)
        delete_audio_after  = False   # session files must stay on disk
        active_conv_id      = conv_id

    else:
        # Single-turn: write to temp file as before
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            audio_path = Path(tmp.name)

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "pending"}

    async def _run() -> None:
        try:
            result = await asyncio.to_thread(
                _run_inference_job,
                service, audio_path, system_prompt, return_audio,
                history, delete_audio_after,
            )
            # Finalize multi-turn session with the completed response
            if multi_turn and active_conv_id:
                from app.core.config import get_settings
                session_mgr = SessionManager(settings=get_settings())
                completed_turn = session_mgr.finalize_turn(
                    active_conv_id, turn_index, result["text"], audio_duration_s
                )
                result["conversation_id"] = active_conv_id
                result["turn_index"]      = completed_turn.turn_index
                result["turn_summary"]    = {
                    "audio_duration_s": completed_turn.audio_duration_s,
                    "audio_tokens":     completed_turn.audio_tokens,
                    "response_preview": completed_turn.response_text[:120],
                }
            _jobs[job_id] = {"status": "done", "result": result}
        except Exception as exc:  # noqa: BLE001
            _jobs[job_id] = {"status": "error", "error": str(exc)}

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "pending", "conversation_id": active_conv_id}


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


# ── POST /conversation/new ─────────────────────────────────────────────────────

@router.post(
    "/conversation/new",
    tags=["multi-turn"],
    summary="Create a new multi-turn conversation session",
)
async def conversation_new() -> dict:
    """
    Create an empty conversation session and return its conversation_id.
    Pass the conversation_id in subsequent POST /infer/submit requests to
    accumulate context.
    """
    from app.core.config import get_settings
    session_mgr     = SessionManager(settings=get_settings())
    conversation_id = session_mgr.create_session()
    return {"conversation_id": conversation_id}


# ── GET /conversation/{conversation_id} ─────────────────────────────────────

@router.get(
    "/conversation/{conversation_id}",
    response_model=ConversationSummaryResponse,
    tags=["multi-turn"],
    summary="Get context stats and turn list for a conversation session",
)
async def conversation_get(conversation_id: str) -> ConversationSummaryResponse:
    """
    Returns all completed turns and the current context budget usage.
    The UI uses this to render the context meter and the turn prune panel.
    """
    from app.core.config import get_settings
    session_mgr = SessionManager(settings=get_settings())
    stats       = session_mgr.context_stats(conversation_id)

    if session_mgr.get_session(conversation_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation not found: {conversation_id}"
        )

    return ConversationSummaryResponse(
        conversation_id   = conversation_id,
        turns             = [
            ConversationTurnSchema(**t) for t in stats["turns"]
        ],
        used_audio_minutes = stats["used_minutes"],
        max_audio_minutes  = stats["max_minutes"],
        context_used_pct   = stats["context_used_pct"],
    )


# ── DELETE /conversation/{conversation_id}/turn/{turn_index} ──────────────────

@router.delete(
    "/conversation/{conversation_id}/turn/{turn_index}",
    tags=["multi-turn"],
    summary="Delete a specific turn from a conversation session (user-initiated prune)",
)
async def conversation_delete_turn(
    conversation_id: str,
    turn_index:      int,
) -> dict:
    """
    Remove a single turn from the session history.  Deletes the associated
    input audio file.  Turn indices of remaining turns are not renumbered—
    the deleted index is simply gone from future model context.

    Returns {"deleted": true} or 404 if the turn was not found.
    """
    from app.core.config import get_settings
    session_mgr = SessionManager(settings=get_settings())

    if session_mgr.get_session(conversation_id) is None:
        raise HTTPException(
            status_code=404, detail=f"Conversation not found: {conversation_id}"
        )

    deleted = session_mgr.delete_turn(conversation_id, turn_index)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Turn {turn_index} not found in conversation {conversation_id}",
        )
    return {"deleted": True, "conversation_id": conversation_id, "turn_index": turn_index}
