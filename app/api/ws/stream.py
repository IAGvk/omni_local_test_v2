"""
stream.py — WebSocket endpoint for real-time audio streaming.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CURRENT STATUS: "chunked request / single response"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The protocol is fully specified and ready for token-level streaming.
Today, the model runs a full inference pass and the result is
returned as one final WSChunk (is_final=True). Zero protocol changes
are needed on the client side when streaming is enabled.

STREAMING UPGRADE PATH (for developers / AI agents)
────────────────────────────────────────────────────
The only change required when Qwen2.5-Omni streaming becomes available
in transformers is:

  1. In app/adapters/model/qwen_omni.py:
     Override ModelBackend.stream() with a real async generator that
     yields StreamChunk objects as tokens are produced by the model.

  2. In this file (stream.py):
     Replace the single service.infer_from_array() call in step [3]
     with an `async for chunk in backend.stream(...)` loop that sends
     one WSChunk text frame per yielded chunk.

  Nothing else changes — not InferenceService, not the port interface,
  not the REST endpoints, not the Streamlit UI.

  See ARCHITECTURE.md § "Streaming — Current Status & Upgrade Path"
  for the full explanation and code sketch.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  WebSocket Protocol  ws://host/ws/stream
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CLIENT → SERVER
  ─────────────────────────────────────────────────────
  [1] JSON text frame: WSInferRequest
      {
        "action":        "infer",
        "system_prompt": null,      // or override string
        "sample_rate":   16000,
        "return_audio":  true
      }

  [2] Binary frame: raw PCM bytes
      • dtype  : float32 little-endian
      • layout : mono (single channel)
      • rate   : sample_rate declared in step [1]
      • any duration — the whole utterance at once

  SERVER → CLIENT
  ─────────────────────────────────────────────────────
  [3] JSON text frame: WSChunk (one or more)
      { "text": "...", "is_final": false }
      ...
      { "text": "complete text", "is_final": true }

  [4] Binary frame (if return_audio=true):
      raw PCM float32, mono, 24000 Hz
      (sent immediately after the final JSON chunk)

  ERROR
  ─────────────────────────────────────────────────────
  If inference fails, the server sends:
      { "error": "description" }
  and closes the connection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Python client example (websockets library):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    import asyncio, json
    import numpy as np
    import soundfile as sf
    import websockets

    async def main():
        audio, sr = sf.read("question.wav", dtype="float32")
        async with websockets.connect("ws://localhost:8000/ws/stream") as ws:
            await ws.send(json.dumps({"sample_rate": sr}))
            await ws.send(audio.astype(np.float32).tobytes())
            text_msg = json.loads(await ws.recv())   # WSChunk
            audio_bytes = await ws.recv()             # PCM bytes
            response_audio = np.frombuffer(audio_bytes, dtype=np.float32)

    asyncio.run(main())
"""

from __future__ import annotations

import json
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.adapters.audio.local_io import LocalAudioIO
from app.domain.schemas import WSChunk, WSError, WSInferRequest
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)

ws_router = APIRouter()


@ws_router.websocket("/ws/stream")
async def audio_stream(websocket: WebSocket) -> None:
    """
    WebSocket endpoint: audio-in / audio-out streaming.
    See module docstring for full protocol specification.
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected: %s", client)

    registry = websocket.app.state.registry

    try:
        # ── [1] Receive control message ───────────────────────────────────────
        raw_ctrl = await websocket.receive_text()
        try:
            ctrl = WSInferRequest(**json.loads(raw_ctrl))
        except Exception as e:
            await _send_error(websocket, f"Invalid control message: {e}")
            return

        logger.info(
            "WS [%s] control: sample_rate=%d return_audio=%s",
            client,
            ctrl.sample_rate,
            ctrl.return_audio,
        )

        # ── [2] Receive audio bytes ───────────────────────────────────────────
        raw_bytes = await websocket.receive_bytes()
        audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        duration = len(audio) / ctrl.sample_rate
        logger.info(
            "WS [%s] received %.2fs of audio @ %dHz",
            client,
            duration,
            ctrl.sample_rate,
        )

        # ── [3] Run inference ─────────────────────────────────────────────────
        service = InferenceService(
            model    = registry.get(),
            audio_io = LocalAudioIO(),
        )
        result = service.infer_from_array(
            audio         = audio,
            sample_rate   = ctrl.sample_rate,
            system_prompt = ctrl.system_prompt,
        )

        # ── [4] Send text response ────────────────────────────────────────────
        chunk = WSChunk(text=result["text"], is_final=True)
        await websocket.send_text(chunk.model_dump_json())

        # ── [5] Send audio bytes (if requested and available) ─────────────────
        if ctrl.return_audio and result.get("audio_array") is not None:
            audio_out: np.ndarray = result["audio_array"]
            await websocket.send_bytes(audio_out.astype(np.float32).tobytes())
            logger.info(
                "WS [%s] sent %.2fs of response audio",
                client,
                len(audio_out) / 24000,
            )

        logger.info("WS [%s] inference complete: %.2fs", client, result["latency_s"])

    except WebSocketDisconnect:
        logger.info("WS [%s] disconnected.", client)

    except Exception as exc:
        logger.exception("WS [%s] unhandled error: %s", client, exc)
        await _send_error(websocket, str(exc))
        await websocket.close()


async def _send_error(websocket: WebSocket, message: str) -> None:
    """Send a WSError JSON frame and log it."""
    try:
        err = WSError(error=message)
        await websocket.send_text(err.model_dump_json())
    except Exception:
        pass  # connection may already be closed
