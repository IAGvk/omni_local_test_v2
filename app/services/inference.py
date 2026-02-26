"""
inference.py — InferenceService: the application's primary use case.

This is the layer that both FastAPI routes and the CLI call.
It knows nothing about:
  • HTTP, WebSockets, multipart forms     → API layer's concern
  • torch, transformers, qwen_omni_utils  → adapter's concern
  • file format, WAV encoding             → AudioIO adapter's concern

It orchestrates:
  AudioIO.read(input) → ModelBackend.process(audio) → AudioIO.write(output)

Depends on ports (interfaces) only — concrete adapters are injected
at construction time, making the service trivially unit-testable
by passing in mock ModelBackend and AudioIO objects.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import Settings, get_settings
from app.domain.ports import AudioIO, InferenceResult, ModelBackend

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Application use case: audio-in → text+audio-out.

    Constructor injection pattern:
        service = InferenceService(
            model    = QwenOmniBackend(settings),
            audio_io = LocalAudioIO(),
            settings = settings,
        )
    """

    def __init__(
        self,
        model:    ModelBackend,
        audio_io: AudioIO,
        settings: Optional[Settings] = None,
    ) -> None:
        self._model    = model
        self._audio_io = audio_io
        self._cfg      = settings or get_settings()

    # ── Use case 1: file path in, file path out ───────────────────────────────

    def infer_from_file(
        self,
        input_path:    str | Path,
        system_prompt: Optional[str]        = None,
        output_path:   Optional[str | Path] = None,
        history:       Optional[list]        = None,  # list[HistoryTurn]
    ) -> dict:
        """
        Full pipeline: audio file → text + saved audio file.

        Args:
            input_path    : path to input WAV/FLAC/MP3
            system_prompt : optional override for the default system prompt
            output_path   : where to write the response WAV.
                            Auto-generated in audio_output_dir if None.

        Returns:
            {
                "text"      : str   — model's text response
                "audio_path": str   — absolute path of saved response WAV, or None
                "latency_s" : float — wall-clock seconds for inference
                "model"     : str   — name of the model that processed this
            }
        """
        input_path = Path(input_path)

        # 1. Load audio via the AudioIO port
        audio, sr = self._audio_io.read(input_path)
        logger.info(
            "Loaded  %s  (%.2fs @ %dHz)",
            input_path.name,
            len(audio) / sr,
            sr,
        )

        # 2. Run inference via the ModelBackend port
        result: InferenceResult = self._model.process(
            audio         = audio,
            sample_rate   = sr,
            system_prompt = system_prompt,
            history       = history or [],
        )

        # 3. Persist audio output if present
        saved_path = self._save_audio_if_present(result, output_path)

        return self._build_response(result, saved_path)

    # ── Use case 2: numpy array in (WebSocket / streaming) ───────────────────

    def infer_from_array(
        self,
        audio:         np.ndarray,
        sample_rate:   int,
        system_prompt: Optional[str]        = None,
        output_path:   Optional[str | Path] = None,
        history:       Optional[list]        = None,  # list[HistoryTurn]
    ) -> dict:
        """
        Inference directly from a numpy audio array.

        Used by the WebSocket handler which receives raw PCM bytes and
        converts them to a numpy array before calling this method.
        The raw audio_array is included in the response so the WebSocket
        handler can send it as binary frames without re-reading a file.

        Returns the same dict as infer_from_file, plus:
            "audio_array": np.ndarray | None  — raw PCM for streaming
        """
        result: InferenceResult = self._model.process(
            audio         = audio,
            sample_rate   = sample_rate,
            system_prompt = system_prompt,
            history       = history or [],
        )

        saved_path = self._save_audio_if_present(result, output_path)

        response = self._build_response(result, saved_path)
        response["audio_array"] = result.audio   # raw array for WS binary frame
        return response

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_audio_if_present(
        self,
        result:      InferenceResult,
        output_path: Optional[str | Path],
    ) -> Optional[Path]:
        """Write audio to disk if the model produced it and audio is enabled."""
        if result.audio is None or not self._cfg.return_audio:
            return None

        if output_path is None:
            ts          = int(time.time())
            output_path = self._cfg.audio_output_dir / f"response_{ts}.wav"

        return self._audio_io.write(
            result.audio,
            destination = output_path,
            sample_rate = result.sample_rate,
        )

    def _build_response(
        self,
        result:     InferenceResult,
        saved_path: Optional[Path],
    ) -> dict:
        return {
            "text"      : result.text,
            "audio_path": str(saved_path) if saved_path else None,
            "latency_s" : result.latency_s,
            "model"     : self._cfg.model_local_path.name,
        }
