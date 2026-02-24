"""
ports.py — Abstract interfaces (ports) for the application core.

This is the "hexagonal heart" of v2. These interfaces define WHAT the
application needs, not HOW it is done. The concrete implementations live
in app/adapters/ and can be swapped freely without touching service logic.

Current implementations:
  ModelBackend  →  app/adapters/model/qwen_omni.py   (Qwen2.5-Omni, MPS + CUDA)
  AudioIO       →  app/adapters/audio/local_io.py    (local filesystem)

Planned:
  ModelBackend  →  app/adapters/model/openai_compat.py  (OpenAI-compatible API)
  AudioIO       →  app/adapters/audio/stream_io.py      (WebSocket byte streams)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import numpy as np


# ── Value objects ──────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """
    What the ModelBackend returns after a full inference pass.

    audio is a float32 mono numpy array at output_sample_rate,
    or None when return_audio=False.
    """
    text: str
    audio: Optional[np.ndarray]  = None   # float32 mono, or None
    sample_rate: int             = 24000  # Qwen2.5-Omni native output
    latency_s: float             = 0.0


@dataclass
class StreamChunk:
    """
    A single chunk yielded during streaming inference.

    The last chunk has is_final=True and may contain the complete text.
    Audio chunks are separate binary frames (not embedded here).
    """
    text: str                    = ""
    audio: Optional[np.ndarray] = None   # float32 mono PCM, or None
    is_final: bool               = False


# ── Port: ModelBackend ─────────────────────────────────────────────────────────

class ModelBackend(ABC):
    """
    Port: anything that can run audio-in → text+audio-out inference.

    The InferenceService depends on this interface only — it never
    imports a concrete adapter directly.
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        Load model weights into memory.
        Idempotent — safe to call multiple times.
        """
        ...

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        system_prompt: Optional[str] = None,
    ) -> InferenceResult:
        """
        Synchronous inference pass.

        Args:
            audio         : float32 mono PCM array
            sample_rate   : sample rate of the input audio
            system_prompt : optional override for the default system prompt

        Returns:
            InferenceResult with text and optional audio
        """
        ...

    async def stream(
        self,
        audio: np.ndarray,
        sample_rate: int,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Streaming inference pass — yields StreamChunks as they are generated.

        Default implementation wraps the synchronous process() call and
        yields the full result as a single final chunk.

        Override this in adapters that support true token-level streaming
        (e.g. when Qwen2.5-Omni streaming API becomes available).
        """
        result = self.process(audio, sample_rate, system_prompt)
        yield StreamChunk(text=result.text, audio=result.audio, is_final=True)

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release model from device memory.
        Called at server shutdown or after CLI use.
        """
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """True if the model is loaded and ready for inference."""
        ...


# ── Port: AudioIO ──────────────────────────────────────────────────────────────

class AudioIO(ABC):
    """
    Port: anything that can read and write audio data.

    Concrete adapters handle where the audio comes from and goes to:
      - LocalAudioIO  : local filesystem (current)
      - StreamAudioIO : WebSocket byte streams (future)
    """

    @abstractmethod
    def read(
        self,
        source: str | Path,
        target_sr: int = 16000,
    ) -> tuple[np.ndarray, int]:
        """
        Load audio from source.

        Args:
            source    : file path or any source identifier
            target_sr : resample to this rate if needed

        Returns:
            (float32 mono array, sample_rate)
        """
        ...

    @abstractmethod
    def write(
        self,
        audio: np.ndarray,
        destination: str | Path,
        sample_rate: int = 24000,
    ) -> Path:
        """
        Persist audio to destination.

        Returns:
            Resolved Path of the written file
        """
        ...
