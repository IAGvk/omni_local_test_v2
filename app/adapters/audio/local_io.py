"""
local_io.py — AudioIO adapter for the local filesystem.

Reads and writes audio files on disk.
Supports WAV, FLAC, OGG natively; falls back to librosa for MP3.

To add WebSocket streaming support in the future:
  1. Create app/adapters/audio/stream_io.py
  2. Implement the AudioIO port against byte streams
  3. Inject it in the WebSocket handler instead of LocalAudioIO
  — zero changes to InferenceService or the model adapter needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.domain.ports import AudioIO

logger = logging.getLogger(__name__)


class LocalAudioIO(AudioIO):
    """
    Read/write audio files from/to the local filesystem.
    """

    # ── AudioIO port ──────────────────────────────────────────────────────────

    def read(
        self,
        source: str | Path,
        target_sr: int = 16000,
    ) -> tuple[np.ndarray, int]:
        """
        Load an audio file → float32 mono numpy array.

        Resamples to target_sr if the file's native rate differs.
        Falls back to librosa for formats soundfile can't handle (MP3, AAC).
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            import librosa  # type: ignore
            audio, sr = librosa.load(str(path), sr=None, mono=True)

        # Stereo → mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            import librosa  # type: ignore
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        logger.debug("Loaded %s  (%.2fs @ %dHz)", path.name, len(audio) / sr, sr)
        return audio.astype(np.float32), sr

    def write(
        self,
        audio: np.ndarray,
        destination: str | Path,
        sample_rate: int = 24000,
    ) -> Path:
        """
        Save float32 audio to a WAV file.
        Creates parent directories automatically.
        Clips to [-1, 1] to prevent WAV encoding artifacts.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)

        audio = audio.reshape(-1).astype(np.float32)
        audio = np.clip(audio, -1.0, 1.0)

        sf.write(str(path), audio, samplerate=sample_rate, subtype="PCM_16")
        logger.info(
            "Audio saved → %s  (%.2fs @ %dHz)",
            path,
            len(audio) / sample_rate,
            sample_rate,
        )
        return path
