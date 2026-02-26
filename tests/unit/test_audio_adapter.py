"""
test_audio_adapter.py — Unit tests for LocalAudioIO

No model needed. Tests cover:
  - write → read roundtrip
  - stereo → mono conversion
  - missing file raises FileNotFoundError
  - clip to [-1, 1]
  - resample on read
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.adapters.audio.local_io import LocalAudioIO


@pytest.fixture
def io():
    return LocalAudioIO()


@pytest.fixture
def tmp_wav(tmp_path):
    """Create a temporary WAV file with a 440 Hz tone, return its path."""
    sr    = 16000
    t     = np.linspace(0, 1.0, sr, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    path  = tmp_path / "test.wav"
    sf.write(str(path), audio, sr)
    return path


class TestWrite:
    def test_creates_file(self, io, tmp_path):
        audio = np.zeros(24000, dtype=np.float32)
        path  = tmp_path / "out.wav"
        result = io.write(audio, path, sample_rate=24000)
        assert result.exists()

    def test_creates_parent_dirs(self, io, tmp_path):
        audio = np.zeros(16000, dtype=np.float32)
        path  = tmp_path / "nested" / "deep" / "out.wav"
        io.write(audio, path, sample_rate=16000)
        assert path.exists()

    def test_clips_to_range(self, io, tmp_path):
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        path  = tmp_path / "clipped.wav"
        io.write(audio, path, sample_rate=16000)
        loaded, _ = sf.read(str(path), dtype="float32")
        assert loaded.max() <= 1.0
        assert loaded.min() >= -1.0

    def test_returns_path(self, io, tmp_path):
        audio  = np.zeros(100, dtype=np.float32)
        path   = tmp_path / "out.wav"
        result = io.write(audio, path, sample_rate=16000)
        assert isinstance(result, Path)
        assert result == path


class TestRead:
    def test_roundtrip(self, io, tmp_path):
        sr    = 16000
        audio = np.sin(np.linspace(0, 2 * np.pi, sr)).astype(np.float32)
        path  = tmp_path / "tone.wav"
        sf.write(str(path), audio, sr)
        loaded, loaded_sr = io.read(path, target_sr=sr)
        assert loaded_sr == sr
        assert len(loaded) == len(audio)
        assert loaded.dtype == np.float32

    def test_stereo_to_mono(self, io, tmp_path):
        sr     = 16000
        stereo = np.random.rand(sr, 2).astype(np.float32)
        path   = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, sr)
        mono, loaded_sr = io.read(path, target_sr=sr)
        assert mono.ndim == 1
        assert loaded_sr == sr

    def test_missing_file_raises(self, io):
        with pytest.raises(FileNotFoundError):
            io.read("/nonexistent/audio/file.wav")

    @pytest.mark.filterwarnings("ignore::DeprecationWarning:librosa")
    @pytest.mark.filterwarnings("ignore::FutureWarning:librosa")
    def test_resample(self, io, tmp_path):
        sr_native = 44100
        sr_target = 16000
        t     = np.linspace(0, 1.0, sr_native, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        path  = tmp_path / "44k.wav"
        sf.write(str(path), audio, sr_native)
        loaded, loaded_sr = io.read(path, target_sr=sr_target)
        assert loaded_sr == sr_target
        # Resampled 1s of audio should be within 1% of sr_target samples
        assert abs(len(loaded) - sr_target) / sr_target < 0.01
