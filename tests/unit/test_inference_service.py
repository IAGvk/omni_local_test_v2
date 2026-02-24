"""
test_inference_service.py — Unit tests for InferenceService

This is the key payoff of having a ports layer:
  • ModelBackend is mocked with unittest.mock.MagicMock
  • InferenceService is tested in full isolation — no model weights needed
  • Tests verify the service's orchestration logic, not the model itself

Tests cover:
  - infer_from_file: text response, no audio
  - infer_from_file: text + audio saved to disk
  - infer_from_array: audio_array included in response
  - infer_from_file: missing input file raises FileNotFoundError
  - Output path auto-generated when not specified
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from app.adapters.audio.local_io import LocalAudioIO
from app.core.config import Settings
from app.domain.ports import InferenceResult
from app.services.inference import InferenceService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_silence_wav(path: Path, sr: int = 16000, duration: float = 0.1) -> Path:
    """Write a tiny silent WAV file for use as a fake input."""
    audio = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def input_wav(tmp_path):
    return _write_silence_wav(tmp_path / "input.wav")


@pytest.fixture
def mock_backend_text_only():
    """A mock ModelBackend that returns text with no audio."""
    m = MagicMock()
    m.is_ready = True
    m.process.return_value = InferenceResult(
        text      = "Mock text response",
        audio     = None,
        latency_s = 0.1,
    )
    return m


@pytest.fixture
def mock_backend_with_audio():
    """A mock ModelBackend that returns text + audio."""
    m = MagicMock()
    m.is_ready = True
    m.process.return_value = InferenceResult(
        text        = "Mock audio response",
        audio       = np.zeros(24000, dtype=np.float32),  # 1s silence
        sample_rate = 24000,
        latency_s   = 0.5,
    )
    return m


# ── Tests: infer_from_file ────────────────────────────────────────────────────

class TestInferFromFile:
    def test_text_only_response(self, mock_backend_text_only, input_wav, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path, return_audio=True)
        service = InferenceService(
            model    = mock_backend_text_only,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        result = service.infer_from_file(input_path=input_wav)

        assert result["text"]       == "Mock text response"
        assert result["audio_path"] is None   # model returned no audio
        assert result["latency_s"]  == 0.1
        assert "model" in result

    def test_audio_saved_to_disk(self, mock_backend_with_audio, input_wav, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path, return_audio=True)
        service = InferenceService(
            model    = mock_backend_with_audio,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        result = service.infer_from_file(input_path=input_wav)

        assert result["text"]       == "Mock audio response"
        assert result["audio_path"] is not None
        assert Path(result["audio_path"]).exists()

    def test_audio_saved_to_explicit_path(self, mock_backend_with_audio, input_wav, tmp_path):
        explicit = tmp_path / "answer.wav"
        cfg      = Settings(audio_output_dir=tmp_path, return_audio=True)
        service  = InferenceService(
            model    = mock_backend_with_audio,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        result = service.infer_from_file(input_path=input_wav, output_path=explicit)

        assert Path(result["audio_path"]) == explicit
        assert explicit.exists()

    def test_missing_input_raises(self, mock_backend_text_only, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path)
        service = InferenceService(
            model    = mock_backend_text_only,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        with pytest.raises(FileNotFoundError):
            service.infer_from_file(input_path="/nonexistent/audio.wav")

    def test_return_audio_false_skips_save(self, mock_backend_with_audio, input_wav, tmp_path):
        """When return_audio=False in settings, audio is not saved even if model provides it."""
        cfg     = Settings(audio_output_dir=tmp_path, return_audio=False)
        service = InferenceService(
            model    = mock_backend_with_audio,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        result = service.infer_from_file(input_path=input_wav)
        assert result["audio_path"] is None

    def test_system_prompt_passed_to_model(self, mock_backend_text_only, input_wav, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path)
        service = InferenceService(
            model    = mock_backend_text_only,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        service.infer_from_file(
            input_path    = input_wav,
            system_prompt = "Custom prompt",
        )
        call_kwargs = mock_backend_text_only.process.call_args.kwargs
        assert call_kwargs["system_prompt"] == "Custom prompt"


# ── Tests: infer_from_array ───────────────────────────────────────────────────

class TestInferFromArray:
    def test_includes_audio_array_in_response(self, mock_backend_with_audio, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path, return_audio=True)
        service = InferenceService(
            model    = mock_backend_with_audio,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        audio  = np.zeros(16000, dtype=np.float32)
        result = service.infer_from_array(audio=audio, sample_rate=16000)

        assert result["text"]        == "Mock audio response"
        assert result["audio_array"] is not None
        assert isinstance(result["audio_array"], np.ndarray)

    def test_audio_array_none_when_text_only(self, mock_backend_text_only, tmp_path):
        cfg     = Settings(audio_output_dir=tmp_path)
        service = InferenceService(
            model    = mock_backend_text_only,
            audio_io = LocalAudioIO(),
            settings = cfg,
        )
        audio  = np.zeros(16000, dtype=np.float32)
        result = service.infer_from_array(audio=audio, sample_rate=16000)

        assert result["audio_array"] is None
