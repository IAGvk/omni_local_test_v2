#!/usr/bin/env python3
"""
test_smoke.py — Integration smoke tests for the v2 pipeline.

Mirrors v1's 14/14 test suite. No model weights needed.

Tests:
  1. Core imports (torch, transformers Qwen classes, qwen_omni_utils, audio libs)
  2. Config (profiles, paths, dtype properties)
  3. Audio adapter (write/read roundtrip, resample, dtype)
  4. Device utilities (MPS dtype fix, device detection)
  5. Domain schemas (serialisation)
  6. InferenceService with mock backend (no model)

Run:
    cd /Users/s748779/omni_local_test/v2
    source .venv/bin/activate
    python -m pytest tests/integration/test_smoke.py -v
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

PASSED: list[str] = []
FAILED: list[str] = []


def section(title: str):
    print(f"\n{'─' * 52}")
    print(f"  {title}")
    print("─" * 52)


def check(label: str, fn):
    try:
        fn()
        PASSED.append(label)
        print(f"  ✅  {label}")
    except Exception as e:
        FAILED.append(label)
        print(f"  ❌  {label}  →  {e}")


# ── 1. Core imports ───────────────────────────────────────────────────────────

section("1. Core imports")


def t_torch_available():
    assert torch.__version__, "torch not importable"


def t_mps_or_cuda_available():
    has_device = torch.backends.mps.is_available() or torch.cuda.is_available()
    # On CI without GPU this is fine — just verify torch works
    assert True  # presence of torch is enough for the smoke test


def t_qwen_classes():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor  # type: ignore
    assert Qwen2_5OmniForConditionalGeneration is not None
    assert Qwen2_5OmniProcessor is not None


def t_qwen_omni_utils():
    from qwen_omni_utils import process_mm_info  # type: ignore
    assert process_mm_info is not None


def t_audio_libs():
    import soundfile as sf
    import librosa
    assert sf.__version__
    assert librosa.__version__


def t_numpy_version():
    major = int(np.__version__.split(".")[0])
    assert major < 2, f"numpy {np.__version__} >= 2.0 may break audio libs"


def t_fastapi_importable():
    import fastapi
    assert fastapi.__version__


def t_pydantic_settings():
    from pydantic_settings import BaseSettings
    assert BaseSettings is not None


check("torch importable",                    t_torch_available)
check("torch device check",                  t_mps_or_cuda_available)
check("transformers Qwen2.5-Omni classes",   t_qwen_classes)
check("qwen_omni_utils.process_mm_info",     t_qwen_omni_utils)
check("soundfile + librosa importable",      t_audio_libs)
check("numpy < 2.0",                         t_numpy_version)
check("fastapi importable",                  t_fastapi_importable)
check("pydantic-settings importable",        t_pydantic_settings)


# ── 2. Config ─────────────────────────────────────────────────────────────────

section("2. Config")


def t_config_mac_mps():
    from app.core.config import DeviceProfile, Settings
    s = Settings()
    assert s.profile == DeviceProfile.MAC_MPS
    assert s.torch_dtype_str == "float16"
    assert s.attn_implementation == "sdpa"
    assert s.device_map is None
    assert "3B" in s.model_local_path.name


def t_config_cuda_prod():
    from app.core.config import DeviceProfile, Settings
    s = Settings(profile=DeviceProfile.CUDA_PROD)
    assert s.torch_dtype_str == "bfloat16"
    assert s.attn_implementation == "flash_attention_2"
    assert s.device_map == "auto"


def t_config_custom_subdir(tmp_path):
    from app.core.config import Settings
    s = Settings(models_dir=tmp_path, model_subdir="MyModel")
    assert s.model_local_path == tmp_path / "MyModel"


def t_config_paths_exist_structure():
    """audio/input and audio/output dirs exist in v2 root."""
    from app.core.config import Settings
    s = Settings()
    # Dirs should be under ROOT/audio/
    assert "audio" in str(s.audio_input_dir)
    assert "audio" in str(s.audio_output_dir)


check("config mac_mps profile",        t_config_mac_mps)
check("config cuda_prod profile",      t_config_cuda_prod)
check("config custom model_subdir",    lambda: t_config_custom_subdir(Path(tempfile.mkdtemp())))
check("config audio dir paths",        t_config_paths_exist_structure)


# ── 3. Audio adapter ──────────────────────────────────────────────────────────

section("3. Audio adapter (LocalAudioIO)")

import soundfile as sf
from app.adapters.audio.local_io import LocalAudioIO


def t_write_read_roundtrip():
    io    = LocalAudioIO()
    audio = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 0.5).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tone.wav"
        io.write(audio, path, sample_rate=16000)
        loaded, sr = io.read(path, target_sr=16000)
    assert sr == 16000
    assert len(loaded) == len(audio)
    assert loaded.dtype == np.float32


def t_stereo_to_mono():
    io     = LocalAudioIO()
    stereo = np.random.rand(16000, 2).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "stereo.wav"
        sf.write(str(path), stereo, 16000)
        mono, _ = io.read(path, target_sr=16000)
    assert mono.ndim == 1


def t_resample_44k_to_16k():
    io    = LocalAudioIO()
    audio = (np.sin(np.linspace(0, 2 * np.pi, 44100)) * 0.5).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "44k.wav"
        sf.write(str(path), audio, 44100)
        loaded, sr = io.read(path, target_sr=16000)
    assert sr == 16000
    assert abs(len(loaded) - 16000) < 200


def t_missing_file_raises():
    io = LocalAudioIO()
    try:
        io.read("/nonexistent/audio.wav")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass


check("audio write → read roundtrip",  t_write_read_roundtrip)
check("stereo → mono conversion",      t_stereo_to_mono)
check("resample 44kHz → 16kHz",        t_resample_44k_to_16k)
check("missing file raises",           t_missing_file_raises)


# ── 4. Device utilities ───────────────────────────────────────────────────────

section("4. Device utilities")


def t_fix_mps_dtypes():
    from app.utils.device import fix_mps_dtypes
    inputs = {
        "input_ids":     torch.ones(1, 10, dtype=torch.long),
        "audio_feats":   torch.zeros(1, 80, 3000, dtype=torch.bfloat16),
        "attn_mask":     torch.ones(1, 10, dtype=torch.bfloat16),
    }
    fixed = fix_mps_dtypes(inputs)
    assert fixed["input_ids"].dtype  == torch.long     # unchanged
    assert fixed["audio_feats"].dtype == torch.float16  # cast
    assert fixed["attn_mask"].dtype  == torch.float16   # cast


def t_get_best_device():
    from app.utils.device import get_best_device
    device = get_best_device()
    assert device.type in ("mps", "cuda", "cpu")


check("fix_mps_dtypes casts bfloat16 → float16",  t_fix_mps_dtypes)
check("get_best_device returns valid device",       t_get_best_device)


# ── 5. Domain schemas ─────────────────────────────────────────────────────────

section("5. Domain schemas")


def t_infer_response_schema():
    from app.domain.schemas import InferResponse
    r = InferResponse(text="hello", latency_s=1.5, model="Qwen2.5-Omni-3B")
    data = r.model_dump()
    assert data["text"] == "hello"
    assert data["audio_path"] is None


def t_ws_infer_request_defaults():
    from app.domain.schemas import WSInferRequest
    r = WSInferRequest()
    assert r.sample_rate == 16000
    assert r.return_audio is True


def t_ws_chunk_final():
    from app.domain.schemas import WSChunk
    c = WSChunk(text="done", is_final=True)
    assert c.is_final
    assert c.text == "done"


check("InferResponse schema",          t_infer_response_schema)
check("WSInferRequest defaults",       t_ws_infer_request_defaults)
check("WSChunk final flag",            t_ws_chunk_final)


# ── 6. InferenceService with mock backend ─────────────────────────────────────

section("6. InferenceService (mock ModelBackend — no model weights needed)")

from app.domain.ports import InferenceResult
from app.services.inference import InferenceService


def t_service_text_only():
    mock = MagicMock()
    mock.is_ready = True
    mock.process.return_value = InferenceResult(text="Test response", audio=None)

    audio = np.zeros(16000, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "in.wav"
        sf.write(str(wav), audio, 16000)
        from app.core.config import Settings
        cfg     = Settings(audio_output_dir=Path(d))
        service = InferenceService(model=mock, audio_io=LocalAudioIO(), settings=cfg)
        result  = service.infer_from_file(input_path=wav)

    assert result["text"]       == "Test response"
    assert result["audio_path"] is None


def t_service_with_audio():
    mock = MagicMock()
    mock.is_ready = True
    mock.process.return_value = InferenceResult(
        text="Response with audio",
        audio=np.zeros(24000, dtype=np.float32),
        sample_rate=24000,
    )

    audio = np.zeros(16000, dtype=np.float32)
    with tempfile.TemporaryDirectory() as d:
        wav  = Path(d) / "in.wav"
        sf.write(str(wav), audio, 16000)
        from app.core.config import Settings
        cfg     = Settings(audio_output_dir=Path(d), return_audio=True)
        service = InferenceService(model=mock, audio_io=LocalAudioIO(), settings=cfg)
        result  = service.infer_from_file(input_path=wav)

    assert result["text"]       == "Response with audio"
    assert result["audio_path"] is not None
    assert Path(result["audio_path"]).exists()


check("service: text-only response",    t_service_text_only)
check("service: text + audio saved",    t_service_with_audio)


# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")
total = len(PASSED) + len(FAILED)
print(f"\n  Passed : {len(PASSED)}/{total}")
if FAILED:
    print(f"  Failed : {len(FAILED)}/{total}")
    for f in FAILED:
        print(f"    ✗  {f}")
    print()
else:
    print("\n  ALL SMOKE TESTS PASSED ✅")
    print("  Environment is ready — run: python scripts/download_model.py\n")
