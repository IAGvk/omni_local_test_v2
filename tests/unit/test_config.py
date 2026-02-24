"""
test_config.py — Unit tests for app/core/config.py

No model needed. Tests cover:
  - Default profile selection
  - model_local_path derivation per profile
  - OMNI_PROFILE env-var override
  - OMNI_MODEL_SUBDIR override
  - dtype / attn / device_map properties
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import DeviceProfile, Settings


class TestDefaultProfile:
    def test_default_is_mac_mps(self):
        s = Settings()
        assert s.profile == DeviceProfile.MAC_MPS

    def test_default_model_subdir_mps(self, tmp_path):
        s = Settings(models_dir=tmp_path)
        assert s.model_local_path == tmp_path / "Qwen2.5-Omni-3B"

    def test_dtype_is_float16_on_mps(self):
        s = Settings(profile=DeviceProfile.MAC_MPS)
        assert s.torch_dtype_str == "float16"

    def test_attn_is_sdpa_on_mps(self):
        s = Settings(profile=DeviceProfile.MAC_MPS)
        assert s.attn_implementation == "sdpa"

    def test_device_map_is_none_on_mps(self):
        s = Settings(profile=DeviceProfile.MAC_MPS)
        assert s.device_map is None


class TestCudaProfile:
    def test_cuda_model_subdir(self, tmp_path):
        s = Settings(profile=DeviceProfile.CUDA_PROD, models_dir=tmp_path)
        assert "Qwen3-Omni" in s.model_local_path.name

    def test_dtype_is_bfloat16_on_cuda(self):
        s = Settings(profile=DeviceProfile.CUDA_PROD)
        assert s.torch_dtype_str == "bfloat16"

    def test_attn_is_flash_attention_on_cuda(self):
        s = Settings(profile=DeviceProfile.CUDA_PROD)
        assert s.attn_implementation == "flash_attention_2"

    def test_device_map_is_auto_on_cuda(self):
        s = Settings(profile=DeviceProfile.CUDA_PROD)
        assert s.device_map == "auto"


class TestEnvVarOverrides:
    def test_profile_from_env(self, monkeypatch):
        monkeypatch.setenv("OMNI_PROFILE", "cuda_prod")
        s = Settings()
        assert s.profile == DeviceProfile.CUDA_PROD

    def test_model_subdir_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMNI_MODEL_SUBDIR", "MyCustomModel")
        s = Settings(models_dir=tmp_path)
        assert s.model_local_path == tmp_path / "MyCustomModel"

    def test_models_dir_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OMNI_MODELS_DIR", str(tmp_path))
        s = Settings()
        assert s.models_dir == tmp_path

    def test_return_audio_false_from_env(self, monkeypatch):
        monkeypatch.setenv("OMNI_RETURN_AUDIO", "false")
        s = Settings()
        assert s.return_audio is False
