"""
config.py — Application configuration via pydantic-settings.

All settings can be overridden by environment variables or a .env file.
Prefix: OMNI_

Examples:
    export OMNI_PROFILE=cuda_prod
    export OMNI_MODELS_DIR=/data/models
    export OMNI_MODEL_SUBDIR=Qwen2.5-Omni-7B
    export OMNI_SYSTEM_PROMPT="You are a BFSI compliance assistant."
    export OMNI_RETURN_AUDIO=false

The 'models_dir' defaults to v2/models/ — the same folder layout as v1.
Override with OMNI_MODELS_DIR to point at weights stored anywhere on disk.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# v2/ root — resolved from this file's location (v2/app/core/config.py)
ROOT = Path(__file__).resolve().parent.parent.parent


class DeviceProfile(str, Enum):
    MAC_MPS   = "mac_mps"    # Apple Silicon — float16, sdpa attention
    CUDA_PROD = "cuda_prod"  # CUDA GPU      — bfloat16, flash_attention_2
    CPU       = "cpu"        # fallback / CI / no GPU


# Default model subfolder per profile — matches v1 model names exactly
_PROFILE_DEFAULTS: dict[DeviceProfile, str] = {
    DeviceProfile.MAC_MPS:   "Qwen2.5-Omni-3B",
    DeviceProfile.CUDA_PROD: "Qwen3-Omni-30B-A3B-Instruct",
    DeviceProfile.CPU:       "Qwen2.5-Omni-3B",
}


class Settings(BaseSettings):
    """
    Single source of truth for all runtime settings.

    Precedence (highest → lowest):
        1. Environment variables  (OMNI_*)
        2. .env file in working directory
        3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="OMNI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # don't fail on unknown env vars
    )

    # ── Device profile ────────────────────────────────────────────────────────
    profile: DeviceProfile = DeviceProfile.MAC_MPS

    # ── Paths ─────────────────────────────────────────────────────────────────
    # Override OMNI_MODELS_DIR to point at weights wherever they live on disk.
    models_dir: Path = ROOT / "models"
    audio_input_dir: Path  = ROOT / "audio" / "input"
    audio_output_dir: Path = ROOT / "audio" / "output"

    # ── Model selection ───────────────────────────────────────────────────────
    # Set OMNI_MODEL_SUBDIR to override the default per-profile folder name.
    # e.g. OMNI_MODEL_SUBDIR=Qwen2.5-Omni-7B to run a mid-size model on CUDA.
    model_subdir: Optional[str] = None

    # ── Generation ───────────────────────────────────────────────────────────
    thinker_max_new_tokens: int   = 512
    thinker_do_sample: bool       = False
    speaker: str                  = "Ethan"    # Available: Ethan, Chelsie
    output_sample_rate: int       = 24000      # Qwen2.5-Omni native output rate
    return_audio: bool            = True

    # ── System prompt ─────────────────────────────────────────────────────────
    # IMPORTANT: This exact wording is required for Qwen2.5-Omni audio output.
    # Changing it may silently disable audio generation.
    system_prompt: str = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    # ── API server ────────────────────────────────────────────────────────────
    api_host: str  = "0.0.0.0"
    api_port: int  = 8000
    api_reload: bool = False   # set OMNI_API_RELOAD=true in dev

    # ── Streaming (reserved — not yet implemented in the model adapter) ───────
    streaming_chunk_tokens: int = 50

    # ── Multi-turn conversation ───────────────────────────────────────────────
    # Set OMNI_MODEL_SUPPORTS_MULTI_TURN=false if the loaded model does not
    # support multi-turn (e.g. very small context window or single-turn only).
    model_supports_multi_turn: bool = True

    # Context window of the model in tokens.
    # Qwen2.5-Omni-3B = 32 768. Update when switching to a different model.
    context_window_tokens: int = 32768

    # Approximate audio feature tokens generated per second of input audio.
    # Qwen2.5-Omni encodes audio at ~25 tokens / second.
    audio_tokens_per_second: float = 25.0

    # Token budget reserved for the system prompt (conservative estimate).
    system_prompt_reserve_tokens: int = 100

    # Root directory for per-session audio files (multi-turn mode).
    sessions_dir: Path = ROOT / "audio" / "sessions"

    # Sessions older than this many hours are deleted at server startup.
    session_ttl_hours: int = 24

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def model_local_path(self) -> Path:
        """
        Full path to the model weights directory.

        Uses OMNI_MODEL_SUBDIR if set, otherwise selects the default
        subfolder for the active profile.
        """
        subdir = self.model_subdir or _PROFILE_DEFAULTS[self.profile]
        return self.models_dir / subdir

    @property
    def torch_dtype_str(self) -> str:
        """String name of the torch dtype for this profile."""
        return "bfloat16" if self.profile == DeviceProfile.CUDA_PROD else "float16"

    @property
    def attn_implementation(self) -> str:
        """Attention backend: flash_attention_2 on CUDA, sdpa on MPS/CPU."""
        return "flash_attention_2" if self.profile == DeviceProfile.CUDA_PROD else "sdpa"

    @property
    def device_map(self) -> Optional[str]:
        """
        device_map="auto" works on CUDA (multi-GPU sharding).
        MPS doesn't support it — returns None for sequential loading.
        """
        return "auto" if self.profile == DeviceProfile.CUDA_PROD else None


# ── Module-level singleton ─────────────────────────────────────────────────────

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Return the singleton Settings instance.
    Safe to call multiple times — constructed once, then cached.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """
    Force recreation of the singleton on next call to get_settings().
    Useful in tests that monkeypatch environment variables.
    """
    global _settings
    _settings = None
