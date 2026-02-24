#!/usr/bin/env python3
"""
cli/main.py — Command-line interface for the Omni audio pipeline.

Identical surface to v1/main.py — same flags, same output format.
Internally wired through v2's InferenceService instead of calling
the model directly, so the CLI benefits from all v2 improvements
(swappable backends, clean error paths, no duplication).

Usage:
    python -m cli.main audio/input/my_question.wav
    python -m cli.main audio/input/my_question.wav --output audio/output/answer.wav
    python -m cli.main audio/input/my_question.wav --no-audio
    python -m cli.main audio/input/my_question.wav --prompt "You are a compliance assistant."
    python -m cli.main --test-tone
    python -m cli.main --env-check
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

# Ensure v2/ root is importable when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.adapters.audio.local_io import LocalAudioIO
from app.adapters.model.qwen_omni import QwenOmniBackend
from app.core.config import get_settings, reset_settings
from app.core.logging import setup_logging
from app.services.inference import InferenceService

setup_logging()
logger = logging.getLogger("cli")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen2.5-Omni  ·  Audio-In → Audio-Out  (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to input audio file (WAV, FLAC, MP3)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output WAV path (default: audio/output/response_<ts>.wav)",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Override system prompt",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Return text only; skip audio generation (faster)",
    )
    parser.add_argument(
        "--test-tone",
        action="store_true",
        help="Generate a 440 Hz test tone and run it through the pipeline",
    )
    parser.add_argument(
        "--env-check",
        action="store_true",
        help="Print environment summary and exit (no model loaded)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=["mac_mps", "cuda_prod", "cpu"],
        help="Device profile override (default: reads OMNI_PROFILE env var)",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _generate_test_tone(output_path: Path, duration: float = 2.0, freq: float = 440.0, sr: int = 16000) -> Path:
    """Write a 440 Hz sine wave to output_path."""
    import soundfile as sf
    t    = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone = (np.sin(2 * math.pi * freq * t) * 0.5).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), tone, sr)
    logger.info("Test tone written → %s", output_path)
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # ── Override profile via CLI flag ─────────────────────────────────────────
    if args.profile:
        import os
        os.environ["OMNI_PROFILE"] = args.profile
        reset_settings()   # force re-read of env vars

    cfg = get_settings()

    # ── Env check ─────────────────────────────────────────────────────────────
    if args.env_check:
        from app.utils.device import print_environment_summary
        print_environment_summary()
        print(f"\n  Profile    : {cfg.profile.value}")
        print(f"  Model path : {cfg.model_local_path}")
        print(f"  Weights    : {'✅ found' if cfg.model_local_path.exists() else '❌ not found'}")
        return 0

    # ── Resolve input audio ───────────────────────────────────────────────────
    if args.test_tone:
        tone_path  = cfg.audio_input_dir / "test_tone.wav"
        input_path = _generate_test_tone(tone_path)
        logger.info("Using generated test tone: %s", input_path)
    elif args.input:
        input_path = Path(args.input)
    else:
        print("❌  Provide an input audio file or use --test-tone")
        print("    Run with --help for usage.")
        return 1

    if not input_path.exists():
        print(f"❌  Input file not found: {input_path}")
        return 1

    # ── Validate model weights ────────────────────────────────────────────────
    if not cfg.model_local_path.exists():
        print(f"\n❌  Model weights not found at: {cfg.model_local_path}")
        print("    Download them first:")
        print("      python scripts/download_model.py")
        print("    Or set OMNI_MODELS_DIR / OMNI_MODEL_SUBDIR environment variables.")
        return 1

    if args.no_audio:
        # Mutate the singleton — only affects this process lifetime
        cfg.return_audio = False

    # ── Wire adapters → service ───────────────────────────────────────────────
    backend = QwenOmniBackend(settings=cfg)
    service = InferenceService(
        model    = backend,
        audio_io = LocalAudioIO(),
        settings = cfg,
    )

    print(f"\n🎤  Input  : {input_path}")
    print(f"🤖  Model  : {cfg.model_local_path.name}")
    print()

    try:
        result = service.infer_from_file(
            input_path    = input_path,
            system_prompt = args.prompt,
            output_path   = args.output,
        )

        print("─" * 55)
        print(f"📝  Text    : {result['text']}")
        print(f"⏱   Latency : {result['latency_s']}s")
        if result["audio_path"]:
            print(f"🔊  Audio   : {result['audio_path']}")
        print("─" * 55 + "\n")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        return 1

    finally:
        backend.cleanup()


if __name__ == "__main__":
    sys.exit(main())
