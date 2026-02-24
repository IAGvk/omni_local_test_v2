"""
device.py — PyTorch device utilities shared across adapters.

Centralises MPS dtype fixes, device cache management, and device detection.
These helpers are infrastructure-level — they belong in utils/, not in
the domain layer (they know about torch internals).
"""

from __future__ import annotations

import torch


def fix_mps_dtypes(inputs: dict) -> dict:
    """
    Apple MPS does not support bfloat16.
    Cast all bfloat16 tensors in the inputs dict to float16.
    Safe no-op on CUDA (where bfloat16 is perfectly fine).
    """
    return {
        k: v.to(torch.float16)
        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16
        else v
        for k, v in inputs.items()
    }


def free_device_cache(device: torch.device) -> None:
    """
    Release the device memory cache to prevent OOM on repeated inference calls.
    No-op on CPU.
    """
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def get_best_device() -> torch.device:
    """
    Return the best available device: MPS > CUDA > CPU.
    Used for env-check diagnostics and device auto-detection.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_environment_summary() -> None:
    """Print a concise environment summary to stdout (for --env-check)."""
    import platform

    device = get_best_device()

    try:
        import transformers
        tf_ver = transformers.__version__
    except ImportError:
        tf_ver = "not installed"

    try:
        import soundfile
        sf_ver = soundfile.__version__
    except ImportError:
        sf_ver = "not installed"

    try:
        import librosa
        lb_ver = librosa.__version__
    except ImportError:
        lb_ver = "not installed"

    print("=" * 58)
    print("  Qwen2.5-Omni v2 Pipeline — Environment Summary")
    print("=" * 58)
    print(f"  Platform     : {platform.platform()}")
    print(f"  Python       : {platform.python_version()}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  Transformers : {tf_ver}")
    print(f"  soundfile    : {sf_ver}")
    print(f"  librosa      : {lb_ver}")
    print(f"  Active device: {device}")
    if device.type == "mps":
        print(f"  MPS built    : {torch.backends.mps.is_built()}")
    elif device.type == "cuda":
        print(f"  CUDA devices : {torch.cuda.device_count()}")
        print(f"  CUDA version : {torch.version.cuda}")
    print("=" * 58)
