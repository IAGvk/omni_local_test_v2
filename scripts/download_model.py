#!/usr/bin/env python3
"""
download_model.py — Download Qwen2.5-Omni weights to v2/models/

Tries three strategies in order:
  1. ModelScope SDK (modelscope.cn)  — Alibaba's own CDN, no HF needed
  2. HuggingFace Hub                 — fallback if ModelScope is blocked
  3. git-lfs clone                   — manual last resort

Target directory is read from Settings, so OMNI_MODELS_DIR and
OMNI_MODEL_SUBDIR environment variables are respected:
    export OMNI_MODELS_DIR=/data/models
    python scripts/download_model.py

Run once before first use:
    cd /Users/s748779/omni_local_test/v2
    source .venv/bin/activate
    python scripts/download_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings

cfg      = get_settings()
MODEL_ID = "Qwen/Qwen2.5-Omni-3B"    # HuggingFace repo ID
LOCAL_DIR = cfg.model_local_path


def main():
    print("=" * 62)
    print("  Downloading Qwen2.5-Omni weights")
    print("=" * 62)
    print(f"  Destination : {LOCAL_DIR}")
    print(f"  Profile     : {cfg.profile.value}")
    print(f"  Size        : ~6 GB  (expect 5–10 min on fast broadband)")
    print()

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("  [1/3] Trying ModelScope …")
    if _try_modelscope():
        return

    print("  [2/3] Trying HuggingFace Hub …")
    if _try_huggingface():
        return

    print("  [3/3] Trying git-lfs clone …")
    if _try_git_lfs():
        return

    _print_manual_instructions()
    sys.exit(1)


def _try_modelscope() -> bool:
    try:
        from modelscope import snapshot_download  # type: ignore
        snapshot_download(
            model_id      = MODEL_ID,
            cache_dir     = str(cfg.models_dir),
            local_dir     = str(LOCAL_DIR),
            ignore_patterns = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        if _verify():
            print(f"\n✅  Downloaded via ModelScope → {LOCAL_DIR}")
            return True
    except Exception as e:
        print(f"     ModelScope failed: {e}")
    return False


def _try_huggingface() -> bool:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
        snapshot_download(
            repo_id         = MODEL_ID,
            local_dir       = str(LOCAL_DIR),
            ignore_patterns = ["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        if _verify():
            print(f"\n✅  Downloaded via HuggingFace Hub → {LOCAL_DIR}")
            return True
    except Exception as e:
        print(f"     HuggingFace failed: {e}")
    return False


def _try_git_lfs() -> bool:
    import shutil
    import subprocess

    if not shutil.which("git"):
        print("     git not found, skipping git-lfs strategy")
        return False

    urls = [
        "https://www.modelscope.cn/qwen/Qwen2.5-Omni-3B.git",
        "https://huggingface.co/Qwen/Qwen2.5-Omni-3B.git",
    ]
    for url in urls:
        try:
            env = {**__import__("os").environ, "GIT_LFS_SKIP_SMUDGE": "0"}
            result = subprocess.run(
                ["git", "clone", url, str(LOCAL_DIR)],
                capture_output=True, text=True, timeout=3600, env=env,
            )
            if result.returncode == 0 and _verify():
                print(f"\n✅  Downloaded via git-lfs → {LOCAL_DIR}")
                return True
            else:
                print(f"     git clone from {url} failed")
        except Exception as e:
            print(f"     git clone error: {e}")
    return False


def _verify() -> bool:
    """Check that minimum required files exist after download."""
    required = ["config.json", "tokenizer_config.json", "special_tokens_map.json"]
    missing  = [f for f in required if not (LOCAL_DIR / f).exists()]
    if missing:
        print(f"  ⚠️  Incomplete download — missing: {missing}")
        return False
    shards = sorted(LOCAL_DIR.glob("*.safetensors"))
    print(f"\n  ✅  Verified download")
    print(f"     Directory  : {LOCAL_DIR}")
    print(f"     Safetensors: {len(shards)} shards")
    for s in shards:
        print(f"       {s.name:<45}  {s.stat().st_size / 1e9:.2f} GB")
    return True


def _print_manual_instructions():
    print("\n" + "=" * 62)
    print("  ⚠️  Automated download failed — manual options below")
    print("=" * 62)
    print(f"""
OPTION A — Download on a personal / unrestricted network
──────────────────────────────────────────────────────────
  Switch to home wifi or mobile hotspot, then run:
    python scripts/download_model.py

  Or use huggingface-cli:
    pip install huggingface_hub
    huggingface-cli download Qwen/Qwen2.5-Omni-3B \\
        --local-dir {LOCAL_DIR}

OPTION B — Manual git-lfs clone
──────────────────────────────────────────────────────────
  git lfs install
  git clone https://huggingface.co/Qwen/Qwen2.5-Omni-3B \\
      {LOCAL_DIR}

OPTION C — Custom path (weights already downloaded elsewhere)
──────────────────────────────────────────────────────────
  export OMNI_MODELS_DIR=/path/to/existing/models
  export OMNI_MODEL_SUBDIR=Qwen2.5-Omni-3B   # if subfolder differs

  Then run:
    python -m cli.main --env-check  # confirms path is found

TARGET DIRECTORY:
  {LOCAL_DIR}

MINIMUM FILES NEEDED:
  config.json, tokenizer_config.json, special_tokens_map.json
  + all .safetensors weight files (~4-5 shards, ~6 GB total)
""")


if __name__ == "__main__":
    main()
