"""
qwen_omni.py — ModelBackend adapter for Qwen2.5-Omni.

Implements the ModelBackend port for both:
  - Apple Silicon MPS  (float16, sdpa attention)
  - NVIDIA CUDA        (bfloat16, flash_attention_2)

Responsibilities of THIS adapter:
  • Load processor + model weights from a local directory
  • Build the message structure Qwen2.5-Omni expects
  • Run model.generate() and extract text + audio outputs
  • Apply MPS-specific dtype fixes
  • Clean up device cache

What this adapter does NOT do:
  • Read/write audio files          → AudioIO adapter's job
  • Know about HTTP or WebSockets   → API layer's job
  • Decide output paths             → InferenceService's job

The adapter receives pre-loaded numpy audio arrays from the service layer.
It writes a temp WAV internally before passing to qwen_omni_utils, because
the process_mm_info utility reliably handles file paths across all versions.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch

from app.core.config import Settings, get_settings
from app.domain.ports import InferenceResult, ModelBackend
from app.utils.device import fix_mps_dtypes, free_device_cache

logger = logging.getLogger(__name__)


class QwenOmniBackend(ModelBackend):
    """
    Concrete adapter: Qwen2.5-Omni (local weights, MPS or CUDA).

    Lazy-initialises on first call to process() or via explicit initialize().
    Use as a context manager for automatic cleanup:

        with QwenOmniBackend() as backend:
            result = backend.process(audio, sr)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._cfg   = settings or get_settings()
        self.model: Any      = None
        self.processor: Any  = None
        self._device: Optional[torch.device] = None
        self._initialized    = False

    # ── ModelBackend port ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Load processor and model weights. Idempotent."""
        if self._initialized:
            return

        model_path = str(self._cfg.model_local_path)
        t0 = time.perf_counter()

        logger.info("Loading processor  : %s", model_path)

        from transformers import (  # type: ignore
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            local_files_only=True,   # never reach the network at runtime
        )

        # Map dtype string → torch dtype
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map[self._cfg.torch_dtype_str]

        logger.info(
            "Loading model weights: %s  (dtype=%s, attn=%s)",
            model_path,
            torch_dtype,
            self._cfg.attn_implementation,
        )

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(  # type: ignore
            model_path,
            torch_dtype=torch_dtype,
            device_map=self._cfg.device_map,
            attn_implementation=self._cfg.attn_implementation,
            local_files_only=True,
        )
        self.model.eval()

        # Resolve the actual device after device_map placement
        self._device = next(self.model.parameters()).device
        logger.info(
            "Model ready  device=%s  load_time=%.1fs",
            self._device,
            time.perf_counter() - t0,
        )
        self._initialized = True

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        system_prompt: Optional[str] = None,
    ) -> InferenceResult:
        """
        Run a full audio-in → text+audio inference pass.

        The numpy audio array is written to a temporary WAV file so that
        qwen_omni_utils.process_mm_info can load it consistently.
        The temp file is deleted before this method returns.
        """
        if not self._initialized:
            self.initialize()

        prompt_text = system_prompt or self._cfg.system_prompt

        # ── Write audio to temp file ──────────────────────────────────────────
        # process_mm_info reliably handles file paths across all versions of
        # qwen_omni_utils. Using a file path avoids potential numpy array
        # handling differences between library versions.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            sf.write(str(tmp_path), audio.reshape(-1), samplerate=sample_rate)
            result = self._run_inference(str(tmp_path), prompt_text)
        finally:
            tmp_path.unlink(missing_ok=True)

        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_inference(self, audio_path: str, prompt_text: str) -> InferenceResult:
        """Core inference logic — separated for readability."""
        from qwen_omni_utils import process_mm_info  # type: ignore

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt_text}],
            },
            {
                "role": "user",
                "content": [{"type": "audio", "audio": audio_path}],
            },
        ]

        # ── Tokenise + prepare multimodal inputs ──────────────────────────────
        text_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        audios, images, videos, _ = process_mm_info(messages, use_audio_in_video=False)

        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        # Move tensors to model device
        inputs = {
            k: v.to(self._device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # MPS requires float16; bfloat16 is not supported on MPS
        if self._device is not None and self._device.type == "mps":
            inputs = fix_mps_dtypes(inputs)

        # ── Generate ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("Generating response …")

        with torch.no_grad():
            raw = self.model.generate(
                **inputs,
                speaker=self._cfg.speaker,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=self._cfg.thinker_max_new_tokens,
                thinker_do_sample=self._cfg.thinker_do_sample,
                return_audio=self._cfg.return_audio,
                use_audio_in_video=False,
            )

        latency = time.perf_counter() - t0
        logger.info("Generation time: %.2fs", latency)

        # ── Extract sequences and audio ───────────────────────────────────────
        sequences   = raw.sequences if hasattr(raw, "sequences") else raw[0]
        audio_raw   = None
        if hasattr(raw, "audio"):
            audio_raw = raw.audio
        elif isinstance(raw, (tuple, list)) and len(raw) > 1:
            audio_raw = raw[1]

        # ── Decode text ───────────────────────────────────────────────────────
        input_len     = inputs["input_ids"].shape[1]
        response_text = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        logger.info("Response (first 120 chars): %s", response_text[:120])

        # ── Convert audio tensor → numpy ──────────────────────────────────────
        audio_np: Optional[np.ndarray] = None
        if audio_raw is not None:
            if isinstance(audio_raw, torch.Tensor):
                audio_np = audio_raw.reshape(-1).detach().cpu().float().numpy()
            else:
                audio_np = np.array(audio_raw, dtype=np.float32).reshape(-1)

        # ── Release device cache ──────────────────────────────────────────────
        if self._device is not None:
            free_device_cache(self._device)

        return InferenceResult(
            text       = response_text,
            audio      = audio_np,
            sample_rate= self._cfg.output_sample_rate,
            latency_s  = round(latency, 3),
        )

    def cleanup(self) -> None:
        """Release model weights from device memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self._device is not None:
            free_device_cache(self._device)
        self._initialized = False
        logger.info("QwenOmniBackend cleaned up.")

    @property
    def is_ready(self) -> bool:
        return self._initialized

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "QwenOmniBackend":
        self.initialize()
        return self

    def __exit__(self, *_) -> None:
        self.cleanup()
