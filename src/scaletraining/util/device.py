"""Device and backend configuration helpers."""
from __future__ import annotations

import gc
import os
from typing import Any

import torch


def clear_cuda_cache() -> None:
    """Release cached CUDA memory if a GPU is available."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def configure_rocm_and_sdp(cfg: Any) -> None:
    """Apply ROCm allocator tweaks and scaled-dot-product attention toggles."""

    # Improve memory utility and segmentation
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.enable_flash_sdp(bool(cfg.use_flash_sdp)) # Often true
        torch.backends.cuda.enable_mem_efficient_sdp(bool(cfg.use_mem_efficient_sdp)) # Often true
        torch.backends.cuda.enable_math_sdp(bool(cfg.use_math_sdp)) # Often false
    except Exception as e:
        print(f"RoCM settings not configured: {e}")



__all__ = ["clear_cuda_cache", "configure_rocm_and_sdp"]
