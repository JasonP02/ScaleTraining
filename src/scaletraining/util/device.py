"""Device and backend configuration helpers."""
from __future__ import annotations

import gc
import os
from typing import Any

import torch
from omegaconf import DictConfig, open_dict


def clear_cuda_cache() -> None:
    """Release cached CUDA memory if a GPU is available."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def resolve_device(cfg: DictConfig) -> str:
    """Return the requested compute device, defaulting to CUDA when available."""

    device_cfg = cfg.device
    requested = getattr(device_cfg, "device", None)

    if isinstance(requested, str) and requested:
        device = requested
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            with open_dict(device_cfg):
                device_cfg.device = device
        except Exception:
            pass

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        try:
            with open_dict(device_cfg):
                device_cfg.device = device
        except Exception:
            pass

    try:
        with open_dict(cfg):
            cfg.device_resolved = device
    except Exception:
        pass

    return str(device)


def configure_rocm_and_sdp(cfg: Any) -> None:
    """Apply ROCm allocator tweaks and scaled-dot-product attention toggles."""

    # Improve memory utility and segmentation
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.backends.cuda.enable_flash_sdp(bool(cfg.use_flash_sdp))  # Often true
        torch.backends.cuda.enable_mem_efficient_sdp(bool(cfg.use_mem_efficient_sdp))  # Often true
        torch.backends.cuda.enable_math_sdp(bool(cfg.use_math_sdp))  # Often false
    except Exception as exc:  # pragma: no cover - ROCm toggles best-effort
        print(f"RoCM settings not configured: {exc}")


__all__ = ["clear_cuda_cache", "configure_rocm_and_sdp", "resolve_device"]
