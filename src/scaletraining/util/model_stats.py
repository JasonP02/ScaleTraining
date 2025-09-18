"""Helpers for computing and formatting model size statistics."""
from __future__ import annotations

from typing import Tuple

import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts for ``model``."""

    total = 0
    trainable = 0
    for param in model.parameters():
        numel = int(param.numel())
        total += numel
        if param.requires_grad:
            trainable += numel
    return total, trainable


def humanize_params(count: int) -> str:
    """Format a parameter count with engineering suffixes (K/M/B/T/Q)."""

    value = float(max(count, 0))
    for suffix in ("", "K", "M", "B", "T", "Q"):
        if value < 1000 or suffix == "Q":
            if suffix:
                return f"{value:.2f}{suffix}"
            return f"{int(value):,}"
        value /= 1000.0
    return f"{value:.2f}Q"


def humanize_bytes(num_bytes: float) -> str:
    """Render a byte count using binary units."""

    value = float(max(num_bytes, 0.0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PiB"


__all__ = [
    "count_parameters",
    "humanize_params",
    "humanize_bytes",
]
