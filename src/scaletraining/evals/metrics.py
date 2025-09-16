"""Evaluation utilities (loss, perplexity, etc.)."""
from __future__ import annotations

import contextlib
import math
from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from scaletraining.training.training_utils import compute_loss_sum, prepare_targets


@torch.inference_mode()
def evaluate_perplexity(
    model: nn.Module,
    data_loader: DataLoader,
    cfg: Any,
    loss_fn: nn.Module,
    *,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Evaluate average per-token loss and perplexity on a data loader."""

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(cfg.device)
        context = (
            autocast(device_type="cuda", dtype=torch.bfloat16)
            if (cfg.device == "cuda" and torch.cuda.is_available())
            else contextlib.nullcontext()
        )
        with context:
            hidden = model.forward_hidden(input_ids)[:, :-1, :]
            targets, effective = prepare_targets(input_ids)
            loss_sum = compute_loss_sum(
                model, hidden, targets, getattr(cfg, "logits_chunk_size", 0), loss_fn
            )
        total_loss += float(loss_sum.item())
        total_tokens += int(effective)
        batches_seen += 1
        if max_batches and batches_seen >= max_batches:
            break
    avg = (total_loss / max(1, total_tokens)) if total_tokens > 0 else float("inf")
    ppl = math.exp(min(50.0, max(-50.0, avg))) if avg != float("inf") else float("inf")
    if was_training:
        model.train()
    return avg, ppl


__all__ = ["evaluate_perplexity"]
