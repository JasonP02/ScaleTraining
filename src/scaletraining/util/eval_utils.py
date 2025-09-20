from __future__ import annotations
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scaletraining.model.model import TransformerNetwork
from scaletraining.util import find_latest_model_path


import contextlib
import math
from typing import Any, Tuple

import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from scaletraining.util.training_utils import compute_loss_sum, prepare_targets


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

def load_pretrained_model_and_tokenizer(flat):
    model_path = getattr(flat, "model_path", None)
    if not model_path or str(model_path).lower() == "latest":
        # Auto-discover latest model under outputs
        output_root = getattr(flat, "output_dir", "outputs")
        auto_path = find_latest_model_path(output_root)
        if not auto_path:
            raise RuntimeError("No model_path provided and no latest model found under outputs/. Pass model_path=... or create outputs/<run>/model.pt.")
        print(f"[generate] Using latest model: {auto_path}")
        model_path = auto_path

    # Build model from config and load weights
    model = TransformerNetwork(flat).to(flat.device)
    ckpt = torch.load(model_path, map_location=flat.device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Normalize keys from compiled/DataParallel checkpoints if present
    def _strip_prefix(sd, prefix: str):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd
    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict)
    model.eval()

    # Load tokenizer, supporting local JSON (dataset-specific) via PreTrainedTokenizerFast
    tok_path = flat.tokenizer_name
    from pathlib import Path as _P
    if isinstance(tok_path, str) and _P(tok_path).exists() and tok_path.endswith('.json'):
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token