from __future__ import annotations
from pathlib import Path

import torch
from omegaconf import DictConfig, open_dict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scaletraining.model.model import TransformerNetwork
from scaletraining.util import find_latest_model_path
from scaletraining.util.device import resolve_device


import contextlib
import math
from typing import Any, Tuple

import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from scaletraining.util.training_utils import compute_loss_sum, prepare_targets


_REPO_ROOT = Path(__file__).resolve().parents[3]


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
    device = resolve_device(cfg)
    total_loss = 0.0
    total_tokens = 0
    batches_seen = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        context = (
            autocast(device_type="cuda", dtype=torch.bfloat16)
            if (device == "cuda" and torch.cuda.is_available())
            else contextlib.nullcontext()
        )
        with context:
            hidden = model.forward_hidden(input_ids)[:, :-1, :]
            targets, effective = prepare_targets(input_ids)
            loss_sum = compute_loss_sum(
                model,
                hidden,
                targets,
                getattr(cfg.training, "logits_chunk_size", 0),
                loss_fn,
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

def _normalize_output_dir(cfg: DictConfig) -> Path:
    output_root_value = cfg.paths.output_dir
    output_root = Path(output_root_value).expanduser()
    if not output_root.is_absolute():
        output_root = (_REPO_ROOT / output_root).expanduser()
    try:
        with open_dict(cfg.paths):
            cfg.paths.output_dir = str(output_root)
    except Exception:
        pass
    return output_root


def _resolve_model_path(cfg: DictConfig, output_root: Path) -> Path:
    model_path_cfg = cfg.generation.model_path
    if not model_path_cfg or str(model_path_cfg).lower() == "latest":
        # Auto-discover latest model under outputs
        auto_path = find_latest_model_path(str(output_root))
        if not auto_path:
            raise RuntimeError(
                "No model_path provided and no latest model found under outputs/. Pass model_path=... or create outputs/<run>/model.pt."
            )
        print(f"[generate] Using latest model: {auto_path}")
        model_path = Path(auto_path)
    else:
        model_path = Path(model_path_cfg).expanduser()
        if not model_path.is_absolute():
            model_path = (_REPO_ROOT / model_path).expanduser()

    try:
        with open_dict(cfg.generation):
            cfg.generation.model_path = str(model_path)
    except Exception:
        pass
    return model_path


def load_pretrained_model_and_tokenizer(cfg: DictConfig):
    device = resolve_device(cfg)
    output_root = _normalize_output_dir(cfg)
    model_path = _resolve_model_path(cfg, output_root)

    # Load tokenizer, supporting local JSON (dataset-specific) via PreTrainedTokenizerFast
    tok_path = cfg.tokenizer.tokenizer_name or cfg.tokenizer.pretrained_tokenizer_name
    if tok_path and not cfg.tokenizer.tokenizer_name:
        try:
            with open_dict(cfg.tokenizer):
                cfg.tokenizer.tokenizer_name = tok_path
        except Exception:
            pass
    if not tok_path:
        raise ValueError(
            "Config must define tokenizer_name or pretrained_tokenizer_name to load tokenizer."
        )
    if isinstance(tok_path, str) and Path(tok_path).exists() and tok_path.endswith('.json'):
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

    if getattr(cfg.model, "vocab_size", None) is None:
        try:
            with open_dict(cfg.model):
                cfg.model.vocab_size = len(tok)
        except Exception:
            pass

    # Build model from config and load weights
    model = TransformerNetwork(cfg).to(device)
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path_obj}. Provide model_path=/absolute/path/to/model.pt or place checkpoints under {cfg.paths.output_dir}."
        )
    ckpt = torch.load(str(model_path_obj), map_location=device)
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

    return model, tok
