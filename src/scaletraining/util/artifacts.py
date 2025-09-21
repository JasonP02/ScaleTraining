"""Helpers for persisting training artifacts and metadata."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import _cfg_subset, config_fingerprint
from .path_utils import _sanitize


_REPO_ROOT = Path(__file__).resolve().parents[3]


def write_metadata(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception as exc:
        print(f"Warning: could not write metadata to {path}: {exc}")


def read_metadata(path: str) -> Dict[str, Any]:
    """Used across codebase for validating the similarity of run config to existing data, tokenizers, etc"""
    try:
        with open(os.path.join(path, "metadata.json"), "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Warning: could not read metadata, returning empty dictionary: {exc}")
        return {}


def save_run_manifest(cfg: Any, out_dir: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """Used for saving the model configuration"""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset": _cfg_subset(cfg),
        "optimizer": {
            "primary": cfg.primary_optimizer,
            "lr": cfg.lr,
            "beta": cfg.beta,
            "beta2": cfg.beta2,
            "weight_decay": cfg.weight_decay,
            "ns_iters": cfg.ns_iters,
            "eps": cfg.eps,
        },
        "training": {
            "batch_size": cfg.batch_size,
            "accum_steps": cfg.accum_steps,
            "effective_batch_size": cfg.batch_size * cfg.accum_steps,
            "grad_clip_norm": cfg.grad_clip_norm,
            "logits_chunk_size": cfg.logits_chunk_size,
            "device": cfg.device,
        },
        "transformer": {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embed": cfg.n_embed,
            "n_hidden": cfg.n_hidden,
            "activation": getattr(cfg, "activation", "relu"),
            "vocab_size": cfg.vocab_size,
            "UE_bias": cfg.UE_bias,
            "use_checkpoint": cfg.use_checkpoint,
        },
        "tokenizer": {
            "tokenizer_name": cfg.tokenizer_name,
            "tokenizer_type": cfg.tokenizer_type,
        },
        "dataset_tag": cfg.dataset_tag,
        "fingerprint": config_fingerprint(cfg),
        "implementation": {
            "optimizer": "baseline_adam" if cfg.use_baseline_adam else cfg.primary_optimizer,
            "rope": {
                "enabled": bool(getattr(cfg, "use_rope", True)),
                "theta": cfg.rope_config.get("theta", 10000),
            },
        },
    }
    if extra:
        manifest.update(extra)
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def save_model(model: torch.nn.Module, cfg: Any, out_root: Optional[str] = None) -> str:
    out_root = out_root or cfg.output_dir
    if out_root and not os.path.isabs(out_root):
        # Prefer cwd resolution if already absolute, otherwise anchor to repo root
        cwd_candidate = Path.cwd() / out_root
        if cwd_candidate.exists():
            out_root = str(cwd_candidate.resolve(strict=False))
        else:
            out_root = str((_REPO_ROOT / out_root).resolve(strict=False))
    tag = _sanitize(cfg.dataset_tag)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fingerprint = config_fingerprint(cfg)[:8]
    run_dir_name = "__".join(filter(None, [tag, f"v={fingerprint}", timestamp]))
    run_dir = os.path.join(out_root, run_dir_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.pt")
    base_mod = getattr(model, "_orig_mod", model)
    state = base_mod.state_dict()
    torch.save({"state_dict": state}, model_path)

    save_run_manifest(cfg, run_dir)
    return run_dir


def find_latest_model_path(output_root: str) -> Optional[str]:
    """
    Return path to the newest \"model.pt\" under `output_root`, if present.
    Used for model generation
    """
    try:
        root = Path(output_root)
        if not root.is_absolute() and not root.exists():
            repo_candidate = (_REPO_ROOT / root).resolve(strict=False)
            if repo_candidate.exists():
                root = repo_candidate
        if not root.exists():
            return None

        latest_link = root / "latest"
        if latest_link.exists():
            link_target = latest_link.resolve()
            candidate = link_target / "model.pt"
            if candidate.exists():
                return str(candidate)

        candidates = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            candidate = child / "model.pt"
            if candidate.exists():
                try:
                    mtime = candidate.stat().st_mtime
                except Exception:
                    mtime = 0.0
                candidates.append((mtime, candidate))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return str(candidates[0][1])
    except Exception:
        return None


__all__ = [
    "write_metadata",
    "read_metadata",
    "save_run_manifest",
    "save_model",
    "find_latest_model_path",
]
