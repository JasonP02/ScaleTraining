"""Path helpers for dataset artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict

import os
from pathlib import Path
from typing import Any

from scaletraining.data_processing.dataset_utils import dataset_label
from .config import config_fingerprint

from omegaconf import DictConfig


def _sanitize(value: str) -> str:
    """Helper for filepaths and such"""
    return str(value).replace("/", "-").replace(" ", "_")

def _search_for_directory_with_tag(base: str, name: str) -> str | None:
    """Return an existing directory matching a pattern with any dataset tag."""
    root = Path(base)
    if not root.exists():
        return None

    pattern = f"tag=*__{name}"
    matches = sorted(root.glob(pattern))
    if matches:
        return str(matches[0])
    return None

def _cfg_subset_for_fingerprint(cfg: DictConfig) -> Dict[str, Any]:
    """Return the fingerprint-relevant subset of the flattened config."""

    _FINGERPRINT_FIELDS = (
        "hf_dataset_names",
        "hf_dataset_config_name",
        "tokenizer_name",
        "max_seq_len",
    )
    out: Dict[str, Any] = {}
    for key in _FINGERPRINT_FIELDS:
        out[key] = getattr(cfg, key, None)
    return out

def config_fingerprint(cfg: DictConfig) -> str:
    """Stable hash summarising dataset/tokenizer-relevant config values."""
    payload = json.dumps(_cfg_subset_for_fingerprint(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def resolve_directory_from_fingerprint(cfg, fingerprint, base):
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    dataset_id = dataset_label(
        getattr(cfg, "hf_dataset_names", None),
        getattr(cfg, "hf_dataset_config_name", None),
    )
    name = (
        f"{tag}ds={_sanitize(dataset_id)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__v={fingerprint}"
    )
    full = os.path.join(base, name)
    if tag or os.path.isdir(full):
        return full

    existing = _search_for_directory_with_tag(base, name)
    return existing or full

def get_tokenized_directory(cfg: DictConfig, for_training: bool = True) -> str:
    """Return the path for tokenized data based on current config"""
    fingerprint = config_fingerprint(cfg)[:8]
    if for_training:
        base = cfg.tokenized_train_path
    else:
        base = cfg.tokenized_eval_path
    return resolve_directory_from_fingerprint(cfg, fingerprint, base)


def get_packed_directory(
        cfg: DictConfig,
        for_training: bool = True,
        ) -> str:
    """Return the directory path for packed batches."""
    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.batched_tokenized_path
    return resolve_directory_from_fingerprint(cfg, fingerprint, base)


__all__ = ["get_tokenized_directory", "get_packed_directory", "_sanitize"]
