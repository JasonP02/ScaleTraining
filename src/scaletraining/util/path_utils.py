"""Path helpers for dataset artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from omegaconf import DictConfig

from scaletraining.data_processing.dataset_utils import dataset_label


def _sanitize(value: str) -> str:
    """Translates huggingface naming schema to be os friendly"""
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

def _first_non_empty(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if value not in (None, "", "null"):
            return value
    return None


def get_cfg_subset(cfg: DictConfig) -> Dict[str, Any]:
    """Return the fingerprint-relevant subset of the config."""

    tokenizer = cfg.tokenizer
    model = cfg.model

    subset: Dict[str, Any] = {
        "dataset_names": list(tokenizer.dataset_names),
        "dataset_tag": list(tokenizer.dataset_tag),
        "tokenizer_name": tokenizer.tokenizer_name or tokenizer.pretrained_tokenizer_name,
        "max_seq_len": model.max_seq_len,
    }
    if tokenizer.hf_dataset_config_name is not None:
        subset["hf_dataset_config_name"] = tokenizer.hf_dataset_config_name
    return subset

def config_fingerprint(cfg: DictConfig) -> str:
    """Stable hash summarising dataset/tokenizer-relevant config values."""
    payload = json.dumps(get_cfg_subset(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def resolve_directory_from_fingerprint(cfg: DictConfig, fingerprint: str, base: str) -> str:
    tokenizer = cfg.tokenizer
    model = cfg.model

    primary_tag = _first_non_empty(tokenizer.dataset_tag)
    tag = f"tag={_sanitize(primary_tag)}__" if primary_tag else ""
    dataset_id = dataset_label(tokenizer.dataset_names, tokenizer.dataset_tag)
    tokenizer_name = tokenizer.tokenizer_name or tokenizer.pretrained_tokenizer_name
    name = (
        f"{tag}ds={_sanitize(dataset_id)}__"
        f"tok={_sanitize(tokenizer_name)}__"
        f"L={model.max_seq_len}__v={fingerprint}"
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
        base = cfg.paths.tokenized_train_path
    else:
        base = cfg.paths.tokenized_eval_path
    return resolve_directory_from_fingerprint(cfg, fingerprint, base)


def get_packed_directory(cfg: DictConfig) -> str:
    """Return the directory path for packed batches."""
    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.paths.batched_tokenized_path
    return resolve_directory_from_fingerprint(cfg, fingerprint, base)


__all__ = [
    "get_cfg_subset",
    "config_fingerprint",
    "get_tokenized_directory",
    "get_packed_directory",
    "_sanitize",
]
