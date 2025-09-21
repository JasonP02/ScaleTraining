"""Path helpers for dataset artifacts."""
from __future__ import annotations

import os
from typing import Any

from .config import config_fingerprint


def _sanitize(value: str) -> str:
    """Helper for filepaths and such"""
    return str(value).replace("/", "-").replace(" ", "_")


def get_tokenized_directory(cfg: Any, for_training: bool = True) -> str:
    """
    Return the path for tokenized shards based on current config.
    Knobs:
        huggingface dataset(s)
        tokenizer 
        sequence length
    """

    fingerprint = config_fingerprint(cfg)[:8]
    if for_training:
        base = cfg.tokenized_train_path
    else:
        base = cfg.tokenized_eval_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__v={fingerprint}"
    )
    return os.path.join(base, name)


def get_packed_directory(cfg: Any, for_training: bool = True) -> str:
    """Return the directory path for packed batches."""

    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.batched_tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__v={fingerprint}"
    )
    return os.path.join(base, name)


__all__ = ["get_tokenized_directory", "get_packed_directory", "_sanitize"]
