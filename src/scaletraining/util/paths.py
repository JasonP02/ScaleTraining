"""Path helpers for dataset artifacts."""
from __future__ import annotations

import os
from typing import Any

from .config import config_fingerprint


def _sanitize(value: str) -> str:
    """Helper for filepaths and such"""
    return str(value).replace("/", "-").replace(" ", "_")


def _dataset_descriptor(cfg: Any) -> str:
    """Build a descriptive token for dataset-related config knobs."""

    parts = [f"ds={_sanitize(getattr(cfg, 'hf_dataset_names', 'unknown'))}"]
    val_ds = getattr(cfg, "val_hf_dataset_names", None)
    if val_ds and val_ds != getattr(cfg, "hf_dataset_names", None):
        parts.append(f"val={_sanitize(val_ds)}")

    default_train = "train"
    default_val = "validation"
    train_split = getattr(cfg, "train_split", None)
    val_split = getattr(cfg, "val_split", None)
    if train_split not in (None, default_train) or val_split not in (None, default_val):
        train_label = _sanitize(train_split or "auto")
        val_label = _sanitize(val_split or "auto")
        parts.append(f"splits={train_label}-{val_label}")

    return "__".join(parts)


def tokenized_dir(cfg: Any) -> str:
    """Return the path for tokenized shards."""

    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}{_dataset_descriptor(cfg)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fingerprint}"
    )
    return os.path.join(base, name)


def packed_dir(cfg: Any) -> str:
    """Return the directory path for packed batches."""

    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.batched_tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}{_dataset_descriptor(cfg)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fingerprint}"
    )
    return os.path.join(base, name)


__all__ = ["tokenized_dir", "packed_dir", "_sanitize"]
