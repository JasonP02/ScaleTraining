"""Path helpers for dataset artifacts."""
from __future__ import annotations

import os
from typing import Any

from .config import config_fingerprint


def _sanitize(value: str) -> str:
    return str(value).replace("/", "-").replace(" ", "_")


def tokenized_dir(cfg: Any) -> str:
    """Return the canonical directory path for tokenized shards."""

    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fingerprint}"
    )
    return os.path.join(base, name)


def packed_dir(cfg: Any) -> str:
    """Return the canonical directory path for packed batches."""

    fingerprint = config_fingerprint(cfg)[:8]
    base = cfg.batched_tokenized_path
    tag = f"tag={_sanitize(cfg.dataset_tag)}__" if getattr(cfg, "dataset_tag", "") else ""
    name = (
        f"{tag}ds={_sanitize(cfg.hf_dataset_names)}__"
        f"tok={_sanitize(cfg.tokenizer_name)}__"
        f"L={cfg.max_seq_len}__mask={int(cfg.use_attention_mask)}__v={fingerprint}"
    )
    return os.path.join(base, name)


__all__ = ["tokenized_dir", "packed_dir", "_sanitize"]
