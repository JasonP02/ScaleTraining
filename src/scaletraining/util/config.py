"""Configuration helpers for Hydra interoperability and metadata fingerprints."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


_FINGERPRINT_FIELDS = (
    "hf_dataset_names",
    "tokenizer_name",
    "max_seq_len",
    "use_attention_mask",
)


def _cfg_subset(cfg: Any) -> Dict[str, Any]:
    """Return the fingerprint-relevant subset of the flattened config."""

    out: Dict[str, Any] = {}
    for key in _FINGERPRINT_FIELDS:
        out[key] = getattr(cfg, key)
    return out


def config_fingerprint(cfg: Any) -> str:
    """Stable hash summarising dataset/tokenizer-relevant config values."""

    payload = json.dumps(_cfg_subset(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_cfg(cfg: Any) -> Any:
    """Flatten namespaced Hydra config groups for ease of use"""

    from types import SimpleNamespace

    if isinstance(cfg, SimpleNamespace):
        # Return a shallow copy so callers can mutate safely
        return SimpleNamespace(**vars(cfg))

    group_names = ("transformer", "tokenizer", "logging", "moe")

    try:
        from omegaconf import OmegaConf

        def to_dict(node: Any) -> Dict[str, Any]:
            return OmegaConf.to_container(node, resolve=True) if node is not None else {}

        def is_config(node: Any) -> bool:
            return OmegaConf.is_config(node)

    except Exception:

        def to_dict(node: Any) -> Dict[str, Any]:
            return dict(node) if node is not None else {}

        def is_config(node: Any) -> bool:
            return hasattr(node, "keys")

    if not any(hasattr(cfg, group) for group in group_names):
        if is_config(cfg):
            values = to_dict(cfg)
            if isinstance(values, dict):
                return SimpleNamespace(**values)
        if isinstance(cfg, dict):
            return SimpleNamespace(**cfg)
        if hasattr(cfg, "__dict__"):
            return SimpleNamespace(**vars(cfg))
        return SimpleNamespace()

    merged: Dict[str, Any] = {}
    for group in group_names:
        try:
            sub = cfg.get(group) if hasattr(cfg, "get") else getattr(cfg, group, None)
        except Exception:
            sub = getattr(cfg, group, None)
        if sub is not None:
            values = to_dict(sub)
            if isinstance(values, dict):
                # Hydra group DictConfigs often serialize as {group: {...}}
                nested = values.get(group)
                if isinstance(nested, dict) and len(values) == 1:
                    values = nested
                merged.update(values)
    return SimpleNamespace(**merged)


__all__ = ["_cfg_subset", "config_fingerprint", "flatten_cfg"]
