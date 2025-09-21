"""Configuration helpers for Hydra interoperability and metadata fingerprints."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


_FINGERPRINT_FIELDS = (
    "hf_dataset_names",
    "tokenizer_name",
    "max_seq_len",
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
    """Return a SimpleNamespace view of a Hydra config."""

    from types import SimpleNamespace

    if isinstance(cfg, SimpleNamespace):
        return SimpleNamespace(**vars(cfg))

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

    if is_config(cfg):
        values = to_dict(cfg)
    elif isinstance(cfg, dict):
        values = dict(cfg)
    elif hasattr(cfg, "__dict__"):
        return SimpleNamespace(**vars(cfg))
    else:
        return SimpleNamespace()

    values.pop("hydra", None)
    return SimpleNamespace(**values)


__all__ = ["_cfg_subset", "config_fingerprint", "flatten_cfg"]
