"""Hydra config package initialisation."""

from __future__ import annotations

from numbers import Real
from typing import Any

from omegaconf import OmegaConf


def _mul_resolver(*values: Any) -> Any:
    """Multiply numeric resolver arguments, returning int when possible."""

    if not values:
        raise ValueError("mul resolver requires at least one argument")

    product = 1.0
    all_int = True
    for value in values:
        if value is None:
            raise ValueError("mul resolver received None")
        if isinstance(value, Real):
            numeric = float(value)
            all_int = all_int and float(value).is_integer()
        else:
            numeric = float(value)
            all_int = all_int and numeric.is_integer()
        product *= numeric

    if all_int and product.is_integer():
        return int(product)
    return product


try:  # avoid double registration when Hydra reloads configs
    OmegaConf.register_new_resolver("mul", _mul_resolver)
except Exception:
    pass


__all__ = ["_mul_resolver"]
