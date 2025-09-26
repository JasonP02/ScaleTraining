"""
Hydra-powered text generation entrypoint.

Usage:
  scaletraining-generate model_path=/path/to/model.pt prompt="Once upon a time"
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig

from scaletraining.config import load_project_config
from scaletraining.util.device import resolve_device
from scaletraining.util.eval_utils import load_pretrained_model_and_tokenizer
from scaletraining.util.generation_utils import generate_autoregressive


def _coerce_top_k(value: Optional[object]) -> Optional[int]:
    """Normalise top-k configuration into an optional positive integer."""

    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            parsed = int(stripped)
            return parsed if parsed > 0 else None
    return None


def generate_text(cfg: DictConfig) -> str:
    """Generate a text sample using the provided Hydra config."""

    cfg = load_project_config(cfg)
    model, tokenizer = load_pretrained_model_and_tokenizer(cfg)
    device = resolve_device(cfg)

    generation_cfg = cfg.generation

    prompt: str = generation_cfg.prompt
    max_new_tokens: int = int(generation_cfg.generation_max_tokens)
    temperature: float = float(generation_cfg.generation_temperature)
    top_k = _coerce_top_k(generation_cfg.generation_top_k)

    return generate_autoregressive(
        model,
        tokenizer,
        device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.parent.parent.parent / "conf"),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Generate text from a saved model."""

    text = generate_text(cfg)
    print("\n=== Generated Sample ===\n" + text + "\n=======================\n")


if __name__ == "__main__":
    main()
