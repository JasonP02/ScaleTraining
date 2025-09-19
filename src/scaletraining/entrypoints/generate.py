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

from scaletraining.util import flatten_cfg, resolve_device
from scaletraining.inference.generation import generate_autoregressive
from scaletraining.util.eval_utils import load_pretrained_model_and_tokenizer


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Generate text from a saved model.

    Required Hydra overrides:
      - model_path: str, filesystem path to a torch checkpoint saved by save_model().

    Optional relevant config keys:
      - prompt: str, seed text for generation (default: simple story prompt)
      - generation_max_tokens: int, number of tokens to generate
      - generation_temperature: float, softmax temperature (>0)
      - generation_top_k: int, top-k filtering; set 0/None to disable
    """
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    model, tok = load_pretrained_model_and_tokenizer(flat)
    prompt: str = flat.prompt
    max_new_tokens: int = int(flat.generation_max_tokens)
    temperature: float = float(flat.generation_temperature)
    top_k: Optional[int] = flat.generation_top_k
    if isinstance(top_k, str):
        top_k = int(top_k) if top_k.isdigit() else None


    text = generate_autoregressive(
        model,
        tok,
        flat.device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print("\n=== Generated Sample ===\n" + text + "\n=======================\n")


if __name__ == "__main__":
    main()
