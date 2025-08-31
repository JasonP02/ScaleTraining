"""
Hydra-powered text generation entrypoint.

Usage:
  scaletraining-generate model_path=/path/to/model.pt prompt="Once upon a time"
"""
from __future__ import annotations

from typing import Optional

import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoTokenizer

from scaletraining.model.model import TransformerNetwork
from scaletraining.util.utils import resolve_device
from scaletraining.inference.generation import generate_autoregressive


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
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
    resolve_device(cfg)

    model_path = getattr(cfg, "model_path", None)
    if not model_path:
        raise RuntimeError("Provide model_path=/path/to/model.pt in CLI overrides.")

    prompt: str = getattr(cfg, "prompt", "Once upon a time")
    max_new_tokens: int = int(getattr(cfg, "generation_max_tokens", 100))
    temperature: float = float(getattr(cfg, "generation_temperature", 1.0))
    top_k: Optional[int] = getattr(cfg, "generation_top_k", 50)
    if isinstance(top_k, str):
        top_k = int(top_k) if top_k.isdigit() else None

    # Build model from config and load weights
    model = TransformerNetwork(cfg).to(cfg.device)
    ckpt = torch.load(model_path, map_location=cfg.device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
    text = generate_autoregressive(
        model,
        tok,
        cfg.device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    print("\n=== Generated Sample ===\n" + text + "\n=======================\n")


if __name__ == "__main__":
    main()
