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
import torch
from transformers import AutoTokenizer

from scaletraining.model.model import TransformerNetwork
from scaletraining.util.utils import resolve_device, flatten_cfg
from scaletraining.inference.generation import generate_autoregressive


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

    model_path = getattr(flat, "model_path", None)
    if not model_path:
        raise RuntimeError("Provide model_path=/path/to/model.pt in CLI overrides.")

    prompt: str = flat.prompt
    max_new_tokens: int = int(flat.generation_max_tokens)
    temperature: float = float(flat.generation_temperature)
    top_k: Optional[int] = flat.generation_top_k
    if isinstance(top_k, str):
        top_k = int(top_k) if top_k.isdigit() else None

    # Build model from config and load weights
    model = TransformerNetwork(flat).to(flat.device)
    ckpt = torch.load(model_path, map_location=flat.device)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()

    tok = AutoTokenizer.from_pretrained(flat.tokenizer_name, use_fast=True)
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
