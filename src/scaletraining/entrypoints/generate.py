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
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from scaletraining.model.model import TransformerNetwork
from scaletraining.util.utils import resolve_device, flatten_cfg, find_latest_model_path
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
    if not model_path or str(model_path).lower() == "latest":
        # Auto-discover latest model under outputs
        output_root = getattr(flat, "output_dir", "outputs")
        auto_path = find_latest_model_path(output_root)
        if not auto_path:
            raise RuntimeError("No model_path provided and no latest model found under outputs/. Pass model_path=... or create outputs/<run>/model.pt.")
        print(f"[generate] Using latest model: {auto_path}")
        model_path = auto_path

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
    # Normalize keys from compiled/DataParallel checkpoints if present
    def _strip_prefix(sd, prefix: str):
        if any(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd
    state_dict = _strip_prefix(state_dict, "_orig_mod.")
    state_dict = _strip_prefix(state_dict, "module.")
    model.load_state_dict(state_dict)
    model.eval()

    # Load tokenizer, supporting local JSON (dataset-specific) via PreTrainedTokenizerFast
    tok_path = flat.tokenizer_name
    from pathlib import Path as _P
    if isinstance(tok_path, str) and _P(tok_path).exists() and tok_path.endswith('.json'):
        tok = PreTrainedTokenizerFast(tokenizer_file=tok_path)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
    else:
        tok = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": ""})
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
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
