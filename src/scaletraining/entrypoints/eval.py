"""
Hydra-powered evaluation entrypoint.

Computes validation perplexity/loss on the prepared dataset using the configured model.

Usage:
  python -m scaletraining.entrypoints.eval
  # or via console script if configured
"""
from __future__ import annotations

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn

from scaletraining.util.utils import resolve_device, flatten_cfg
from scaletraining.data_processing import build_loaders
from scaletraining.model import TransformerNetwork
from scaletraining.training.loop import evaluate_perplexity


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    # Data
    _, val_loader = build_loaders(flat)
    if val_loader is None:
        raise RuntimeError("No validation split found. Ensure your dataset includes a validation/test split.")

    # Model
    model = TransformerNetwork(flat).to(flat.device)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    # Evaluate
    v_loss, v_ppl = evaluate_perplexity(model, val_loader, flat, loss_fn, max_batches=int(getattr(flat, 'eval_max_batches', 0)))
    print(f"Validation loss: {v_loss:.6f}\nValidation ppl:  {v_ppl:.3f}")


if __name__ == "__main__":
    main()

