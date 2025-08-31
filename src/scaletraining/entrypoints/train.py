"""
Hydra-powered training entrypoint.

This wraps the existing trainer with Hydra config + Weights & Biases (W&B).
Run from CLI: `python -m scaletraining.entrypoints.train` or via console script.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

from scaletraining.data_processing import build_loaders
from scaletraining.model import TransformerNetwork
from scaletraining.util import configure_rocm_and_sdp, clear_cuda_cache
from scaletraining.util.utils import (
    tokenized_dir,
    packed_dir,
    init_wandb,
    log_dataset_artifacts,
    save_model,
    log_model_artifact,
    resolve_device,
)
from scaletraining.training.loop import training_run
from scaletraining.inference.generation import generate_autoregressive
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Train the model using Hydra config and log to W&B.

    Args:
        cfg: Hydra DictConfig with fields matching the previous Config dataclass.
             Access via attribute syntax, e.g., cfg.batch_size. See conf/config.yaml.
    """
    # Resolve device, configure kernels, and free any stale CUDA cache
    resolve_device(cfg)
    configure_rocm_and_sdp(cfg)
    clear_cuda_cache()

    # Initialize W&B early, logging the full resolved config
    init_wandb(cfg, OmegaConf.to_container(cfg, resolve=True))

    # Build data
    train_loader, val_loader = build_loaders(cfg)

    # Log dataset artifacts (tokenized + packed directories)
    try:
        log_dataset_artifacts(tokenized_dir(cfg), packed_dir(cfg), cfg)
    except Exception as e:
        print(f"Dataset artifact logging skipped: {e}")

    # Model + loss
    model = TransformerNetwork(cfg)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # summed CE, normalized per token in loop

    # Sanity check embedding size vs vocab size after metadata auto-set
    assert model.token_embedding.num_embeddings == cfg.vocab_size, (
        f"Model vocab ({model.token_embedding.num_embeddings}) != cfg.vocab_size ({cfg.vocab_size})"
    )

    # Training loop
    training_run(cfg, model, train_loader, loss_fn=loss_fn)

    # Save model and log as W&B model artifact
    try:
        run_dir = save_model(model, cfg, getattr(cfg, "output_dir", "outputs"))
        log_model_artifact(os.path.join(run_dir, "model.pt"), cfg)
    except Exception as e:
        print(f"Model save/logging skipped: {e}")

    # Optional: sample generation after training for quick qualitative check
    try:
        if bool(getattr(cfg, 'generate_after_train', False)):
            tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
            text = generate_autoregressive(
                model,
                tok,
                cfg.device,
                prompt=getattr(cfg, 'prompt', 'Once upon a time'),
                max_new_tokens=int(getattr(cfg, 'generation_max_tokens', 100)),
                temperature=float(getattr(cfg, 'generation_temperature', 1.0)),
                top_k=int(getattr(cfg, 'generation_top_k', 50)),
            )
            print("\n=== Generated Sample (post-train) ===\n" + text + "\n====================================\n")
            try:
                import wandb
                wandb.log({"generated_sample": text})
            except Exception:
                pass
    except Exception as e:
        print(f"Post-train generation skipped: {e}")


if __name__ == "__main__":
    main()
