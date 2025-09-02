"""
Hydra-powered training entrypoint.

This wraps the existing trainer with Hydra config + Weights & Biases (W&B).
Run from CLI: `python -m scaletraining.entrypoints.train` or via console script.
"""
from __future__ import annotations

from pathlib import Path
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn

from scaletraining.data_processing import build_loaders
from scaletraining.model import TransformerNetwork
from scaletraining.util import configure_rocm_and_sdp, clear_cuda_cache
from scaletraining.util.utils import (
    init_wandb,
    save_model,
    resolve_device,
)
from scaletraining.training.loop import training_run
from scaletraining.inference.generation import generate_autoregressive
from transformers import AutoTokenizer


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> float:
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

    # Dataset artifact logging intentionally disabled.

    # Model + loss
    model = TransformerNetwork(cfg)
    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # summed CE, normalized per token in loop

    # Sanity check embedding size vs vocab size after metadata auto-set
    assert model.token_embedding.num_embeddings == cfg.vocab_size, (
        f"Model vocab ({model.token_embedding.num_embeddings}) != cfg.vocab_size ({cfg.vocab_size})"
    )

    # Training loop
    stats = training_run(cfg, model, train_loader, loss_fn=loss_fn)

    # Save model locally only
    run_dir = save_model(model, cfg, cfg.output_dir)
    print(f"Model saved locally to: {run_dir}")

    # Optional: sample generation after training for quick qualitative check
    if cfg.generate_after_train:
        try:
            tok = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
            text = generate_autoregressive(
                model,
                tok,
                cfg.device,
                prompt=cfg.prompt,
                max_new_tokens=int(cfg.generation_max_tokens),
                temperature=float(cfg.generation_temperature),
                top_k=int(cfg.generation_top_k),
            )
            print("\n=== Generated Sample (post-train) ===\n" + text + "\n====================================\n")
        except Exception as e:
            print(f"Post-train generation skipped: {e}")

    # Persist a lightweight result.json in the job directory for easy aggregation
    job_result = {
        "final_train_loss": float(stats['train_loss'][-1]) if stats.get('train_loss') else None,
        "primary_optimizer": cfg.primary_optimizer,
        "rope_implementation": cfg.rope_implementation,
        "lr": float(cfg.lr),
        "batch_size": int(cfg.batch_size),
        "accum_steps": int(cfg.accum_steps),
        "max_seq_len": int(cfg.max_seq_len),
        "n_layer": int(cfg.n_layer),
        "n_head": int(cfg.n_head),
        "n_embed": int(cfg.n_embed),
    }
    with open(Path.cwd() / "result.json", "w", encoding="utf-8") as f:
        json.dump(job_result, f, indent=2, sort_keys=True)
    # Also print a single-line summary that's easy to grep
    print("RESULT:", json.dumps(job_result))

    # Return an objective for Hydra sweepers (e.g., Optuna)
    if stats.get('train_loss'):
        return float(stats['train_loss'][-1])
    return float('inf')

if __name__ == "__main__":
    # Standard Hydra entrypoint; objective value returned for sweepers
    main()

