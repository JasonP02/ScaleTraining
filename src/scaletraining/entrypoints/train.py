"""
Hydra-powered training entrypoint.

This wraps the existing trainer with Hydra config + Weights & Biases (W&B).
Run from CLI: `python -m scaletraining.entrypoints.train` or via console script.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch.nn as nn
import torch

from scaletraining.data_processing import build_loaders
from scaletraining.data_processing.tokenization import get_tokenizer_name_from_dataset
from scaletraining.model import TransformerNetwork
from scaletraining.util import (
    clear_cuda_cache,
    configure_rocm_and_sdp,
    init_wandb,
    save_model,
)
from scaletraining.model.training_loop import training_run
from scaletraining.util.generation_utils import generate_autoregressive
from scaletraining.util.model_stats import (
    count_parameters,
    humanize_bytes,
    humanize_params,
)

from tokenizers import Tokenizer as _RawTokenizer
from scaletraining.data_processing.tokenizer import Tokenizer


LOGGER = logging.getLogger(__name__)


class DatasetSpec:
    def __init__(self):
        self.train_dataset_paths: list[str] = self.get_training_data_paths
        self.tokenizer = Tokenizer()
    
    def get_training_data_paths(self):
        pass

@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> float:
    """
    Train the model using Hydra config and log to W&B.
    """

    # Resolve device, configure kernels, and free any stale CUDA cache
    configure_rocm_and_sdp(cfg.device)
    clear_cuda_cache()

    tokenizer = Tokenizer(cfg.tokenizer)

    init_wandb(cfg, tok=tokenizer.tok, tokenizer_vocab_size=tokenizer.vocab_size)

    train_loader, val_loader = build_loaders(cfg, for_training=True)


    # Dataset artifact logging intentionally disabled.

    # Model + loss
    model = TransformerNetwork(cfg)

    total_params, trainable_params = count_parameters(model)
    readable_total = humanize_params(total_params)
    readable_trainable = humanize_params(trainable_params)
    bytes_fp32 = total_params * 4
    bytes_bf16 = total_params * 2

    size_msg = (
        f"Model parameters: {total_params:,} ({readable_total}); "
        f"Trainable: {trainable_params:,} ({readable_trainable}); "
        f"Approx size fp32: {humanize_bytes(bytes_fp32)}, bf16/fp16: {humanize_bytes(bytes_bf16)}"
    )
    print(size_msg)
    LOGGER.info(size_msg)

    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "model/total_params": total_params,
                    "model/trainable_params": trainable_params,
                    "model/size_bytes_fp32": bytes_fp32,
                    "model/size_bytes_bf16": bytes_bf16,
                },
                step=0,
            )
            wandb.run.summary["model/total_params"] = total_params
            wandb.run.summary["model/trainable_params"] = trainable_params
    except ModuleNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - W&B logging is best-effort
        LOGGER.warning("Failed to log model size to W&B: %s", exc)

    # Compile model for massive speedups
    model = torch.compile(model, mode="max-autotune")
    loss_fn = nn.CrossEntropyLoss(reduction='sum')  # summed CE, normalized per token in loop

    # Sanity check embedding size vs vocab size after metadata auto-set
    assert model.token_embedding.num_embeddings == flat.vocab_size, (
        f"Model vocab ({model.token_embedding.num_embeddings}) != cfg.vocab_size ({flat.vocab_size})"
    )

    # Training loop
    stats = training_run(flat, model, train_loader, loss_fn=loss_fn, val_loader=val_loader)

    # Save model locally only
    run_dir = save_model(model, flat, flat.output_dir)
    print(f"Model saved locally to: {run_dir}")


    # Persist a lightweight result.json in the job directory for easy aggregation
    job_result = {
        "final_train_loss": float(stats['train_loss'][-1]) if stats.get('train_loss') else None,
        "primary_optimizer": flat.primary_optimizer,
        "use_rope": bool(flat.use_rope),
        "lr": float(flat.lr),
        "batch_size": int(flat.batch_size),
        "accum_steps": int(flat.accum_steps),
        "max_seq_len": int(flat.max_seq_len),
        "n_layer": int(flat.n_layer),
        "n_head": int(flat.n_head),
        "n_embed": int(flat.n_embed),
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
