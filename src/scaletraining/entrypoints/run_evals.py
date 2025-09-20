"""
Entrypoint for evaluating model performance across supported benchmarks
Currently supported benchmarks:

To run an eval on a single benchmark, use the --{benchmark} flag, otherwise all supported benchmarks will run
"""
from __future__ import annotations

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from scaletraining.util import flatten_cfg, resolve_device
from scaletraining.util.eval_utils import evaluate_perplexity
from scaletraining.data_processing import build_loaders

def benchmark_model(model, eval_loader):
    pass


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    for eset in flat.eval_datasets:
        # Download dataset
        dataset = load_dataset(eset)
        # Tokenize using tokenizer from config
        tokenizer = cfg.tokenizer  # assumes tokenizer is already loaded in config
        def tokenize_fn(example):
            return tokenizer(example["text"], truncation=True, padding="max_length", max_length=cfg.data.max_length)
        tokenized_dataset = dataset.map(tokenize_fn, batched=True)
        # Build DataLoader
        _, val_loaders = build_loaders(tokenized_dataset, for_training=False)

        num_workers = int(getattr(cfg, "loader_num_workers", 0))
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(getattr(cfg, "loader_pin_memory", False)),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(getattr(cfg, "loader_persistent_workers", False))
            prefetch = getattr(cfg, "loader_prefetch_factor", None)
            if prefetch:
                loader_kwargs["prefetch_factor"] = int(prefetch)

        eval_loader = DataLoader(
            tokenized_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        # Pass into model for evaluation
        model = cfg.model  # assumes model is already loaded in config
        benchmark_model(model, eval_loader)
    






if __name__ == "__main__":
    main()
