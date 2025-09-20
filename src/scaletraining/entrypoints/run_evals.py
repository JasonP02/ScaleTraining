"""
Entrypoint for evaluating model performance across supported benchmarks
Currently supported benchmarks:

To run an eval on a single benchmark, use the --{benchmark} flag, otherwise all supported benchmarks will run
"""
from __future__ import annotations
from dataclasses import dataclass

from pathlib import Path
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

from scaletraining.util import flatten_cfg, resolve_device
from scaletraining.util.eval_utils import evaluate_perplexity
from scaletraining.data_processing import build_loaders, get_loader_kwargs
from scaletraining.util.eval_utils import load_pretrained_model_and_tokenizer

def benchmark_model(model, eval_loader):
    print(f"benchmarking {eval_loader}")

def tokenize_fn(example, column, tokenizer):
    return tokenizer(example[column], truncation=True, padding="max_length")

def eval_on_gsm8k(cfg, model, tok):
    print("in eval")
    dataset = load_dataset('openai/gsm8k', 'main')
    tokenized_dataset = dataset.map(tokenize_fn, batched=True)

    loader_kwargs = get_loader_kwargs(cfg)

    eval_loader = DataLoader(
        tokenized_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    model = cfg.model  # assumes model is already loaded in config
    print(eval_loader)
    benchmark_model(model, eval_loader)


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    model, tok = load_pretrained_model_and_tokenizer(flat)

    eval_on_gsm8k(flat, model, tok)







if __name__ == "__main__":
    main()
