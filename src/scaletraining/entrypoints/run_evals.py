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

from scaletraining.util import flatten_cfg, resolve_device
from scaletraining.util.eval_utils import evaluate_perplexity
from scaletraining.data_processing import build_loaders


@hydra.main(version_base=None, config_path=str(Path(__file__).parent.parent.parent.parent / "conf"), config_name="config")
def main(cfg: DictConfig) -> None:
    flat = flatten_cfg(cfg)
    resolve_device(flat)

    _, val_loaders = build_loaders(flat, for_training=False)
    






if __name__ == "__main__":
    main()
