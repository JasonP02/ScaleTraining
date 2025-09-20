"""Public utility surface for scaletraining."""

from .artifacts import (
    find_latest_model_path,
    read_metadata,
    save_model,
    save_run_manifest,
    write_metadata,
)
from .config import _cfg_subset, config_fingerprint, flatten_cfg
from .device import clear_cuda_cache, configure_rocm_and_sdp, resolve_device
from .model_stats import count_parameters, humanize_bytes, humanize_params
from .path_utils import packed_dir, tokenized_dir
from .wandb_utils import init_wandb, log_eval_metrics, log_train_metrics

__all__ = [
    "clear_cuda_cache",
    "configure_rocm_and_sdp",
    "resolve_device",
    "init_wandb",
    "log_train_metrics",
    "log_eval_metrics",
    "flatten_cfg",
    "config_fingerprint",
    "_cfg_subset",
    "tokenized_dir",
    "packed_dir",
    "write_metadata",
    "read_metadata",
    "save_run_manifest",
    "save_model",
    "find_latest_model_path",
    "count_parameters",
    "humanize_params",
    "humanize_bytes",
]
