"""Helper routines for structured Weights & Biases logging."""
from __future__ import annotations

from pathlib import Path as PathLib
import typing as t


try:  # Lazy optional dependency
    import wandb as wandb_sdk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - logging simply becomes a no-op
    wandb_sdk = None  # type: ignore


def _log(metrics: t.Mapping[str, float], *, step: t.Optional[int] = None) -> None:
    """Safely dispatch metrics to W&B if the library is available."""

    if wandb_sdk is None or not metrics:
        return
    # wandb.log mutates the dict, so pass a shallow copy.
    payload: t.MutableMapping[str, float] = dict(metrics)
    wandb_sdk.log(payload, step=step)


def log_train_metrics(
    *,
    used_tokens: int,
    loss: float,
    lr: float,
    throughput: float,
    flops_used: float,
) -> None:
    """Log core training-loop statistics keyed by total tokens processed."""

    _log(
        {
            "used tokens": used_tokens,
            "train_per_token_loss": loss,
            "lr": lr,
            "throughput_tokens_per_s": throughput,
            "FLOPs": flops_used,
        },
        step=used_tokens,
    )


def log_eval_metrics(
    *,
    used_tokens: int,
    val_loss: float,
    val_perplexity: float,
) -> None:
    """Log validation loss/perplexity keyed by the training token count."""

    _log(
        {
            "used tokens": used_tokens,
            "valid_per_token_loss": val_loss,
            "valid_ppl": val_perplexity,
        },
        step=used_tokens,
    )


def init_wandb(cfg: t.Any, config_dict: t.Optional[t.Mapping[str, t.Any]] = None) -> None:
    """Initialise W&B with descriptive names derived from the tokenizer."""

    if wandb_sdk is None:
        return

    tokenizer_name = getattr(cfg, "tokenizer_name", "unknown")
    token_path = PathLib(tokenizer_name)
    is_custom = token_path.exists() and token_path.suffix == ".json"

    if is_custom:
        if "roneneldan_TinyStories" in tokenizer_name:
            name_suffix = "custom_tinystories"
        else:
            name_suffix = "custom"
        tags = ["custom_tokenizer"]
    else:
        if "gpt-neo" in tokenizer_name:
            name_suffix = "gpt_neo"
        else:
            name_suffix = tokenizer_name.split("/")[-1] if "/" in tokenizer_name else tokenizer_name
        tags = ["hf_tokenizer"]

    wandb_sdk.init(
        project=cfg.wandb_project_name,
        config=dict(config_dict) if config_dict is not None else None,
        reinit=True,
        name=f"sweep_{name_suffix}",
        tags=tags,
    )


__all__ = ["log_train_metrics", "log_eval_metrics", "init_wandb"]
