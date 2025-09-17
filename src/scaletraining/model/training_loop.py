"""
Functional training loop utilities.

These helpers implement the training loop without an object-oriented trainer.
Each function has a narrow purpose and explicit inputs/outputs.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from scaletraining.evals import evaluate_perplexity
from scaletraining.util.training_utils import (
    apply_moe_schedules,
    build_optimizers,
    compute_loss_sum,
    compute_lr_scale_tokens,
    compute_progress_t,
    prepare_targets,
    scale_optimizer_lr,
    split_model_matrix_params,
    log_implementation,
    estimate_flops
)
from scaletraining.util.wandb_utils import log_eval_metrics, log_train_metrics


def training_run(
    cfg,
    model: nn.Module,
    train_loader: DataLoader,
    *,
    loss_fn: nn.Module,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, list]:
    """Functional training loop until reaching token budget.

    Args:
        cfg: Hydra config with fields used: device, accum_steps, grad_clip_norm,
             logits_chunk_size, max_train_tokens, debug_memory.
        model: nn.Module with `forward_hidden` and `W_ue` attributes.
        train_loader: DataLoader yielding dicts with 'input_ids'.
        loss_fn: nn.CrossEntropyLoss(reduction='sum') for per-token normalization.
    Returns:
        stats: dict with key 'train_loss' (list of averaged per-token losses per accumulation window).
    """

    matrix_params, other_params = split_model_matrix_params(
        model.named_parameters(), model=model, cfg=cfg
    )
    if cfg.log_implementation_details:
        log_implementation(matrix_params, other_params)
        
    opt_primary, opt_secondary = build_optimizers(cfg, matrix_params, other_params)

    primary_base_lr = (float(opt_primary.param_groups[0]["lr"]) if opt_primary is not None else 0.0)
    secondary_base_lr = (float(opt_secondary.param_groups[0]["lr"]) if opt_secondary is not None else 0.0)

    model.to(cfg.device)
    model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stats = {"train_loss": []}
    used_tokens = 0
    step_in_accum = 0
    accum_loss_sum = 0.0
    accum_token_count = 0
    last_eval_tokens = 0

    stop_training = False
    while used_tokens < cfg.max_train_tokens and not stop_training:
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(cfg.device)

            ctx = (
                autocast(device_type="cuda", dtype=torch.bfloat16)
                if (cfg.device == "cuda" and torch.cuda.is_available())
                else contextlib.nullcontext()
            )
            with ctx:
                hidden = model.forward_hidden(input_ids)
                hidden = hidden[:, :-1, :]
                targets, effective = prepare_targets(input_ids)
                loss_sum = compute_loss_sum(
                    model, hidden, targets, cfg.logits_chunk_size, loss_fn
                )
                per_token_loss = loss_sum / max(1, effective)

                aux = (
                    model.moe_aux_loss()
                    if hasattr(model, "moe_aux_loss")
                    else hidden.new_tensor(0.0, dtype=torch.float32)
                )
                total_loss = per_token_loss + float(cfg.moe_lb_coef) * aux.to(per_token_loss.dtype)

                loss = total_loss / cfg.accum_steps

            loss.backward()
            accum_loss_sum += float(loss_sum.item())
            accum_token_count += int(effective)
            step_in_accum += 1

            used_tokens += int(effective)

            if idx % 100 == 0:
                flops_used = estimate_flops(
                    tokens_used=used_tokens,
                    d_model=cfg.n_embed,
                    d_hidden=cfg.n_hidden,
                    n_heads=cfg.n_head,
                    seq_len=cfg.max_seq_len,
                    n_layers=cfg.n_layer,
                    n_moe_layers=cfg.moe_n_layers,
                    top_k=cfg.moe_top_k,
                    n_experts=cfg.moe_n_experts,
                    using_moe=cfg.use_moe
                )

            if step_in_accum == cfg.accum_steps:
                import time

                _t0 = time.time()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                lr_scale = compute_lr_scale_tokens(used_tokens, cfg)
                progress_t = compute_progress_t(used_tokens, cfg)
                cfg.moe_lb_coef = apply_moe_schedules(model, cfg, progress_t)

                scale_optimizer_lr(opt_primary, primary_base_lr, lr_scale)
                scale_optimizer_lr(opt_secondary, secondary_base_lr, lr_scale)

                opt_primary.step()
                if opt_secondary is not None:
                    opt_secondary.step()

                opt_primary.zero_grad(set_to_none=True)
                if opt_secondary is not None:
                    opt_secondary.zero_grad(set_to_none=True)
                step_in_accum = 0

                avg_loss = accum_loss_sum / max(1, accum_token_count)
                stats["train_loss"].append(avg_loss)
                current_lr = (opt_primary.param_groups[0]["lr"] if opt_primary is not None else 0.0)
                elapsed = max(1e-6, time.time() - _t0)
                tps = accum_token_count / elapsed if accum_token_count > 0 else 0.0
                print(
                    f"Tokens: {used_tokens:,}, Loss: {avg_loss:.4f}, LR: {current_lr:.6g}, tok/s: {tps:.0f}"
                )
                log_train_metrics(
                    used_tokens=used_tokens,
                    loss=avg_loss,
                    lr=current_lr,
                    throughput=tps,
                    flops_used = flops_used
                )
                accum_loss_sum = 0.0
                accum_token_count = 0

                eval_interval = cfg.eval_interval_tokens
                max_val_batches = cfg.eval_max_batches

                if (val_loader is not None
                    and eval_interval > 0
                    and (used_tokens - last_eval_tokens) >= eval_interval
                ):
                    v_loss, v_ppl = evaluate_perplexity(
                        model,
                        val_loader,
                        cfg,
                        loss_fn,
                        max_batches=max_val_batches,
                    )
                    print(f"[eval] tokens={used_tokens:,} val_loss={v_loss:.4f} val_ppl={v_ppl:.3f}")
                    log_eval_metrics(
                        used_tokens=used_tokens,
                        val_loss=v_loss,
                        val_perplexity=v_ppl,
                    )
                    last_eval_tokens = used_tokens

            if cfg.debug_memory and torch.cuda.is_available() and (idx % 100 == 0):
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                    peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
                    print(
                        f"peak MB after step: alloc={peak_alloc:.2f}, reserved={peak_reserv:.2f}"
                    )
                except Exception:
                    pass

            if used_tokens >= cfg.max_train_tokens:
                stop_training = True
                break

    return stats
