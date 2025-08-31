"""
Functional training loop utilities.

These helpers implement the training loop without an object‑oriented trainer.
Each function has a narrow purpose and explicit inputs/outputs.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import contextlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

from scaletraining.model.optimizers import AdaMuon, Muon


def split_model_matrix_params(named_params: Iterable[Tuple[str, nn.Parameter]]) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """Split parameters into matrix (2D) vs other, deduplicated by identity.

    Args:
        named_params: iterable of (name, parameter) from `model.named_parameters()`.
    Returns:
        (matrix_params, other_params) lists of torch.nn.Parameter.
    """
    seen, matrix_params, other_params = set(), [], []
    for _, p in named_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        if getattr(p, 'ndim', None) == 2 or p.dim() == 2:
            matrix_params.append(p)
        else:
            other_params.append(p)
    return matrix_params, other_params


def build_optimizers(cfg, matrix_params: List[nn.Parameter], other_params: List[nn.Parameter]):
    """Construct primary and secondary optimizers based on cfg.

    Args:
        cfg: Hydra config object with optimizer fields (primary_optimizer, lr, beta, beta2, weight_decay, ns_iters, eps).
        matrix_params: list of 2D parameters.
        other_params: list of non‑2D parameters.
    Returns:
        (primary_optimizer, secondary_optimizer) torch.optim.Optimizer instances.
        Secondary optimizer is None when using baseline Adam or AdamW.
    """
    if getattr(cfg, 'use_baseline_adam', False):
        # Use single Adam optimizer for all parameters
        all_params = list(matrix_params) + list(other_params)
        baseline_cfg = getattr(cfg, 'baseline_adam_config', {})
        lr = baseline_cfg.get('lr', cfg.lr)
        weight_decay = baseline_cfg.get('weight_decay', cfg.weight_decay)
        betas = baseline_cfg.get('betas', (cfg.beta, cfg.beta2))
        
        optimizer = torch.optim.AdamW(
            params=all_params, 
            lr=lr, 
            weight_decay=weight_decay, 
            betas=betas,
            eps=cfg.eps
        )
        return optimizer, None  # No secondary optimizer needed
    
    # Check if using AdamW as primary optimizer
    name = getattr(cfg, 'primary_optimizer', 'adamuon').lower()
    if name == 'adamw':
        # Use single AdamW optimizer for all parameters (like baseline Adam)
        all_params = list(matrix_params) + list(other_params)
        betas = (cfg.beta, cfg.beta2)
        
        optimizer = torch.optim.AdamW(
            params=all_params, 
            lr=cfg.lr, 
            betas=betas,
            weight_decay=cfg.weight_decay, 
            eps=cfg.eps
        )
        return optimizer, None  # No secondary optimizer needed
    
    # Custom optimizers (adamuon/muon) - use parameter splitting
    betas = (cfg.beta, cfg.beta2)
    if name == 'muon':
        primary = Muon(params=matrix_params, lr=cfg.lr, beta=betas[0], beta2=betas[1],
                       weight_decay=cfg.weight_decay, ns_iters=cfg.ns_iters, eps=cfg.eps)
    elif name == 'adamuon':
        primary = AdaMuon(params=matrix_params, lr=cfg.lr, beta=betas[0], beta2=betas[1],
                          weight_decay=cfg.weight_decay, ns_iters=cfg.ns_iters, eps=cfg.eps)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {cfg.primary_optimizer}")

    secondary = torch.optim.AdamW(params=other_params, lr=cfg.lr, betas=betas,
                                  weight_decay=cfg.weight_decay, eps=cfg.eps)
    return primary, secondary


def prepare_targets(input_ids: torch.Tensor, attn_mask: torch.Tensor | None) -> Tuple[torch.Tensor, int]:
    """Create next‑token targets and compute effective token count.

    Args:
        input_ids: LongTensor shape [B, T], token ids.
        attn_mask: Optional mask [B, T] with 1 for valid and 0 for padding.
    Returns:
        (targets, total_effective_tokens) where targets is [B, T-1] with -100 ignored indices.
    """
    targets = input_ids[:, 1:]
    if attn_mask is not None:
        target_mask = attn_mask[:, 1:]
        ignore = (target_mask == 0)
        targets = targets.clone()
        targets[ignore] = -100
        total_effective = targets.ne(-100).sum().item()
    else:
        total_effective = targets.numel()
    return targets, int(total_effective)


def compute_loss_sum(model, hidden: torch.Tensor, targets: torch.Tensor, chunk_size: int, loss_fn: nn.Module) -> torch.Tensor:
    """Compute summed CE loss across time, optionally in chunks to save memory.

    Args:
        model: model with `W_ue` projection (vocab head).
        hidden: Tensor [B, T, E], pre‑logits states.
        targets: LongTensor [B, T], next‑token targets with -100 for ignored.
        chunk_size: int, max time length per chunk; 0 disables chunking.
        loss_fn: loss module with reduction='sum'.
    Returns:
        Scalar tensor, total summed cross‑entropy over all non‑ignored tokens.
    """
    T = hidden.size(1)
    if chunk_size <= 0 or chunk_size >= T:
        logits = model.W_ue(hidden)
        return loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    loss_sum = hidden.new_tensor(0.0, dtype=torch.float32)
    start = 0
    while start < T:
        end = min(start + chunk_size, T)
        logits_chunk = model.W_ue(hidden[:, start:end, :])
        targets_chunk = targets[:, start:end]
        loss_sum = loss_sum + loss_fn(
            logits_chunk.reshape(-1, logits_chunk.size(-1)),
            targets_chunk.reshape(-1)
        ).to(loss_sum.dtype)
        start = end
    return loss_sum


def training_run(cfg, model, train_loader: DataLoader, *, loss_fn: nn.Module) -> Dict[str, list]:
    """Functional training loop until reaching token budget.

    Args:
        cfg: Hydra config with fields used: device, accum_steps, grad_clip_norm,
             logits_chunk_size, max_train_tokens, debug_memory.
        model: nn.Module with `forward_hidden` and `W_ue` attributes.
        train_loader: DataLoader yielding dicts with 'input_ids' and optional 'attention_mask'.
        loss_fn: nn.CrossEntropyLoss(reduction='sum') for per‑token normalization.
    Returns:
        stats: dict with key 'train_loss' (list of averaged per‑token losses per accumulation window).
    """
    import wandb

    matrix_params, other_params = split_model_matrix_params(model.named_parameters())
    opt_primary, opt_secondary = build_optimizers(cfg, matrix_params, other_params)

    model.to(cfg.device)
    model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stats = {"train_loss": []}
    used_tokens = 0
    step_in_accum = 0
    accum_loss_sum = 0.0
    accum_token_count = 0

    stop_training = False
    while used_tokens < cfg.max_train_tokens and not stop_training:
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(cfg.device)
            attn_mask = batch.get('attention_mask', None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(cfg.device)

            ctx = autocast(device_type='cuda', dtype=torch.bfloat16) if (cfg.device == 'cuda' and torch.cuda.is_available()) else contextlib.nullcontext()
            with ctx:
                hidden = model.forward_hidden(input_ids)
                hidden = hidden[:, :-1, :]
                targets, effective = prepare_targets(input_ids, attn_mask)
                loss_sum = compute_loss_sum(model, hidden, targets, getattr(cfg, 'logits_chunk_size', 0), loss_fn)
                per_token_loss = loss_sum / max(1, effective)
                loss = per_token_loss / cfg.accum_steps

            loss.backward()
            accum_loss_sum += float(loss_sum.item())
            accum_token_count += int(effective)
            step_in_accum += 1

            used_tokens += int(effective)

            if step_in_accum == cfg.accum_steps:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                opt_primary.step()
                if opt_secondary is not None:
                    opt_secondary.step()
                
                opt_primary.zero_grad(set_to_none=True)
                if opt_secondary is not None:
                    opt_secondary.zero_grad(set_to_none=True)
                step_in_accum = 0

                avg_loss = accum_loss_sum / max(1, accum_token_count)
                stats['train_loss'].append(avg_loss)
                print(f"Tokens: {used_tokens:,}, Loss: {avg_loss:.4f}")
                try:
                    wandb.log({'used tokens': used_tokens, 'train_per_token_loss': avg_loss}, step=used_tokens)
                except Exception:
                    pass
                accum_loss_sum = 0.0
                accum_token_count = 0

            if cfg.debug_memory and torch.cuda.is_available() and (idx % 100 == 0):
                try:
                    peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
                    peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
                    print(f"peak MB after step: alloc={peak_alloc:.2f}, reserved={peak_reserv:.2f}")
                except Exception:
                    pass

            if used_tokens >= cfg.max_train_tokens:
                stop_training = True
                break

    return stats

