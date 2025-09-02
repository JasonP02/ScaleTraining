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
    """Split parameters into Muon‑eligible hidden matrices vs everything else.

    Muon should only optimize hidden weight matrices. We explicitly exclude:
    - input embedding matrix (e.g., `token_embedding.weight`)
    - final output projection (e.g., `W_ue.*`)
    - all biases/gains (non‑2D tensors)

    Deduplication by identity ensures tied weights (embedding <-> head) are placed once.

    Args:
        named_params: iterable of (name, parameter) from `model.named_parameters()`.
    Returns:
        (matrix_params, other_params)
    """
    def excluded_from_muon(name: str, p: nn.Parameter) -> bool:
        # Exclude embeddings and the tied output head; names are stable in our model
        if name.startswith('token_embedding'):
            return True
        if name.startswith('W_ue'):
            return True
        # Non‑2D tensors are not Muon targets
        if p.dim() != 2:
            return True
        return False

    seen, muon_mats, other_params = set(), [], []
    for name, p in named_params:
        if id(p) in seen:
            continue
        seen.add(id(p))
        if excluded_from_muon(name, p):
            other_params.append(p)
        else:
            muon_mats.append(p)
    return muon_mats, other_params


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
    if cfg.use_baseline_adam:
        # Use single Adam optimizer for all parameters
        all_params = list(matrix_params) + list(other_params)
        baseline_cfg = cfg.baseline_adam_config
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
    name = cfg.primary_optimizer.lower()
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
        muon_lr = cfg.muon_lr
        primary = Muon(params=matrix_params, lr=muon_lr, beta=betas[0], beta2=betas[1],
                       weight_decay=cfg.weight_decay, ns_iters=cfg.ns_iters, eps=cfg.eps)
    elif name == 'adamuon':
        muon_lr = cfg.muon_lr
        primary = AdaMuon(params=matrix_params, lr=muon_lr, beta=betas[0], beta2=betas[1],
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


def _compute_lr_scale_tokens(used_tokens: int, cfg) -> float:
    """Compute LR scale based on token progress with warmup and schedule.

    Schedules:
    - none: linear warmup to 1.0, then flat.
    - cosine: linear warmup, then cosine decay to min_lr_scale.
    - linear: linear warmup, then linear decay to min_lr_scale.
    """
    schedule = cfg.lr_schedule
    warmup_tokens = int(cfg.warmup_tokens or 0)
    min_scale = float(cfg.min_lr_scale)
    total = int(cfg.max_train_tokens or 0)
    if total <= 0:
        total = max(used_tokens, 1)

    # Warmup
    if warmup_tokens > 0 and used_tokens < warmup_tokens:
        return max(0.0, min(1.0, used_tokens / max(1, warmup_tokens)))

    # Post-warmup progress in [0,1]
    post = used_tokens - warmup_tokens
    denom = max(1, total - warmup_tokens)
    t = max(0.0, min(1.0, post / denom))

    if schedule == 'cosine':
        import math
        return float(min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * t)))
    elif schedule == 'linear':
        return float(min_scale + (1.0 - min_scale) * (1.0 - t))
    else:  # 'none'
        return 1.0


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
    # Log optimizer wiring if requested
    if cfg.log_implementation_details:
        def _summarize(ps):
            return [tuple(p.shape) for p in ps][:10]
        print("[opt-wiring] muon-eligible (hidden 2D) count:", len(matrix_params), "sample shapes:", _summarize(matrix_params))
        print("[opt-wiring] adamw params (embeddings, head, biases, etc.) count:", len(other_params), "sample shapes:", _summarize(other_params))
    opt_primary, opt_secondary = build_optimizers(cfg, matrix_params, other_params)

    # Capture base LRs for scheduling
    primary_base_lr = float(opt_primary.param_groups[0]['lr']) if opt_primary is not None else 0.0
    secondary_base_lr = float(opt_secondary.param_groups[0]['lr']) if opt_secondary is not None else 0.0

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
                loss_sum = compute_loss_sum(model, hidden, targets, cfg.logits_chunk_size, loss_fn)
                per_token_loss = loss_sum / max(1, effective)
                loss = per_token_loss / cfg.accum_steps

            loss.backward()
            accum_loss_sum += float(loss_sum.item())
            accum_token_count += int(effective)
            step_in_accum += 1

            used_tokens += int(effective)

            if step_in_accum == cfg.accum_steps:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
                # LR scheduling based on tokens
                lr_scale = _compute_lr_scale_tokens(used_tokens, cfg)
                # Apply scaled LRs
                if opt_primary is not None:
                    for g in opt_primary.param_groups:
                        g['lr'] = primary_base_lr * lr_scale
                if opt_secondary is not None:
                    for g in opt_secondary.param_groups:
                        g['lr'] = secondary_base_lr * lr_scale

                opt_primary.step()
                if opt_secondary is not None:
                    opt_secondary.step()
                
                opt_primary.zero_grad(set_to_none=True)
                if opt_secondary is not None:
                    opt_secondary.zero_grad(set_to_none=True)
                step_in_accum = 0

                avg_loss = accum_loss_sum / max(1, accum_token_count)
                stats['train_loss'].append(avg_loss)
                try:
                    current_lr = opt_primary.param_groups[0]['lr'] if opt_primary is not None else 0.0
                except Exception:
                    current_lr = 0.0
                print(f"Tokens: {used_tokens:,}, Loss: {avg_loss:.4f}, LR: {current_lr:.6g}")
                try:
                    wandb.log({'used tokens': used_tokens, 'train_per_token_loss': avg_loss, 'lr': current_lr}, step=used_tokens)
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
