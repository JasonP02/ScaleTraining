import contextlib
import torch.nn as nn
import torch.nn.functional as F
import torch
import plotly.express as px
from torch.utils.data import DataLoader
from torch.amp import autocast

from scaletraining.model.model import TransformerNetwork
from scaletraining.data_processing import build_loaders
from scaletraining.config import Config

from scaletraining.model.optimizers import AdaMuon, Muon
import wandb
from transformers import AutoTokenizer

def main():
    cfg = Config()

    max_train_tokens = cfg.max_train_tokens
    used_train_tokens = 0

    model = TransformerNetwork(cfg)

    train_loader , _ = build_loaders(cfg)
    model = model.to(cfg.device)

    loss_fn = nn.CrossEntropyLoss(reduction='sum') # why sum

    stats = {
        'train_loss': [],
        'val_loss': []
    }

    matrix_params, other_params = split_model_matrix_params(model.named_parameters)

    primary_optimizer, secondary_optimizer = get_primary_and_secondary_optimizer(
        matrix_params=matrix_params,
        other_params=other_params,
        optim_name=cfg.primary_optimizer,
        lr=cfg.lr,
        betas=(cfg.beta, cfg.beta2),
        weight_decay=cfg.weight_decay,
        eps=cfg.eps,
        ns_iters=cfg.ns_iters)

    wandb.init(project=cfg.wandb_project_name, entity='thajpo')

    model.train()
    accum_steps = cfg.accum_steps

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    device = cfg.device

    step_in_accum = 0
    accum_loss_sum = 0.0
    accum_token_count = 0

    stop_training = False
    while used_train_tokens < max_train_tokens and not stop_training:
        for idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch.get('attention_mask', None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            # Safe autocast: use CUDA bf16 if on CUDA, else no-op context
            ctx = autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else contextlib.nullcontext()

            with ctx:
                # Compute hidden states once
                hidden = model.forward_hidden(input_ids)
                targets = input_ids[:, 1:]
                hidden = hidden[:, :-1, :]

                if attn_mask is not None:
                    target_mask = attn_mask[:, 1:]
                    ignore = (target_mask == 0)
                    targets = targets.clone()
                    targets[ignore] = -100

                # Count effective tokens for normalization
                total_effective_tokens = targets.ne(-100).sum() if attn_mask is not None else targets.numel()

                # Chunked logits/loss along time to reduce peak memory
                T = hidden.size(1)
                chunk_size = cfg.logits_chunk_size
                if chunk_size <= 0 or chunk_size >= T:
                    logits = model.W_ue(hidden)
                    loss_sum = loss_fn(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                else:
                    loss_sum = 0.0
                    start = 0
                    while start < T:
                        end = min(start + chunk_size, T)
                        logits_chunk = model.W_ue(hidden[:, start:end, :])
                        targets_chunk = targets[:, start:end]
                        loss_sum = loss_sum + loss_fn(
                            logits_chunk.reshape(-1, logits_chunk.size(-1)),
                            targets_chunk.reshape(-1)
                        )
                        start = end

                # Compute per-token loss for logging; scale for backward only
                per_token_loss = loss_sum / max(1, total_effective_tokens)
                loss = per_token_loss / accum_steps
            
            # if self.cfg.debug_memory and (idx % 25 == 0):
            #     print(f"dtypes: logits={logits.dtype}, loss={loss.dtype}, bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

            loss.backward()
            accum_loss_sum += float(loss_sum.item())
            accum_token_count += int(total_effective_tokens)
            step_in_accum += 1

            num_tokens = targets.ne(-100).sum().item() if attn_mask is not None else targets.numel() 
            used_train_tokens += num_tokens

            if step_in_accum == accum_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)

                primary_optimizer.step()
                secondary_optimizer.step()
                primary_optimizer.zero_grad(set_to_none=True)
                secondary_optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0
                # Log averaged per-token loss over the accumulation window
                avg_per_token_loss = accum_loss_sum / max(1, accum_token_count)
                stats['train_loss'].append(avg_per_token_loss)
                try:
                    wandb.log({'used tokens': used_train_tokens, 'train_per_token_loss': avg_per_token_loss})
                except Exception:
                    pass
                accum_loss_sum = 0.0
                accum_token_count = 0

            if cfg.debug_memory and torch.cuda.is_available() and (idx % 100 == 0):
                print("Memory stats after step")
                debug_memory()

            if used_train_tokens % 100 == 0 and len(stats['train_loss']) > 0:
                try:
                    wandb.log({'used tokens': used_train_tokens, 'train_per_token_loss': stats['train_loss'][-1]})
                except Exception:
                    pass
            

            if idx % 10 == 0:
                try:
                    current = stats['train_loss'][-1]
                except IndexError:
                    current = float(per_token_loss.item())
                print(f"Train per-token loss: {current:.4f}  Tokens: {used_train_tokens}/{max_train_tokens}")

            # Early stop within the epoch once token budget is reached
            if used_train_tokens >= max_train_tokens:
                stop_training = True
                break

def debug_memory():
    try:
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
        print(f"peak MB after step: alloc={peak_alloc:.2f}, reserved={peak_reserv:.2f}")
    except Exception as e:
        print(f"peak mem debug skipped: {e}")

def split_model_matrix_params(model_named_params):
    seen, matrix_params, other_params = set(), [], []

    for _, p in model_named_params():
        if id(p) in seen:
            continue
        seen.add(id(p))
        (matrix_params if (getattr(p, 'ndim', None) == 2 or p.dim() == 2) else other_params).append(p)
    return matrix_params, other_params

def get_primary_and_secondary_optimizer(
        matrix_params,
        other_params,
        optim_name,
        lr,
        betas,
        weight_decay,
        eps,
        ns_iters
    ):

    if optim_name == 'muon':
        optimizer = Muon(
            params=matrix_params,
            lr=lr,
            beta=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            eps=eps
        )
    elif optim_name == 'adamuon':
        optimizer = AdaMuon(
            params=matrix_params,
            lr=lr,
            beta=betas[0],
            beta2=betas[1],
            weight_decay=weight_decay,
            ns_iters=ns_iters,
            eps=eps
        )
    elif optim_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params=matrix_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )

    else:
        raise NotImplementedError("Optimizer not supported")

    secondary_optimizer = torch.optim.AdamW(
            params=other_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps
        )

    return optimizer, secondary_optimizer

if __name__ == "__main__":
    main()
