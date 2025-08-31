import torch.nn as nn
import torch.nn.functional as F
import torch
import plotly.express as px
from torch.utils.data import DataLoader
from torch.amp import autocast

from scaletraining.model.model import TransformerNetwork

from scaletraining.model.optimizers import AdaMuon, Muon
import wandb
from transformers import AutoTokenizer

class LLMTrainer:
    """
    Trainer class for the LLM
    Args:
        cfg: Config object
        model: Model to train
        train_loader: DataLoader object for the train dataset
        val_loader: DataLoader object for the validation dataset
    """
    def __init__(self,
                 cfg,
                 train_loader: DataLoader,
                 val_loader: DataLoader):
        
        self.cfg = cfg
        self.max_train_tokens: int = cfg.max_train_tokens
        self.used_train_tokens: int = 0

        self.max_val_tokens: int = cfg.max_val_tokens
        self.used_val_tokens: int = 0
        # Build model internally and move to device
        self.model: nn.Module = TransformerNetwork(cfg)

        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.model.to(self.cfg.device)

        # Use summed CE so we can normalize across chunks by total tokens
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.stats = {
            'train_loss': [],
            'val_loss': []
        }

        # Setup optimizers
        matrix_params = []
        other_params = []
        for name, p in self.model.named_parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                other_params.append(p)
        
        self.admuon_optimizer = AdaMuon(
            params=matrix_params, 
            lr=cfg.lr,
            beta=cfg.beta,
            beta2=cfg.beta2,
            weight_decay=cfg.weight_decay,
            ns_iters=cfg.ns_iters,
            eps=cfg.eps
        )
        self.adam_optimizer = torch.optim.AdamW(
            other_params, 
            lr=cfg.lr,
            betas=(cfg.beta, cfg.beta2),
            weight_decay=cfg.weight_decay,
            eps=cfg.eps
        ) 

        wandb.init(project=cfg.wandb_project_name, entity='thajpo')

        # Lightweight tokenizer for eval/generation
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name, use_fast=True)
            if self._tokenizer.eos_token_id is None:
                self._tokenizer.add_special_tokens({"eos_token": ""})
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        except Exception:
            self._tokenizer = None

    def debug_memory(self):
        try:
            peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserv = torch.cuda.max_memory_reserved() / (1024**2)
            print(f"peak MB after step: alloc={peak_alloc:.2f}, reserved={peak_reserv:.2f}")
        except Exception as e:
            print(f"peak mem debug skipped: {e}")

    def training_run(self):
        self.model.train()
        accum_steps = self.cfg.accum_steps

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        step_in_accum = 0
        total_loss = 0

        stop_training = False
        while self.used_train_tokens < self.max_train_tokens and not stop_training:
            for idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.cfg.device)
                attn_mask = batch.get('attention_mask', None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.cfg.device)
                
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    # Compute hidden states once
                    hidden = self.model.forward_hidden(input_ids)
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
                    chunk_size = getattr(self.cfg, 'logits_chunk_size', 0) or 0
                    if chunk_size <= 0 or chunk_size >= T:
                        logits = self.model.W_ue(hidden)
                        loss_sum = self.loss_fn(
                            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    else:
                        loss_sum = 0.0
                        start = 0
                        while start < T:
                            end = min(start + chunk_size, T)
                            logits_chunk = self.model.W_ue(hidden[:, start:end, :])
                            targets_chunk = targets[:, start:end]
                            loss_sum = loss_sum + self.loss_fn(
                                logits_chunk.reshape(-1, logits_chunk.size(-1)),
                                targets_chunk.reshape(-1)
                            )
                            start = end

                    # Normalize by number of tokens and accumulation
                    loss = (loss_sum / max(1, total_effective_tokens)) / accum_steps
                
                # if self.cfg.debug_memory and (idx % 25 == 0):
                #     print(f"dtypes: logits={logits.dtype}, loss={loss.dtype}, bf16_supported={torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

                loss.backward()
                total_loss += loss.item() * accum_steps
                step_in_accum += 1

                num_tokens = targets.ne(-100).sum().item() if attn_mask is not None else targets.numel() 
                self.used_train_tokens += num_tokens

                if step_in_accum == accum_steps:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip_norm)

                    self.admuon_optimizer.step()
                    self.adam_optimizer.step()
                    self.admuon_optimizer.zero_grad(set_to_none=True)
                    self.adam_optimizer.zero_grad(set_to_none=True)
                    step_in_accum = 0
                    total_loss = 0
                    self.stats['train_loss'].append(loss.item() / accum_steps)

                if self.cfg.debug_memory and torch.cuda.is_available() and (idx % 100 == 0):
                    print("Memory stats after step")
                    self.debug_memory()

                if self.used_train_tokens % 100 == 0 and len(self.stats['train_loss']) > 0:
                    wandb.log({'used tokens': self.used_train_tokens, "loss": self.stats['train_loss'][-1]})
                

                if idx % 10 == 0:
                    print(f"Train loss: {loss.item():.4f}, Tokens: {self.used_train_tokens}/{self.max_train_tokens}")

                # Early stop within the epoch once token budget is reached
                if self.used_train_tokens >= self.max_train_tokens:
                    stop_training = True
                    break

    @torch.no_grad()
    def generate_sample_story(self,
                               prompt: str = "Once upon a time",
                               max_new_tokens: int = 100,
                               temperature: float = 1.0,
                               top_k: int = 50) -> str:
        """
        Simple autoregressive generation for a short story sample.
        """
        self.model.eval()
        if self._tokenizer is None:
            print("Tokenizer unavailable; skipping generation.")
            return ""

        input_ids = self._tokenizer.encode(prompt, return_tensors='pt').to(self.cfg.device)
        for _ in range(max_new_tokens):
            with autocast(dtype=torch.bfloat16, device_type='cuda'):
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / max(1e-6, temperature)

                if top_k is not None and top_k > 0 and top_k < next_token_logits.size(-1):
                    topk_vals, _ = torch.topk(next_token_logits, top_k)
                    min_topk = topk_vals[:, -1].unsqueeze(-1)
                    next_token_logits = torch.where(next_token_logits < min_topk, torch.full_like(next_token_logits, float('-inf')), next_token_logits)

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on EOS if defined
            if self._tokenizer.eos_token_id is not None and next_token.item() == self._tokenizer.eos_token_id:
                break

        text = self._tokenizer.decode(input_ids[0], skip_special_tokens=True)
        print("\n=== Generated Story Sample ===\n" + text + "\n==============================\n")
        try:
            wandb.log({"generated_story": text})
        except Exception:
            pass
        return text



    def eval(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.used_val_tokens >= self.max_val_tokens:
                    break
                    
                # Get batch data - assuming pre-tokenized data
                input_ids = batch['input_ids'].to(self.cfg.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Create targets
                targets = input_ids[:, 1:]
                logits = logits[:, :-1, :]
                
                # Calculate loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                total_loss += loss.item() * targets.numel()
                total_tokens += targets.numel()
                self.used_val_tokens += targets.numel()
        
        avg_loss = total_loss / total_tokens
        self.stats['val_loss'].append(avg_loss)
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def plot_stats(self, save_path: str = 'stats.png'):
        """
        Plot the training and validation loss and save it to the disk
        Args:
            save_path: Path to save the plot
        """
        fig = px.line(self.stats, x=list(self.stats.keys()), y=list(self.stats.values()), title='Training and Validation Loss')
        fig.write_image(save_path)