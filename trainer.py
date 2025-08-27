import torch.nn as nn
import torch.nn.functional as F
import torch
import plotly.express as px
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.amp import autocast
from transformers import is_av_available
from torch.utils.checkpoint import checkpoint as ckpt

from optimizers import AdaMuon, Muon
from model import TransformerNetwork

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
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader):
        
        self.cfg = cfg
        self.max_train_tokens: int = cfg.max_train_tokens
        self.used_train_tokens: int = 0

        self.max_val_tokens: int = cfg.max_val_tokens
        self.used_val_tokens: int = 0
        self.model: nn.Module = model

        self.train_loader = train_loader
        self.val_loader = val_loader 
        self.model.to(self.cfg.device)

        self.loss_fn = nn.CrossEntropyLoss()
        self.stats = {
            'train_loss': [],
            'val_loss': []
        }

        # Setup optimizers
        matrix_params = []
        other_params = []
        for name, p in model.named_parameters():
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

        if torch.cuda.is_av_available():
            torch.cuda.reset_peak_memory_stats()

        micro_batch = self.cfg.batch_size
        steps_in_accum = 0

        self.adam_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()
        
        while self.used_train_tokens < self.max_train_tokens:
            for idx, batch in enumerate(self.train_loader):
                input_ids = batch['input_ids'].to(self.cfg.device)
                
                with autocast(dtype=torch.bfloat16, device_type='cuda'):
                    logits = self.model(input_ids)
                    targets = input_ids[:, 1:]
                    logits = logits[:, :-1, :]
                    
                    # Calculate loss
                    loss = self.loss_fn(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1)) / accum_steps
                
                if self.cfg.debug_memory and torch.cuda.is_available():
                    print("memory stats before step")
                    self.debug_memory()

                loss.backward()
                step_in_accum += 1

                self.used_train_tokens += targets.numel()

                if step_in_accum == accum_steps:
                    self.admuon_optimizer.step()
                    self.adam_optimizer.step()
                    self.admuon_optimizer.zero_grad()
                    self.adam_optimizer.zero_grad()
                    step_in_accum = 0

                if self.cfg.debug_memory and torch.cuda.is_available():
                    print("Memory stats after step")
                    self.debug_memory()
                

                if idx % 10 == 0:
                    print(f"Train loss: {loss.item():.4f}, Tokens: {self.used_train_tokens}/{self.max_train_tokens}")

                self.stats['train_loss'].append(loss.item())


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