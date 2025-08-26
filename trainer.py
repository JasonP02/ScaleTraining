import torch.nn as nn
import torch.nn.functional as F
import torch
import plotly.express as px
from dataclasses import dataclass
from torch.utils.data import DataLoader

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

    def training_run(self):
        self.model.train()
        
        while self.used_train_tokens < self.max_train_tokens:
            for batch in self.train_loader:
                print(batch)
                # Get batch data - assuming pre-tokenized data
                input_ids = batch['input_ids'].to(self.cfg.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Create targets (shifted input_ids)
                targets = input_ids[:, 1:]
                logits = logits[:, :-1, :]
                
                # Calculate loss
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                # Backward pass
                self.admuon_optimizer.zero_grad()
                self.adam_optimizer.zero_grad()
                loss.backward()
                self.admuon_optimizer.step()
                self.adam_optimizer.step()
                
                # Update token count
                self.used_train_tokens += targets.numel()
                
                self.stats['train_loss'].append(loss.item())
                print(f"Train loss: {loss.item():.4f}, Tokens: {self.used_train_tokens}/{self.max_train_tokens}")

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