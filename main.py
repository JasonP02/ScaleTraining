
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataload import load_tiny_stories
from dataclasses import dataclass
from torch.utils.data import DataLoader

from optimizers import AdaMuon, Muon
from model import TransformerNetwork



@dataclass
class Config():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 256
    vocab_size: int = 16000
    n_layer: int = 1
    max_seq_len: int = 1000

    n_head: int = 4
    n_embed: int = 256
    bias: bool = True
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2

    n_hidden: int = 256*4
    momentum_coef: float = 0.99
    lr: float = 1e-4
    max_train_tokens: int = 100000
    max_val_tokens: int = 100000
    
    # Optimizer parameters
    beta: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    ns_iters: int = 5
    eps: float = 1e-8

cfg = Config()
lr = cfg.lr

model = TransformerNetwork(cfg)

matrix_params = []
other_params = []

for name, p in model.named_parameters():
    if p.ndim == 2:     # example predicate
        matrix_params.append(p)
    else:
        other_params.append(p)

admuon = AdaMuon(params=matrix_params, lr=lr)

class LLMTrainer:
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

        # Setup optimizers
        matrix_params = []
        other_params = []
        for name, p in model.named_parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                other_params.append(p)
        
        self.admuon = AdaMuon(
            params=matrix_params, 
            lr=cfg.lr,
            beta=cfg.beta,
            beta2=cfg.beta2,
            weight_decay=cfg.weight_decay,
            ns_iters=cfg.ns_iters,
            eps=cfg.eps
        )
        self.other_optimizer = torch.optim.AdamW(
            other_params, 
            lr=cfg.lr,
            betas=(cfg.beta, cfg.beta2),
            weight_decay=cfg.weight_decay,
            eps=cfg.eps
        ) 

    def train(self):
        self.model.train()
        self.model.to(self.cfg.device)
        
        while self.used_train_tokens < self.max_train_tokens:
            for batch in self.train_loader:
                if self.used_train_tokens >= self.max_train_tokens:
                    break
                    
                # Get batch data
                texts = batch['text']
                
                # Simple tokenization (placeholder - you'll need proper tokenizer)
                # For now, let's use character-level tokenization as placeholder
                batch_tokens = []
                for text in texts:
                    tokens = [ord(c) % 1000 for c in text[:100]]  # Simple char encoding
                    batch_tokens.append(tokens)
                
                # Pad sequences to same length
                max_len = max(len(tokens) for tokens in batch_tokens)
                input_ids = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
                for i, tokens in enumerate(batch_tokens):
                    input_ids[i, :len(tokens)] = torch.tensor(tokens)
                
                input_ids = input_ids.to(self.cfg.device)
                
                # Forward pass
                logits = self.model(input_ids)
                
                # Create targets (shifted input_ids)
                targets = input_ids[:, 1:]
                logits = logits[:, :-1, :]
                
                # Calculate loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                # Backward pass
                self.admuon.zero_grad()
                self.other_optimizer.zero_grad()
                loss.backward()
                self.admuon.step()
                self.other_optimizer.step()
                
                # Update token count
                self.used_train_tokens += targets.numel()
                
                print(f"Train loss: {loss.item():.4f}, Tokens: {self.used_train_tokens}/{self.max_train_tokens}")

    def eval(self):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if self.used_val_tokens >= self.max_val_tokens:
                    break
                    
                # Get batch data
                texts = batch['text']
                
                # Simple tokenization (placeholder)
                batch_tokens = []
                for text in texts:
                    tokens = [ord(c) % 1000 for c in text[:100]]
                    batch_tokens.append(tokens)
                
                # Pad sequences
                max_len = max(len(tokens) for tokens in batch_tokens)
                input_ids = torch.zeros((len(batch_tokens), max_len), dtype=torch.long)
                for i, tokens in enumerate(batch_tokens):
                    input_ids[i, :len(tokens)] = torch.tensor(tokens)
                
                input_ids = input_ids.to(self.cfg.device)
                
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
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss