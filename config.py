from dataclasses import dataclass
import torch

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
