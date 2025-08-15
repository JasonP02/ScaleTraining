
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataload import load_tiny_stories


@dataclass
class Config():
    batch_size: int = 256

    n_head: int = 4
    n_embed: int = 256
    bias: bool = True
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2

cfg = Config()

train_loader, val_loader = load_tiny_stories(cfg)

# Flash attention block
class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0 # Ensure that embedding can be evenly split between heads
        self.kqv_block = nn.Linear(cfg.n_embed, cfg.n_embed * 3, bias=cfg.bias) # 3E, E
        self.out_projection = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias) # E, E

        self.attn_dropout = nn.Dropout(cfg.attn_dropout)
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)

        self.n_head = cfg.n_head
        self.n_embed = cfg.n_embed
        self.resid_dropout = cfg.resid_dropout
        self.attn_dropout = cfg.attn_dropout

        self.flash = F.scaled_dot_product_attention

    def forward(self, x):
        B, T, E = x.shape

        # x (BTE) ; W.T (E, 3E)
        # x @ W.T -> (B, T, 3E)
        # Then, we split along the embed dimension to get the matrices
        q, k, v = self.kqv_block(x).split(self.n_embed, dim=2) 

        # Reshape kqv to expected flash attention shapes
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1,2) # (B, nh, T, hs)




