
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataload import load_tiny_stories
from dataclasses import dataclass
from torch.utils.data import DataLoader


# Flash attention block
class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        # TODO: Add Rope

        super().__init__()
        assert cfg.n_embed % cfg.n_head == 0 # Ensure that embedding can be evenly split between heads
        self.kqv_block = nn.Linear(cfg.n_embed, cfg.n_embed * 3, bias=cfg.bias) # 3E, E
        self.out_projection = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias) # E, E

        self.n_head = cfg.n_head
        self.n_embed = cfg.n_embed
        self.resid_dropout = nn.Dropout(cfg.resid_dropout)
        self.attn_dropout = cfg.attn_dropout

        self.head_dim = cfg.n_embed // cfg.n_head
        self.max_seq_len = cfg.max_seq_len

        self.create_rope_lookup()

    def create_rope_lookup(self):
        """
        A function for creating lookup table for RoPE
        Takes in a max sequence length, and returns two torch tensors:
        cos_freqs, sin_freqs
        Shape: (max_sequence_length, frequency resolution) -> (1000,10000)
        """
        positions = torch.arange(0, self.max_seq_len-1 ,1)
        inv_freq = 1.0 / (10000 ** (torch.arange(0,self.head_dim,2).float() / self.head_dim))
        frequencies = torch.outer(positions, inv_freq)

        self.cos_freqs = torch.cos(frequencies)
        self.sin_freqs = torch.sin(frequencies)

    def _apply_rope(self, q, k):
        B, N, T, H = q.shape

        # Paper math:
        # For a 2x2 block 
        # [cosm, -sinm]
        # [sinm, cosm]
        cos_freqs = self.cos_freqs[T,:]
        sin_freqs = self.sin_freqs[T,:]

        # We do: 
        # [cosm * q1 - sinm * q2]
        # [sinm * q1 + cosm * q2]

        # Next step: determine how to split q into q1 and q2... I think we use .view
        q_even = q[...,::2]
        q_odd = q[...,1::2]
        k_even = k[...,::2]
        k_odd = k[...,1::2]

        # Now, we just perform the operations for each
        # Remember that frequencies are shape (T, freq_resolution)
        q_rot_even = q_even * cos_freqs - q_odd * sin_freqs
        q_rot_odd = q_even * sin_freqs + q_odd * cos_freqs

        k_rot_even = k_even * cos_freqs - k_odd * sin_freqs
        k_rot_odd = k_even * sin_freqs + k_odd * cos_freqs

        q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).reshape(B,N,T,H)
        v_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).reshape(B,N,T,H)

        return q_rot, v_rot

        
    def forward(self, x):
        B, T, E = x.shape

        # x (BTE) ; W.T (E, 3E)
        # x @ W.T -> (B, T, 3E)
        # Then, we split along the embed dimension to get the matrices
        q, k, v = self.kqv_block(x).split(self.n_embed, dim=2) 

        # Reshape kqv to expected flash attention shapes
        q = q.view(B, T, self.n_head, E // self.n_head).transpose(1,2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, E // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, E // self.n_head).transpose(1,2) # (B, nh, T, hs)

        q,k = self._apply_rope(q,k)

        # SDPA takes in tensors, with dropout for attention scores
        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None, dropout_p=self.attn_dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,E)
        return self.resid_dropout(self.out_projection(y))


class MLPBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wh = nn.Linear(cfg.n_embed, cfg.n_hidden, bias=cfg.bias, device=cfg.device)
        self.We = nn.Linear(cfg.n_hidden, cfg.n_embed, bias=cfg.bias, device=cfg.device)
        self.dropout = nn.Dropout(cfg.resid_dropout)

    def forward(self,x):
        residual = x
        # x -> (B T E)
        x = self.Wh(x)
        x = F.relu(x)
        x = self.We(x)
        x = self.dropout(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embed)
        self.attention = AttentionBlock(cfg)
        self.mlp = MLPBlock(cfg)
    
    def forward(self, x):
        x = x + self.attention(self.ln(x))
        x = x + self.mlp(self.ln(x))
        return x
    
class TransformerNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # We need to create: embedding matrix, stacked transformer block, out logits
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.W_e = nn.Linear(cfg.vocab_size, cfg.n_embed)
        self.W_ue = nn.Linear(cfg.n_embed, cfg.vocab_size)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg, ) for _ in range(cfg.n_layer)
        ])
        self.ln = nn.LayerNorm(cfg.n_embed)

    def forward(self, x):
        B,T = x.shape
        x = self.token_embedding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.ln(x)
        x = self.W_ue(x)
        return x

class AdaMuon(torch.optim.Optimizer):
    def __init__(self, params, lr, cfg):
        if lr < 0.0: raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr)
        super(AdaMuon, self).__init__(params, defaults)

        for group in self.param_groups:
            for param in group['params']:
                self.state[param]['step'] = 0 # Initalize step
                self.state[param]['momentum'] = torch.zeros_like(param) # Momentum is initially zero

        
        self.AS_iters = 5 # Number of newton-shulz iterations. 5 is an emperical value
        self.momentum_coef = cfg.momentum_coef

    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                grad = param.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()

                self.state[param]['momentum'] = self.momentum_coef * self.state[param]['momentum'] + grad

                NS = self._newton_shulz(self.state[param]['momentum']) # Apply N-S for 5 iterations for weight update
                update = group['lr'] * (0.2 * NS * torch.sqrt(torch.max(NS.shape)) + group['weight_decay']*param)
                param.sub_(update)   

                self.state[param]['step'] += 1

        return loss


    def _newton_shulz(self, momentum):
        X_prev = momentum / (momentum.norm() + 1e-8)
        a = 3.4445
        b = -4.7750
        c = 2.0315
        

        for i in range(self.AS_iters):
            X_prev = a * X_prev + b * (X_prev @ X_prev.T) @ X_prev + c * (X_prev @ X_prev.T)**2 @ X_prev

        return X_prev
             




@dataclass
class Config():
    device = 'cuda' if torch.cuda.is_available else 'cpu'
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

admuon = AdaMuon(params=matrix_params, lr=lr, cfg=cfg)

class LLMTrainer:
    def __init__(self,
                 cfg,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader):
        
        self.max_train_tokens: int = cfg.max_train_tokens
        self.used_train_tokens: int = 0

        self.max_val_tokens: int = cfg.max_val_tokens
        self.used_val_tokens: int = 0
        self.model: nn.Module = model

        self.train_loader = train_loader
        self.val_loader = val_loader 

        # self.optimizer = 

    def train(self):
        while self.used_train_tokens < self.max_train_tokens and batch_is_finished:
            batch_is_finished = False
            for batch, idx in range(1):
                pass

            batch_is_finished = True