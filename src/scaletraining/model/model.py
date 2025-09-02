import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt



class AttentionBlock(nn.Module):
    def __init__(self, cfg):
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

        # RoPE configuration (custom | torch_builtin | none)
        self.rope_implementation = cfg.rope_implementation
        self.theta = cfg.rope_config.get('theta', 10000)

        if self.rope_implementation != 'none':
            assert self.head_dim % 2 == 0, "Head dimension must be even for RoPE, but got %d" % self.head_dim
            self.create_rope_lookup()

    def create_rope_lookup(self):
        """
        A function for creating lookup table for RoPE
        Takes in a max sequence length, and returns two torch tensors:
        cos_freqs, sin_freqs
        Shape: (max_sequence_length, frequency resolution) -> (1000,10000)
        """
        positions = torch.arange(0, self.max_seq_len, 1)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0,self.head_dim,2).float() / self.head_dim))
        frequencies = torch.outer(positions, inv_freq)

        self.register_buffer('cos_freqs', torch.cos(frequencies), persistent=False)
        self.register_buffer('sin_freqs', torch.sin(frequencies), persistent=False)

    def _apply_rope(self, q, k, cos_freqs, sin_freqs):
        """Apply RoPE given precomputed cos/sin lookup tables."""
        B, N, T, H = q.shape
        cos = cos_freqs[:T, :].unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=q.dtype)
        sin = sin_freqs[:T, :].unsqueeze(0).unsqueeze(0).to(device=q.device, dtype=q.dtype)

        q_even, q_odd = q[..., ::2], q[..., 1::2]
        k_even, k_odd = k[..., ::2], k[..., 1::2]

        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos

        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos

        q_rot = torch.stack([q_rot_even, q_rot_odd], dim=-1).reshape(B, N, T, H).to(device=q.device, dtype=q.dtype)
        k_rot = torch.stack([k_rot_even, k_rot_odd], dim=-1).reshape(B, N, T, H).to(device=k.device, dtype=k.dtype)
        return q_rot, k_rot

    def _apply_rope_torch_builtin(self, q, k):
        """Prefer PyTorch built-in RoPE; fallback to local implementation."""
        if hasattr(torch.nn.functional, 'rotary_position_embedding'):
            T = q.size(2)
            cos = self.cos_freqs[:T, :].to(device=q.device, dtype=q.dtype)
            sin = self.sin_freqs[:T, :].to(device=q.device, dtype=q.dtype)
            return torch.nn.functional.rotary_position_embedding(q, k, cos, sin)
        # Fallback
        return self._apply_rope(q, k, self.cos_freqs, self.sin_freqs)

        
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

        # Apply RoPE based on configuration
        if self.rope_implementation == 'torch_builtin':
            q, k = self._apply_rope_torch_builtin(q, k)
        elif self.rope_implementation == 'custom':
            q, k = self._apply_rope(q, k, self.cos_freqs, self.sin_freqs)
        # 'none': do not apply RoPE

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
        return x + residual
    
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
        self.W_ue = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=cfg.UE_bias)
        self.W_ue.weight = self.token_embedding.weight
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(cfg, ) for _ in range(cfg.n_layer)
        ])
        self.ln = nn.LayerNorm(cfg.n_embed)
        self.use_checkpoint = cfg.use_checkpoint

    def forward_hidden(self, x):
        """
        Returns pre-logits hidden states after LayerNorm, without projecting to vocab.
        Shape: (B, T, E)
        """
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            if self.training and self.use_checkpoint:
                x = ckpt(block, x)
            else:
                x = block(x)
        x = self.ln(x)
        return x

    def forward(self, x):
        hidden = self.forward_hidden(x)
        return self.W_ue(hidden)
