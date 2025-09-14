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
        """Use a library RoPE if available, else fallback to local implementation.

        Attempts HuggingFace's LLaMA apply_rotary_pos_emb; otherwise uses local.
        """
        T = q.size(2)
        cos = self.cos_freqs[:T, :].to(device=q.device, dtype=q.dtype)
        sin = self.sin_freqs[:T, :].to(device=q.device, dtype=q.dtype)
        try:
            # HuggingFace implementation (LLaMA). Shapes: [B, N, T, H] are supported.
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _hf_apply_rope
            return _hf_apply_rope(q, k, cos, sin)
        except Exception:
            # Fallback to local implementation
            return self._apply_rope(q, k, self.cos_freqs, self.sin_freqs)

        
    def forward(self, x):
        B, T, E = x.shape
        # Ensure precomputed RoPE tables cover the current sequence length when enabled
        if self.rope_implementation != 'none':
            assert T <= self.max_seq_len, (
                f"Sequence length {T} exceeds RoPE table size {self.max_seq_len}. "
                "Increase model.max_seq_len or reduce input length."
            )

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
        self.Wh = nn.Linear(cfg.n_embed, cfg.n_hidden, bias=cfg.bias)
        self.We = nn.Linear(cfg.n_hidden, cfg.n_embed, bias=cfg.bias)
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

class ExpertFFN(nn.Module):
    def __init__(self, d_model, d_hidden, act="swiGLU", bias=True, device=None):
        super().__init__()
        self.act = act
        self.W1 = nn.Linear(d_model, 2*d_hidden if act=="swiGLU" else d_hidden,
                            bias=bias)
        self.W2 = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        if self.act == "swiGLU":
            a, b = self.W1(x).chunk(2, dim=-1)
            h = F.silu(a) * b
        else:
            h = F.relu(self.W1(x))
        return self.W2(h)


class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_experts  = cfg.moe_n_experts
        self.top_k      = cfg.moe_top_k
        assert 1 <= self.top_k <= self.n_experts

        self._last_aux_loss = None
        self.router_noise = float(cfg.moe_router_noise)
        self.router_temp = float(cfg.moe_router_temp)
        self.router     = nn.Linear(cfg.n_embed, self.n_experts, bias=False)

        self.experts = nn.ModuleList([
            ExpertFFN(cfg.n_embed, cfg.moe_n_hidden, act=cfg.moe_activation,
                      bias=cfg.bias)
            for _ in range(self.n_experts)
        ])
        self.shared = (ExpertFFN(cfg.n_embed, cfg.moe_n_hidden, act=cfg.moe_activation,
                                 bias=cfg.bias)
                       if getattr(cfg, "moe_use_shared", False) else None)

    def forward(self, x):
        B,T,D = x.shape
        z = self.router(x)

        if self.router_noise > 0:
            z = z + self.router_noise * torch.randn_like(z)
        if self.router_temp != 1.0:
            z = z / self.router_temp

        vals, idx = torch.topk(z, self.top_k, dim=-1)
        gates = F.softmax(vals, dim=-1)

        # generally dont understand this.
        p = F.softmax(z, dim=-1) # [B,T,E]
        imp = p.mean(dim=(0,1)) # importance distribution
        E = z.size(-1)
        mass = z.new_zeros(E)
        mass.scatter_add_(0, idx.reshape(-1), gates.reshape(-1))
        load = mass / mass.sum().clamp_min(1e-12)
        self._last_aux_loss = E * (imp * load).sum()

        N = B * T
        flat_x = x.view(N,D) # Tokens flattened for vectorization
        flat_idx = idx.view(N, -1) # [N,k] expert indices per token
        flat_gates = gates.view(N, -1) # [N,k] gate weights per token

        assign_expert = flat_idx.reshape(-1)
        assign_token = torch.arange(N, device=x.device).repeat_interleave(self.top_k) # what
        assign_weight = flat_gates.reshape(-1).unsqueeze(-1)

        order = torch.argsort(assign_expert) # what
        sorted_expert = assign_expert[order]
        sorted_token = assign_token[order]
        sorted_weight = assign_weight[order]

        counts = torch.bincount(assign_expert, minlength=self.n_experts)
        offsets = torch.zeros_like(counts)
        offsets[1:] = torch.cumsum(counts[:-1], dim=0)


        flat_out = torch.zeros_like(flat_x)

        for expert in range(self.n_experts):
            c = counts[expert].item()
            if c == 0:
                continue

            start = offsets[expert].item()
            end = start + c

            tok_ids = sorted_token[start:end]
            w_e = sorted_weight[start:end]
            x_e = flat_x.index_select(0, tok_ids)

            # OLD
            # token_mask = (flat_idx == expert).any(dim=1) # if expert activates on token, make value 1
            # if not token_mask.any():
            #     continue # pass if no expert is used; 
            #     # might want to use this spot for storing statistics
            # tok_ids = torch.where(token_mask)[0]
            # x_e = flat_x[tok_ids]
            # g_e = (flat_gates[tok_ids] * (flat_idx[tok_ids] == expert)).sum(dim=1, keepdim=True)
            y_e = self.experts[expert](x_e)
            flat_out.index_add_(dim=0, index=tok_ids, source=w_e*y_e)
        
        out = flat_out.view(B,T,D)
        if self.shared is not None:
            out += self.shared(x)
        
        return out

class MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embed)
        self.attention = AttentionBlock(cfg)
        self.moe = MoELayer(cfg)
    
    def forward(self, x):
        x = x + self.attention(self.ln(x))
        x = x + self.moe(self.ln(x))
        return x
    
class TransformerNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # We need to create: embedding matrix, stacked transformer block, out logits
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.W_ue = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=cfg.UE_bias)
        self.W_ue.weight = self.token_embedding.weight

        # TODO enable the usage of MoE blocks in designated layers
        block_cls = MoEBlock if cfg.use_moe else TransformerBlock # For comparing MoE and transformer
        self.transformer_blocks = nn.ModuleList([
            block_cls(cfg) for _ in range(cfg.n_layer)
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

    # in TransformerNetwork (src/scaletraining/model/model.py)
    def moe_aux_loss(self):
        total = None
        for m in self.modules():
            if isinstance(m, MoELayer) and getattr(m, "_last_aux_loss", None) is not None:
                total = m._last_aux_loss if total is None else total + m._last_aux_loss
        if total is None:
            return self.W_ue.weight.new_tensor(0.0, dtype=torch.float32)
        return total
