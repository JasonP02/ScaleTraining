import torch, torch.nn.functional as F
B,H,T,D = 2, 8, 512, 64
q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.MATH]):
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(y.shape)
