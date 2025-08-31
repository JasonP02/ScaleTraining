import torch
from types import SimpleNamespace

from scaletraining.model.model import AttentionBlock


def rope_and_flash_attention_test():
    """Sanity-check AttentionBlock preserves shape across typical sizes."""
    cfg = SimpleNamespace(n_embed=256, n_head=8, bias=True, resid_dropout=0.1, attn_dropout=0.1, max_seq_len=128)
    test_cases = [
        (2, 10, cfg.n_embed),
        (1, 1, cfg.n_embed),
        (4, 128, cfg.n_embed),
    ]
    for batch, seq, embed in test_cases:
        x = torch.randn(batch, seq, embed)
        attention_block = AttentionBlock(cfg)
        out = attention_block.forward(x)
        assert x.shape == out.shape, "Output shape should match input"


if __name__ == "__main__":
    rope_and_flash_attention_test()
    print("Attention test passed")
