# Code audit: bugs and fixes

This report identifies concrete bugs, likely runtime errors, and risky patterns across your repository. Each item references exact locations.

High-impact runtime issues







Quality and consistency improvements

10) Device handling consistency in modules
- [MLPBlock.__init__()](model.py:95) creates layers with device=cfg.device, while other modules rely on model.to(cfg.device). For consistency and to simplify checkpoint loading, prefer constructing on default device and moving the whole model with .to().

12) Tokenization-vocab mismatch is a placeholder
- [LLMTrainer.train()](main.py:111) uses ord(c) % 1000 while [TransformerNetwork.__init__()](model.py:123) sets vocab_size=16000. Not an error, but you are only using the first 1000 ids. When you swap in a real tokenizer, ensure targets in range [0, vocab_size).

13) Minor nits
- [AttentionBlock.forward()](model.py:73) uses self.attn_dropout as a float for SDPA. That's valid; if you prefer symmetry with residual dropout, you could also keep a nn.Dropout layer for attention outputs, but it's optional.
- [dataload.py](dataload.py) imports dataclass but does not use it.

Suggested minimal patches (no behavioral changes beyond fixes)

- tests import fix:
  - At [tests.py](tests.py:1), import [AttentionBlock.__init__()](model.py:7) from model, and [Config](main.py:15) from main.
- Dataloader fixes:
  - Remove both set_format calls in [load_tiny_stories()](dataload.py:6-12).
  - Set val DataLoader shuffle to False at [dataload.py](dataload.py:12).
- RoPE buffers/device:
  - In [create_rope_lookup()](model.py:25), register cos/sin as non-persistent buffers.
  - In [_apply_rope()](model.py:39), add .to(q.device, dtype=q.dtype) and unsqueeze for broadcasting.
- Remove unused optimizer block:
  - Delete [main.py](main.py:47-57).
- Test harness truthfulness:
  - In [run_all_tests()](tests.py:29-47), gate the final success message on a flag.