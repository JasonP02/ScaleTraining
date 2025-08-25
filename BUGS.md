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
