# Code audit: bugs and fixes

This report identifies concrete bugs, likely runtime errors, and risky patterns across your repository. Each item references exact locations.

High-impact runtime issues


3) RoPE buffers are on CPU; device/dtype mismatch
- [create_rope_lookup()](model.py:25) creates cos/sin tensors that are not registered buffers and live on CPU. When the model is moved to CUDA in [LLMTrainer.train()](main.py:102), q/k are on GPU, leading to device mismatch in [_apply_rope()](model.py:39). Fix:
  - Register buffers and ensure correct device/dtype at use-time:
    - self.register_buffer('cos_freqs', cos, persistent=False)
    - self.register_buffer('sin_freqs', sin, persistent=False)
    - In [_apply_rope()](model.py:39), take slices then .to(q.device, dtype=q.dtype) and unsqueeze to broadcast: cos = cos[:T].unsqueeze(0).unsqueeze(0); same for sin.

4) RoPE broadcasting and even head_dim assumption
- [_apply_rope()](model.py:39) relies on implicit broadcasting of (T, H/2) over (B, N, T, H/2). This works but is brittle. Safer to unsqueeze as noted above.
- Add an explicit check in [AttentionBlock.__init__()](model.py:7) that self.head_dim % 2 == 0 to avoid odd-dimension failures.

5) Unused global optimizer instantiation
- The block [main.py](main.py:47-57) builds matrix_params and constructs [AdaMuon.__init__()](optimizers.py:88) into admuon, but training only uses the optimizer instantiated in [LLMTrainer.__init__()](main.py:85). This global optimizer is unused and can lead to confusion or accidental step calls. Remove that block.

6) Validation loader should not shuffle
- In [load_tiny_stories()](dataload.py:12) the validation loader uses shuffle=True. Use shuffle=False for validation/eval.

7) Test harness reports success even on failure
- In [run_all_tests()](tests.py:29), it prints All tests passed! unconditionally after the try/except. If the attention test fails, it still reports success. Track a status flag and only print success when all checks pass.

Mixed-precision and numerical robustness

8) Optimizer dtype safety when training with bf16/fp16
- In [_BaseOptimizer.step()](optimizers.py:45), gradients are cast to float for half/bfloat16, which is fine, but the computed update direction (e.g., from [AdaMuon.update_weights()](optimizers.py:114)) stays float. If model params are bf16, param.add_(direction) will raise a dtype mismatch. Fix: cast direction = direction.to(param.dtype) before the in-place add.

9) Weight decay formulation
- In [_BaseOptimizer.step()](optimizers.py:77), the decoupled weight decay uses param.add_(param, alpha=-lr*wd), which is mathematically equivalent to param.mul_(1 - lr*wd). That's correct, just confirming intent.

Quality and consistency improvements

10) Device handling consistency in modules
- [MLPBlock.__init__()](model.py:95) creates layers with device=cfg.device, while other modules rely on model.to(cfg.device). For consistency and to simplify checkpoint loading, prefer constructing on default device and moving the whole model with .to().

11) Ensure device move in eval path
- [LLMTrainer.eval()](main.py:151) sets eval mode but does not ensure model is on cfg.device (it relies on train() having been run). Either move inside eval too or enforce a single initialization path that moves once.

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