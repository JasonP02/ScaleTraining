# Baseline Comparison Examples

This document provides examples for using the new baseline comparison features.

## 1. Optimizer Baseline Comparisons

### Custom Optimizer (Default)
```bash
# Use your custom AdaMuon optimizer (current default)
scaletraining-train primary_optimizer=adamuon

# Use your custom Muon optimizer
scaletraining-train primary_optimizer=muon

# Use AdamW for matrix parameters only
scaletraining-train primary_optimizer=adamw
```

### Baseline Adam (Single Optimizer for All Parameters)
```bash
# Use standard AdamW for all parameters (baseline comparison)
scaletraining-train use_baseline_adam=true

# Use baseline Adam with custom settings
scaletraining-train use_baseline_adam=true baseline_adam_config.lr=1e-3 baseline_adam_config.weight_decay=0.01

# Use baseline Adam with different betas
scaletraining-train use_baseline_adam=true baseline_adam_config.betas='[0.9, 0.999]'
```

## 2. RoPE Implementation Comparisons

### Custom RoPE (Current Implementation)
```bash
# Use your custom RoPE implementation (current default)
scaletraining-train rope_implementation=custom

# Use custom RoPE with different theta
scaletraining-train rope_implementation=custom rope_config.theta=50000
```

### PyTorch Built-in RoPE
```bash
# Use PyTorch's optimized RoPE implementation
scaletraining-train rope_implementation=torch_builtin

# Use PyTorch RoPE with custom theta
scaletraining-train rope_implementation=torch_builtin rope_config.theta=50000
```

### No RoPE (No Positional Encoding)
```bash
# Disable RoPE entirely (no positional encoding)
scaletraining-train use_rope=false
```

## 3. Combined Baseline Experiments

### Pure Transformer Baseline
```bash
# No RoPE + Baseline Adam (pure transformer baseline)
scaletraining-train use_rope=false use_baseline_adam=true

# Tag the experiment for easy tracking
scaletraining-train use_rope=false use_baseline_adam=true experiment_tags='["pure_transformer_baseline"]'
```

### Custom Implementation vs Built-in
```bash
# Your custom implementations
scaletraining-train rope_implementation=custom primary_optimizer=adamuon experiment_tags='["custom_impl"]'

# Built-in implementations  
scaletraining-train rope_implementation=torch_builtin use_baseline_adam=true experiment_tags='["builtin_impl"]'
```

## 4. Performance Comparison Commands

### Compare All Optimizers
```bash
# Custom AdaMuon
scaletraining-train primary_optimizer=adamuon experiment_tags='["adamuon"]'

# Custom Muon
scaletraining-train primary_optimizer=muon experiment_tags='["muon"]'

# Baseline Adam
scaletraining-train use_baseline_adam=true experiment_tags='["baseline_adam"]'
```

### Compare All RoPE Implementations
```bash
# Custom RoPE
scaletraining-train rope_implementation=custom experiment_tags='["custom_rope"]'

# PyTorch RoPE
scaletraining-train rope_implementation=torch_builtin experiment_tags='["torch_rope"]'

# No RoPE
scaletraining-train use_rope=false experiment_tags='["no_rope"]'
```

## 5. Configuration File Examples

### config_custom.yaml (Your Implementations)
```yaml
# Use your custom implementations
primary_optimizer: adamuon
use_baseline_adam: false
use_rope: true
rope_implementation: custom
rope_config:
  theta: 10000
experiment_tags: ["custom_implementation"]
```

### config_baseline.yaml (Standard Implementations)
```yaml
# Use standard implementations for baseline comparison
use_baseline_adam: true
baseline_adam_config:
  lr: 1e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]
use_rope: true
rope_implementation: torch_builtin
rope_config:
  theta: 10000
experiment_tags: ["baseline_implementation"]
```

### config_pure_transformer.yaml (Minimal Transformer)
```yaml
# Pure transformer with no custom features
use_baseline_adam: true
use_rope: false
experiment_tags: ["pure_transformer"]
```

## 6. Running Comparative Experiments

### Method 1: Command Line Overrides
```bash
# Run all experiments with command line overrides
for optimizer in "adamuon" "muon" "baseline_adam"; do
  for rope in "custom" "torch_builtin" "none"; do
    if [ "$optimizer" = "baseline_adam" ]; then
      cmd="scaletraining-train use_baseline_adam=true"
    else
      cmd="scaletraining-train primary_optimizer=$optimizer"
    fi
    
    if [ "$rope" = "none" ]; then
      cmd="$cmd use_rope=false"
    else
      cmd="$cmd rope_implementation=$rope"
    fi
    
    cmd="$cmd experiment_tags='[\"${optimizer}_${rope}\"]'"
    echo "Running: $cmd"
    eval $cmd
  done
done
```

### Method 2: Configuration Files
```bash
# Run experiments using different config files
scaletraining-train --config-path=config_custom.yaml
scaletraining-train --config-path=config_baseline.yaml  
scaletraining-train --config-path=config_pure_transformer.yaml
```

## 7. Monitoring and Analysis

### Check Implementation Details in W&B
All experiments automatically log implementation details:
- `optimizer`: Which optimizer is being used
- `rope_enabled`: Whether RoPE is enabled
- `rope_implementation`: Which RoPE implementation is used

### Check Implementation Details in Run Manifest
Each run saves a `run_manifest.json` with detailed implementation information:
```json
{
  "implementation": {
    "optimizer": "baseline_adam",
    "rope": {
      "enabled": true,
      "implementation": "torch_builtin",
      "theta": 10000
    }
  }
}
```

## 8. Expected Results

### Performance Expectations
- **Custom RoPE vs PyTorch RoPE**: PyTorch built-in should be faster and more memory efficient
- **Custom Optimizers vs Baseline Adam**: Your custom optimizers should show better convergence for large models
- **With RoPE vs Without RoPE**: Models with RoPE should perform better on longer sequences

### Memory Usage
- **Baseline Adam**: Uses slightly less memory (single optimizer vs dual optimizers)
- **PyTorch RoPE**: Uses less memory than custom RoPE implementation
- **No RoPE**: Uses the least memory (no frequency buffers)

### Training Speed
- **Baseline Adam**: Faster step time (single optimizer update)
- **PyTorch RoPE**: Faster attention computation
- **No RoPE**: Fastest (no positional encoding computations)

These features enable rigorous A/B testing of your custom implementations against standard baselines while maintaining full backward compatibility.