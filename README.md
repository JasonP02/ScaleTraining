# ScaleTrain: A repository for scaling up LLM training

This is a personal project where I start with a simple baseline: train a 10m transformer on tinystories, and scale up parameters so I can train a 7B model.

Status: implementing decoder only transformer architecture for tinystories, and training

Quick start
- Configure via `conf/config.yaml` (Hydra is the single source of truth).
- Tokenize dataset: `scaletraining-tokenize`
- Pack dataset into fixed blocks: `scaletraining-pack`
- Fully prepare data (tokenize + pack). Supports multiple datasets in `hf_dataset_names` (list):
  - `scaletraining-prepare-data`
- Train: `scaletraining-train`
- Generate: `scaletraining-generate model_path=outputs/<run>/model.pt prompt="Once upon a time"`

Notes
- All parameters are driven by Hydra YAML; there is no `Config` dataclass anymore.
- Weights & Biases logs metrics and artifacts (datasets/models). Define your project in `conf/config.yaml` or set env vars.
- Tokenized/packed datasets are organized per-config; compatibility is checked before loading.
