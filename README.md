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

Hydra sweeps
- Grid sweep (uses `conf/hydra/sweeper/grid.yaml`):
  - `python -m scaletraining.entrypoints.train -m`
- Ad‑hoc grid from CLI (no YAML change):
  - `python -m scaletraining.entrypoints.train -m lr=1e-4,5e-5 n_head=4,8`
- Inspect composed config (no training):
  - `python -m scaletraining.entrypoints.train --cfg hydra --resolve`
- Outputs:
  - Single run working dir: `runs/<date>/<time>/`
  - Multirun working dir: `multirun/<date>/<time>/<job_num>/`
  - Models saved under `outputs/` (root), independent of Hydra working dir.

Compare results
- Each job writes a `result.json` in its multirun job directory with `final_train_loss` and key params.
- Summarize a sweep:
  - Latest: `python scripts/summarize_multirun.py`
  - Specific: `python scripts/summarize_multirun.py multirun/<date>/<time>`
  - Outputs a table and saves `summary.csv` in the sweep folder.

Notes
- All parameters are driven by Hydra YAML; there is no `Config` dataclass anymore.
- Weights & Biases logs metrics and artifacts (datasets/models). Define your project in `conf/config.yaml` or set env vars.
- Tokenized/packed datasets are organized per-config; compatibility is checked before loading.
