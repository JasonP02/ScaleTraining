# Repository Guidelines

## Project Structure & Module Organization
- `src/scaletraining/`: core library (models, training loop, data, utils).
- `conf/`: Hydra configs (defaults, sweeper). Sweeps write to `multirun/`.
- `scripts/`: utilities (e.g., `summarize_multirun.py`).
- `tests/`: test suite entry (`tests/tests.py`).
- `data/`, `outputs/`, `wandb/`, `multirun/`: datasets, artifacts, and run logs.

## Build, Test, and Development Commands
- Run tests: `python tests/tests.py`
- Single test: `python -c "from tests.tests import rope_and_flash_attention_test; rope_and_flash_attention_test()"`
- Build wheel: `python -m build`
- Editable install: `pip install -e .` (ask before installing new packages)
- Train (Hydra): `python -m scaletraining.entrypoints.train`
- Sweep (Hydra): `python -m scaletraining.entrypoints.train -m`

## Coding Style & Naming Conventions
- Use type hints consistently; docstrings for public APIs.
- Import order: standard library, third‑party, local modules.
- Naming: functions `snake_case`, classes `PascalCase`.
- Config objects: `SimpleNamespace` or dataclasses.
- Error handling: `try/except` with specific exceptions.
- PyTorch: follow existing model/training patterns; do not modify Torch/ROCm internals.

## Testing Guidelines
- Keep tests fast and deterministic. Place tests in `tests/`.
- Prefer focused unit tests around model utilities and training steps.
- Run locally via commands above; ensure failures are actionable.

## Commit & Pull Request Guidelines
- Small, focused commits with imperative subject lines (e.g., "Add RoPE summary fallback").
- Include context and rationale in body; reference issues when relevant.
- Open PRs with: purpose, minimal repro (if bug), before/after notes, and any config changes.

## Security & Configuration Tips
- Environment: use conda env `j`. Ask before adding dependencies.
- ROCm: changing or reinstalling `torch` can break the setup—avoid unless approved.
- Hydra paths: default run dir is repo root; sweeps go under `multirun/YYYY-MM-DD/HH-MM-SS/`.
- W&B: local logs in `wandb/run-*/files/` (e.g., `wandb-summary.json`, `config.yaml`).
