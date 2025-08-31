# ScaleTraining Agent Guidelines

# Enviornment:
Use conda enviornment "j"
**The use has a custom torch build with ROCm, do not do any torch dependencies or installs to avoid breaking**

## Build/Lint/Test Commands
- Run single test: `python tests/tests.py`
- Install package: `pip install -e .`
- CLI commands available: `scaletraining-tokenize`, `scaletraining-train`, `scaletraining-generate`, `scaletraining-pack`, `scaletraining-prepare-data`

## Code Style Guidelines
- **Imports**: Use standard Python import ordering (stdlib, third-party, local)
- **Formatting**: Follow existing code style - 4-space indentation, no trailing whitespace
- **Types**: Type hints not strictly enforced but encouraged for clarity
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error Handling**: Use assertions for preconditions, try/except for runtime errors
- **Config**: All parameters driven by Hydra YAML in `conf/config.yaml` - no Config dataclasses
- **Structure**: Modular design with separate modules for data_processing, model, training, inference