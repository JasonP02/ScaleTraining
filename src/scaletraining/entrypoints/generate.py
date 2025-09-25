"""Legacy entrypoint for `scaletraining-generate`.

This wrapper keeps the console script defined in `pyproject.toml`
functional after the main implementation moved to
`generate_from_pretrained.py`.
"""

from .generate_from_pretrained import main

__all__ = ["main"]
