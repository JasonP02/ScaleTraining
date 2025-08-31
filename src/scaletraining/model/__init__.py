from .model import TransformerNetwork
# Keep public API stable by importing the class-based trainer
# from the legacy file while the new functional trainer evolves.
from .oldtrainer import LLMTrainer
from .optimizers import AdaMuon, Muon

__all__ = [
    "TransformerNetwork",
    "LLMTrainer",
    "AdaMuon",
    "Muon",
]
