"""Varformer — gene tractability prediction from population genetic variants.

See docs/superpowers/specs/2026-05-15-varformer-refactor-design.md (local).
"""
__version__ = "0.2.0"

from varformer.models.varformer import Varformer
from varformer.models.ensemble import VarformerEnsemble
from varformer.training.train import VarformerTrainer

__all__ = ["Varformer", "VarformerEnsemble", "VarformerTrainer", "__version__"]
