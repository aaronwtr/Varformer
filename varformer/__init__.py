"""Varformer — gene tractability prediction from population genetic variants.

See README.md for usage examples and benchmark/README.md for the regression
benchmark protocol.
"""
__version__ = "0.2.0"

from varformer.models.varformer import Varformer
from varformer.training.train import VarformerTrainer

__all__ = ["Varformer", "VarformerTrainer", "__version__"]
