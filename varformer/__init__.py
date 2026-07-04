"""Varformer — drug-target identification and prioritisation from population genetic variants.

See README.md for usage examples.
"""
__version__ = "1.0.0"

from varformer.models.varformer import Varformer
from varformer.training.train import VarformerTrainer

__all__ = ["Varformer", "VarformerTrainer", "__version__"]
