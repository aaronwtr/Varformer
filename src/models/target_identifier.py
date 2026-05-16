"""Shim — re-exports from varformer.models.

DELETED classes (no replacement — they were dead):
  - BaseTargetIdentifier
  - VarformerTargetIdentifier (was a useless wrapper with an unused classifier)
  - MultiModalTargetIdentifierV1 (legacy)
"""
from varformer.models.attention import GeneVariantAttention  # noqa: F401
from varformer.models.varformer import Varformer as MultiModalTargetIdentifier  # noqa: F401
