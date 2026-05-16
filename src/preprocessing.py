"""Shim — content moved to varformer.data + paper.baselines. Delete in Phase 8."""
from varformer.data.features.gc import GeneCharacterisationPreprocessor  # noqa: F401
from varformer.data.features.go import GeneOntologyPreprocessor  # noqa: F401
from varformer.data.features.variants import PopulationVariantPreprocessor, extract_pvc_features  # noqa: F401
# LogisticRegressionPreprocessor moved to paper/baselines/preprocessor.py in Phase 5.


def __getattr__(name):
    """Lazy-load ModelPreprocessorEval/Inference to avoid circular import with varformer.data.loaders."""
    if name in ('ModelPreprocessorEval', 'ModelPreprocessorInference'):
        from varformer.data import loaders as _loaders
        return getattr(_loaders, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
