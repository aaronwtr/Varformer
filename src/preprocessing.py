"""Shim — content moved to varformer.data + paper.baselines. Delete in Phase 8."""
from varformer.data.features.gc import GeneCharacterisationPreprocessor  # noqa: F401
from varformer.data.features.go import GeneOntologyPreprocessor  # noqa: F401
from varformer.data.features.variants import PopulationVariantPreprocessor, extract_pvc_features  # noqa: F401
from varformer.data.loaders import ModelPreprocessorEval, ModelPreprocessorInference  # noqa: F401
# LogisticRegressionPreprocessor moved to paper/baselines/preprocessor.py in Phase 5.
