"""Shape/contract tests for the SDK surface. Real data tests happen in benchmark."""
from unittest.mock import patch

import numpy as np
import pytest

from varformer import Varformer, VarformerTrainer


def test_sdk_re_exports():
    """Top-level imports work; basic types are as expected."""
    assert Varformer is not None
    assert VarformerTrainer is not None


def test_varformer_predict_shape(tiny_hyperparams):
    """predict() returns the documented dict shape (mocked underlying loader)."""
    cfg = {"hyperparameters": tiny_hyperparams}
    model = Varformer(
        config=cfg,
        num_features_gc=20, num_features_go=40, num_mutations=200,
        max_seq_len=16, num_genes=10, use_pvc=True,
    )
    fake_out = {
        "ENSG1": {
            "prediction": 0.7, "classification": 1,
            "z_var": np.zeros(32), "attn_weights": np.zeros(16),
        }
    }
    with patch("varformer.inference.predict.predict_subset", return_value=fake_out):
        result = model.predict(genes=["ENSG1"], return_attention=True)
    assert "ENSG1" in result
    assert isinstance(result["ENSG1"]["prediction"], float)
    assert isinstance(result["ENSG1"]["classification"], int)
    assert hasattr(result["ENSG1"]["z_var"], "tolist")  # JSON-serializable


