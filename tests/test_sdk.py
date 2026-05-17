"""Shape/contract tests for the SDK surface. Real data tests happen in benchmark."""
from unittest.mock import patch

import numpy as np
import pytest

from varformer import Varformer, VarformerEnsemble, VarformerTrainer


def test_sdk_re_exports():
    """Top-level imports work; basic types are as expected."""
    assert Varformer is not None
    assert VarformerEnsemble is not None
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


def test_varformer_ensemble_averaging():
    """Ensemble averages probabilities; classification derived from mean > 0.5."""
    class Stub:
        def __init__(self, p):
            self._p = p
        def predict(self, genes, return_attention=False):
            return {
                g: {"prediction": self._p, "classification": int(self._p > 0.5),
                    "z_var": np.zeros(8)}
                for g in genes
            }
    # mean = (0.2 + 0.5 + 0.9) / 3 = 0.533... > 0.5 -> classification = 1
    ens = VarformerEnsemble([Stub(0.2), Stub(0.5), Stub(0.9)])
    out = ens.predict(genes=["ENSG_A"], return_attention=False)
    assert out["ENSG_A"]["prediction"] == pytest.approx((0.2 + 0.5 + 0.9) / 3)
    assert out["ENSG_A"]["classification"] == 1  # mean > 0.5


def test_varformer_ensemble_empty_raises():
    with pytest.raises(ValueError):
        VarformerEnsemble([])
