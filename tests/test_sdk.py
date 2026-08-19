"""Shape/contract tests for the SDK surface. Real data tests happen in benchmark."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


def test_trainer_passes_output_directory_to_training(tmp_path):
    """The SDK's output_dir must reach the checkpoint-producing function."""
    trainer = VarformerTrainer(population="nfe", output_dir=tmp_path)
    processed = {"synthetic": "data"}

    with (
        patch("varformer.data.pipeline.ModuleDataProcessor.process", return_value=processed),
        patch(
            "varformer.training.train.train_model",
            return_value=str(tmp_path / "nfe" / "seed42.ckpt"),
        ) as train,
    ):
        paths = trainer.fit(seeds=[42])

    train.assert_called_once_with(processed, output_dir=tmp_path)
    assert paths == [Path(tmp_path / "nfe" / "seed42.ckpt")]

