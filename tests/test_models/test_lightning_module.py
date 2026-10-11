"""Tests for VarformerLightningModule."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchmetrics import SpearmanCorrCoef
from varformer.training.lightning_module import VarformerLightningModule


@pytest.fixture
def tiny_config(tiny_hyperparams):
    return {"hyperparameters": tiny_hyperparams}


@pytest.fixture
def tiny_lm(tiny_config):
    return VarformerLightningModule(
        config=tiny_config,
        num_samples_per_class=None,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        class_prior=0.1,
        use_pvc=True,
    )


def test_instantiation(tiny_lm):
    assert tiny_lm.model is not None
    assert hasattr(tiny_lm, "acc")
    assert hasattr(tiny_lm, "auroc")
    assert hasattr(tiny_lm, "spearman")
    assert hasattr(tiny_lm, "val_spearman_epoch")
    assert hasattr(tiny_lm, "recall")
    assert hasattr(tiny_lm, "precision")
    assert hasattr(tiny_lm, "f1")
    assert hasattr(tiny_lm, "auprc")
    assert tiny_lm.observed_positive_rate == pytest.approx(0.1)
    assert tiny_lm.pu_class_prior == pytest.approx(0.1)
    assert tiny_lm.threshold_positive_fraction == pytest.approx(0.1)


def test_pu_prior_and_threshold_fraction_are_separate(tiny_config):
    tiny_config["hyperparameters"]["pu_class_prior"] = 0.2
    tiny_config["hyperparameters"]["threshold_positive_fraction"] = 0.3
    module = VarformerLightningModule(
        config=tiny_config,
        num_samples_per_class=None,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        class_prior=0.1,
    )
    assert module.observed_positive_rate == pytest.approx(0.1)
    assert module.pu_class_prior == pytest.approx(0.2)
    assert module.threshold_positive_fraction == pytest.approx(0.3)


def test_nnpu_clipping_indicator_matches_unclipped_negative_risk(tiny_config):
    tiny_config["hyperparameters"]["pu_class_prior"] = 0.2
    module = VarformerLightningModule(
        config=tiny_config,
        num_samples_per_class=None,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=50,
        max_seq_len=16,
        num_genes=100,
        class_prior=0.1,
    )
    labels = torch.tensor([1, 1, 0, 0])
    _, negative_risk, correction_active, loss = module._nnpu_loss_terms(
        torch.tensor([0.9, 0.9, 0.1, 0.1]), labels
    )
    assert negative_risk.item() < 0
    assert correction_active.item() is True
    assert torch.isfinite(loss)

    _, negative_risk, correction_active, loss = module._nnpu_loss_terms(
        torch.tensor([0.1, 0.1, 0.9, 0.9]), labels
    )
    assert negative_risk.item() > 0
    assert correction_active.item() is False
    assert torch.isfinite(loss)


def test_metrics_not_on_model(tiny_lm):
    """Metrics must live on the LightningModule, NOT inside model."""
    model_children = {name for name, _ in tiny_lm.model.named_children()}
    assert "acc" not in model_children
    assert "auroc" not in model_children
    assert "spearman" not in model_children


def test_forward_shape(tiny_lm):
    B, S = 4, 10
    x = {
        "gc": (torch.rand(B, 30),),
        "go": (torch.rand(B, 20),),
        "pvc": {
            "pathogenicity": torch.rand(B, S),
            "position": torch.randint(0, 16, (B, S)),
            "mutation": torch.randint(0, 50, (B, S)),
        },
    }
    mask = torch.zeros(B, S, dtype=torch.bool)
    out = tiny_lm(x, mask)
    assert len(out) == 5  # return_attn=True by default in tiny_hyperparams


def test_model_attribute_names(tiny_lm):
    """Key model attribute names must be preserved for checkpoint loading."""
    names = {n for n, _ in tiny_lm.model.named_children()}
    assert "gc_projection" in names
    assert "go_projection" in names
    assert "varformer" in names
    assert "gene_variant_attention" in names
    assert "classification_head" in names


def test_state_dict_has_no_metric_keys(tiny_lm):
    """State_dict keys under 'model.' must not include metric names."""
    sd = tiny_lm.model.state_dict()
    metric_prefixes = ("acc.", "auroc.", "recall.", "precision.", "f1.", "auprc.", "spearman.")
    for k in sd:
        assert not any(k.startswith(p) for p in metric_prefixes), f"Unexpected metric key in model: {k}"


def test_metrics_registered_as_submodules(tiny_lm):
    """Metrics must be registered as named sub-modules on the LightningModule."""
    children = {name for name, _ in tiny_lm.named_children()}
    assert "acc" in children, "acc not a child of LM"
    assert "spearman" in children, "spearman not a child of LM"
    assert "val_spearman_epoch" in children, "epoch Spearman not a child of LM"
    assert "auroc" in children, "auroc not a child of LM"


def test_validation_spearman_is_one_full_epoch_metric_not_prefix_average(tiny_lm):
    tiny_lm.log = Mock()
    probabilities = (
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([0.4, 0.7, 0.3, 0.6]),
    )
    labels = (
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    )
    for batch_probs, batch_labels in zip(probabilities, labels):
        tiny_lm._log(
            batch_labels,
            "val",
            torch.tensor(0.1),
            (batch_probs > 0.5).float(),
            batch_probs,
        )
    expected = SpearmanCorrCoef()(
        torch.cat(probabilities), torch.cat(labels)
    )
    assert tiny_lm.val_spearman_epoch.compute() == pytest.approx(expected)
    calls = [call for call in tiny_lm.log.call_args_list if call.args[0] == "val_spearman"]
    assert len(calls) == 2
    assert all(call.args[1] is tiny_lm.val_spearman_epoch for call in calls)


def test_validation_spearman_casts_mixed_precision_probabilities_to_float32(tiny_lm):
    tiny_lm.log = Mock()
    probabilities = torch.tensor([0.9, 0.1, 0.8, 0.2], dtype=torch.float16)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    tiny_lm._log(
        labels,
        "val",
        torch.tensor(0.1),
        (probabilities > 0.5).float(),
        probabilities,
    )
    expected = SpearmanCorrCoef()(
        probabilities.to(torch.float32), labels.to(torch.float32)
    )
    assert tiny_lm.val_spearman_epoch.compute() == pytest.approx(expected)


def test_prediction_export_without_pvc_uses_aligned_gene_ids(tiny_config):
    tiny_config["hyperparameters"]["return_attn"] = False
    tiny_config["hyperparameters"]["use_pvc"] = False
    module = VarformerLightningModule(
        config=tiny_config,
        num_samples_per_class=None,
        num_features_gc=30,
        num_features_go=20,
        num_mutations=0,
        max_seq_len=16,
        num_genes=0,
        class_prior=0.1,
        use_pvc=False,
        use_gc=True,
        use_go=True,
    )
    module._trainer = SimpleNamespace(sanity_checking=False)
    batch = {
        "gc": (torch.rand(2, 30), torch.tensor([1.0, 0.0]), "pfam"),
        "go": (torch.rand(2, 20), torch.tensor([1.0, 0.0]), "pfam"),
        "pvc": {"gene_name": ["ENSG1", "ENSG2"]},
    }
    output = module.predict_step(batch, 0)
    assert list(output) == ["ENSG1", "ENSG2"]
    assert output["ENSG1"]["label"] == 1
    assert "z_var" not in output["ENSG1"]
