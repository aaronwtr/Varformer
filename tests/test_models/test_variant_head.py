"""Tests for the supervised per-variant head."""
import copy

import pytest
import torch

from varformer.models.varformer import Varformer

B, S, NUM_GC, NUM_GO, NUM_MUTS = 4, 10, 30, 20, 50


def build(hyperparams, **overrides):
    params = copy.deepcopy(hyperparams)
    params.update(overrides)
    return Varformer(
        config={"hyperparameters": params},
        num_features_gc=NUM_GC,
        num_features_go=NUM_GO,
        num_mutations=NUM_MUTS,
        max_seq_len=16,
        num_genes=100,
        use_pvc=True,
    )


def batch(seed=0):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, S, dtype=torch.bool)
    mask[:, S - 3:] = True
    return {
        "gc": (torch.rand(B, NUM_GC, generator=generator),),
        "go": (torch.rand(B, NUM_GO, generator=generator),),
        "pvc": {
            "pathogenicity": torch.rand(B, S, generator=generator),
            "position": torch.randint(0, 16, (B, S), generator=generator),
            "mutation": torch.randint(0, NUM_MUTS, (B, S), generator=generator),
        },
    }, mask


def test_head_absent_by_default(tiny_hyperparams):
    model = build(tiny_hyperparams)
    assert model.variant_head is None
    assert len(model(*batch())) == 5


def test_head_appends_scores_of_token_shape(tiny_hyperparams):
    model = build(tiny_hyperparams, variant_head=True)
    out = model(*batch())
    assert len(out) == 6
    assert out[-1].shape == (B, S)


def test_gene_pathway_unchanged_when_head_enabled(tiny_hyperparams):
    x, mask = batch()
    torch.manual_seed(0)
    without = build(tiny_hyperparams).eval()
    torch.manual_seed(0)
    with_head = build(tiny_hyperparams, variant_head=True).eval()
    with torch.no_grad():
        a, b = without(x, mask), with_head(x, mask)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[3], b[3], rtol=0, atol=0)


@pytest.mark.parametrize("ablate", [["mutation"], ["position"], ["mutation", "position"]])
def test_token_ablation_changes_representation(tiny_hyperparams, ablate):
    x, mask = batch()
    torch.manual_seed(0)
    full = build(tiny_hyperparams, variant_head=True).eval()
    torch.manual_seed(0)
    reduced = build(tiny_hyperparams, variant_head=True, variant_token_ablate=ablate).eval()
    with torch.no_grad():
        assert not torch.allclose(full(x, mask)[-1], reduced(x, mask)[-1])
    assert reduced.varformer.token_ablate == tuple(ablate)


def test_masked_loss_ignores_padding(tiny_hyperparams):
    from varformer.training.lightning_module import VarformerLightningModule

    scores = torch.zeros(B, S, requires_grad=True)
    observed = torch.zeros(B, S, dtype=torch.bool)
    observed[:, :2] = True
    labels = torch.zeros(B, S)
    labels[:, S - 1] = 1.0

    payload = {"pvc": {"variant_labels": labels, "variant_label_mask": observed,
                       "variant_strata": torch.zeros(B, S, dtype=torch.long)}}
    stub = LossStub(variant_objective="pooled_bce")
    loss = VarformerLightningModule._variant_loss(stub, scores, payload)
    torch.testing.assert_close(loss, torch.tensor(0.6931472))


def test_missing_labels_yield_no_loss(tiny_hyperparams):
    from varformer.training.lightning_module import VarformerLightningModule

    class Stub:
        hyperparams = {"variant_loss_weight": 0.0, "variant_objective": "pooled_bce"}

    assert VarformerLightningModule._variant_loss(Stub(), torch.zeros(B, S), {"pvc": {}}) is None
    assert VarformerLightningModule._variant_loss(Stub(), None, {"pvc": {}}) is None


def test_missing_labels_raise_when_loss_is_weighted():
    from varformer.training.lightning_module import VarformerLightningModule

    class Stub:
        hyperparams = {"variant_loss_weight": 0.3}

    with pytest.raises(ValueError, match="no per-variant labels|per-variant labels"):
        VarformerLightningModule._variant_loss(Stub(), torch.zeros(B, S), {"pvc": {}})


def test_labels_load_from_pydantic_config(tmp_path):
    """Config is a pydantic model, not a dict; the resolver must not assume dict."""
    import pandas as pd
    from varformer.data import variant_labels as module

    table = tmp_path / "labels.parquet"
    pd.DataFrame({"variant_id": ["chr1_1_A_G_M1V"], "label": [1.0],
                  "observed": [True], "stratum": ["T>C"]}).to_parquet(table)

    class Hyper(dict):
        pass

    cfg = {"hyperparameters": Hyper(variant_labels_path=str(table))}
    monkey = {"ENSG1": ["chr1_1_A_G_M1V", "chr1_2_C_T_A2V"]}
    module.resolve_variant_features_dir = lambda *a, **k: str(tmp_path)
    import pickle
    (tmp_path / "gene_loc_var_map.pkl").write_bytes(pickle.dumps(monkey))
    aligned = module.load_variant_labels(cfg, "sas")
    labels, in_loss, probe, strata = aligned["ENSG1"]
    assert labels.tolist() == [1.0, 0.0]
    assert (in_loss | probe).tolist() == [True, False]
    assert strata.tolist() == [4, -1]


class LossStub:
    """Minimal stand-in exposing only what _variant_loss reads."""

    def __init__(self, **hyperparams):
        self.hyperparams = {"variant_loss_weight": 0.3,
                            "variant_objective": "within_gene_ranking",
                            "variant_pairs_per_gene": 256}
        self.hyperparams.update(hyperparams)


def ranking_batch(scores, labels, in_loss, strata):
    return {"pvc": {"variant_labels": torch.tensor([labels], dtype=torch.float),
                    "variant_label_mask": torch.tensor([in_loss], dtype=torch.bool),
                    "variant_strata": torch.tensor([strata], dtype=torch.long)}}, \
           torch.tensor([scores], dtype=torch.float)


def test_ranking_loss_matches_hand_computation():
    from varformer.training.lightning_module import VarformerLightningModule

    # stratum 0: rare 1.2, -0.4 vs common 0.1 | stratum 1: rare 0.8 vs common 1.5
    batch, scores = ranking_batch([1.2, -0.4, 0.1, 0.8, 1.5], [1, 1, 0, 1, 0],
                                  [1, 1, 1, 1, 1], [0, 0, 0, 1, 1])
    loss = VarformerLightningModule._variant_loss(LossStub(), scores, batch)
    expected = torch.nn.functional.softplus(-torch.tensor([1.1, -0.5, -0.7])).mean()
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)


def test_ranking_loss_is_invariant_to_a_per_gene_shift():
    """The property the design rests on: gene-level offsets get zero gradient."""
    from varformer.training.lightning_module import VarformerLightningModule

    base = [1.2, -0.4, 0.1, 0.8, 1.5]
    args = ([1, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 0, 0, 1, 1])
    a = VarformerLightningModule._variant_loss(LossStub(), *reversed(ranking_batch(base, *args)))
    shifted = [v + 5.0 for v in base]
    b = VarformerLightningModule._variant_loss(LossStub(), *reversed(ranking_batch(shifted, *args)))
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_ranking_loss_never_pairs_across_strata():
    from varformer.training.lightning_module import VarformerLightningModule

    # rare in stratum 0, common only in stratum 1: no eligible pair anywhere
    batch, scores = ranking_batch([1.0, 2.0], [1, 0], [1, 1], [0, 1])
    assert VarformerLightningModule._variant_loss(LossStub(), scores, batch) is None


def test_ranking_loss_ignores_tokens_withheld_from_the_loss():
    from varformer.training.lightning_module import VarformerLightningModule

    full, _ = ranking_batch([1.2, -0.4, 0.1], [1, 1, 0], [1, 1, 1], [0, 0, 0])
    scores = torch.tensor([[1.2, -0.4, 0.1]])
    both = VarformerLightningModule._variant_loss(LossStub(), scores, full)
    partial, _ = ranking_batch([1.2, -0.4, 0.1], [1, 1, 0], [1, 0, 1], [0, 0, 0])
    one = VarformerLightningModule._variant_loss(LossStub(), scores, partial)
    expected = torch.nn.functional.softplus(-torch.tensor(1.1))
    torch.testing.assert_close(one, expected, rtol=1e-5, atol=1e-5)
    assert not torch.isclose(both, one)


def test_pooled_objective_still_available():
    from varformer.training.lightning_module import VarformerLightningModule

    batch, scores = ranking_batch([0.0, 0.0], [1, 0], [1, 1], [0, 0])
    stub = LossStub(variant_objective="pooled_bce")
    loss = VarformerLightningModule._variant_loss(stub, scores, batch)
    torch.testing.assert_close(loss, torch.tensor(0.6931472), rtol=1e-5, atol=1e-5)


def test_probe_genes_are_stable_and_disjoint_from_loss():
    from varformer.data.variant_labels import is_probe_gene

    genes = [f"ENSG{i:08d}" for i in range(4000)]
    flags = [is_probe_gene(g) for g in genes]
    assert 0.12 < sum(flags) / len(flags) < 0.18
    assert flags == [is_probe_gene(g) for g in genes]


def test_dataset_emits_probe_strata_and_disjoint_masks():
    from varformer.data.datasets import MultiModalData

    variant_data = {"data": {"G1": torch.tensor([[0.1, 1, 2, 0], [0.2, 3, 4, 0],
                                                 [0.9, 5, 6, 0]])},
                    "labels": {"G1": 1.0}}
    labels = (torch.tensor([1.0, 0.0, 1.0]),
              torch.tensor([True, True, False]),      # in loss
              torch.tensor([False, False, True]),     # probe
              torch.tensor([2, 2, 5]))
    ds = MultiModalData(None, None, ["G1"], torch.float32, variant_data=variant_data,
                        max_variants=6, variant_labels={"G1": labels})
    item = ds[0]
    assert item["variant_label_mask"].tolist() == [True, True, False, False, False, False]
    assert item["variant_probe_mask"].tolist() == [False, False, True, False, False, False]
    assert not (item["variant_label_mask"] & item["variant_probe_mask"]).any()
    assert item["variant_strata"].tolist() == [2, 2, 5, 0, 0, 0]
    # padding is never labelled, probed, or given a real stratum position
    assert not item["variant_label_mask"][3:].any()
    assert not item["variant_probe_mask"][3:].any()


def cached_loader_model(hyperparams, monkeypatch, **overrides):
    x, mask = batch()
    x["pvc"]["mask"] = mask
    x["pvc"]["gene_name"] = [f"ENSG{i}" for i in range(B)]
    model = build(hyperparams, **overrides)
    monkeypatch.setattr("varformer.inference.predict._gene_loader", lambda model, genes=None: [x])
    return model, x, mask


def test_score_variants_returns_unpadded_token_scores(tiny_hyperparams, monkeypatch):
    model, x, mask = cached_loader_model(tiny_hyperparams, monkeypatch, variant_head=True)
    scored = model.score_variants()
    with torch.no_grad():
        expected = model(x, mask)[-1]
    assert set(scored) == {f"ENSG{i}" for i in range(B)}
    for i in range(B):
        payload = scored[f"ENSG{i}"]
        assert payload["variant_scores"].shape == (S - 3,)
        torch.testing.assert_close(
            torch.from_numpy(payload["variant_scores"]), expected[i, : S - 3]
        )
        torch.testing.assert_close(
            torch.from_numpy(payload["pathogenicity"]), x["pvc"]["pathogenicity"][i, : S - 3]
        )


def test_gene_loader_keeps_requested_genes_present_in_all_modalities():
    import pandas as pd

    from varformer.data.loaders import ModelPreprocessorInference

    genes = ["G1", "G2", "G3"]
    data = {"gc": pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=genes),
            "go": pd.DataFrame({"b": [1.0, 2.0, 3.0]}, index=genes)}
    pvc = {g: torch.tensor([[0.5, 1, 2, 0]]) for g in ("G1", "G2")}
    config = {"hyperparameters": {"max_seq_len": 4}}
    loader = ModelPreprocessorInference.create_gene_loader(config, data, pvc, ["G1", "G3", "missing"], torch.float32)
    assert [b["pvc"]["gene_name"] for b in loader] == [["G1"]]
    assert ModelPreprocessorInference.create_gene_loader(config, data, pvc, ["missing"], torch.float32) is None


def test_score_variants_requires_head(tiny_hyperparams, monkeypatch):
    model, _, _ = cached_loader_model(tiny_hyperparams, monkeypatch)
    with pytest.raises(ValueError, match="variant head"):
        model.score_variants()
