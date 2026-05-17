"""Diagnostic: compare OLD path (load_from_checkpoint shim) vs NEW path (Varformer.from_pretrained).

Loads one checkpoint via both paths, compares state_dicts + a single forward pass.
"""
import sys
from pathlib import Path
import yaml
import pickle
import torch
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def load_old_path(population, seed):
    """Replicates benchmark/generate_reference.py model-loading exactly."""
    from dataloader import ModuleDataProcessor
    from models.lightning import MultiModalLightningTargetIdentifier
    from preprocessing import ModelPreprocessorInference

    config_path = REPO / "src" / f"cluster_config_{population}.yml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["hyperparameters"]["population"] = population
    config["hyperparameters"]["return_attn"] = True

    print(f"[OLD] running ModuleDataProcessor...")
    data = ModuleDataProcessor(gc=True, go=True, pvc=True, psc=False, config=config).process()
    splits = data if isinstance(data, list) else [data]
    first = splits[0]
    num_features_gc = first["train"]["gc"].shape[1] - (1 if "target" in first["train"]["gc"].columns else 0)
    num_features_go = first["train"]["go"].shape[1] - (1 if "target" in first["train"]["go"].columns else 0)
    num_genes = len(first["labels"])

    with open(config["paths"]["MISSENSE_MAP"], "rb") as f:
        missense_map = pickle.load(f)
    num_mutations = len(missense_map)

    ckpt_dir = Path(config["paths"]["CKPT_PATH"]) / population
    ckpt_path = list(ckpt_dir.glob(f"seed{seed}-epoch=*-val_spearman=*.ckpt"))[0]

    print(f"[OLD] loading {ckpt_path.name}")
    model = MultiModalLightningTargetIdentifier.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        config=config,
        num_features_gc=num_features_gc,
        num_features_go=num_features_go,
        num_mutations=num_mutations,
        max_seq_len=config["hyperparameters"]["max_seq_len"],
        num_genes=num_genes,
        num_samples_per_class=None,
        class_prior=None,
    )

    consolidated_data = {
        "gc": pd.concat([first["train"]["gc"], first["test_data"]["gc"]]),
        "go": pd.concat([first["train"]["go"], first["test_data"]["go"]]),
    }
    consolidated_pvc = {**first["train"]["pvc"], **first["test_data"]["pvc"]}
    consolidated_pvc.pop("labels", None)
    test_loaders = ModelPreprocessorInference.create_test_loaders(
        config=config,
        consolidated_data=consolidated_data,
        pvc_data=consolidated_pvc,
        torch_dtype=config["hyperparameters"]["precision"],
    )
    return model, test_loaders


def load_new_path(population, seed):
    """SDK path."""
    from varformer import Varformer
    print(f"[NEW] Varformer.from_pretrained({population!r}, seed={seed})")
    instance = Varformer.from_pretrained(population, seed=seed)
    return instance._lightning_module, instance._test_loaders


def compare_state_dicts(sd_a, sd_b, label_a="OLD", label_b="NEW"):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    print(f"  keys only in {label_a}: {len(keys_a - keys_b)}; only in {label_b}: {len(keys_b - keys_a)}")
    common = sorted(keys_a & keys_b)
    print(f"  common keys: {len(common)}")
    max_diff = 0.0
    worst_key = None
    for k in common:
        if sd_a[k].shape != sd_b[k].shape:
            print(f"  SHAPE mismatch: {k} -- {label_a}={tuple(sd_a[k].shape)} vs {label_b}={tuple(sd_b[k].shape)}")
            continue
        diff = (sd_a[k].float() - sd_b[k].float()).abs().max().item()
        if diff > max_diff:
            max_diff = diff
            worst_key = k
    print(f"  max abs diff across params: {max_diff:.3e}  (key: {worst_key})")
    return max_diff


def compare_first_batch_forward(lm_old, lm_new, loaders_old, loaders_new):
    """Run forward on the first batch of pfam loader for both models. Compare outputs."""
    lm_old.eval()
    lm_new.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_old = lm_old.to(device)
    lm_new = lm_new.to(device)

    loader_old = loaders_old["pfam"]
    loader_new = loaders_new["pfam"]

    it_old = iter(loader_old)
    it_new = iter(loader_new)
    batch_old = next(it_old)
    batch_new = next(it_new)

    # Move tensors in dicts to device
    def to_dev(b):
        out = {}
        for k, v in b.items():
            if isinstance(v, dict):
                out[k] = {kk: vv.to(device) if hasattr(vv, "to") else vv for kk, vv in v.items()}
            elif isinstance(v, (list, tuple)):
                out[k] = [x.to(device) if hasattr(x, "to") else x for x in v]
            elif hasattr(v, "to"):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    batch_old = to_dev(batch_old)
    batch_new = to_dev(batch_new)

    # Use the LightningModule's _common_step with predict to get the dict output
    with torch.no_grad():
        out_old = lm_old._common_step(batch_old, 0, "predict")
        out_new = lm_new._common_step(batch_new, 0, "predict")

    keys_common = set(out_old.keys()) & set(out_new.keys())
    print(f"  forward genes in common: {len(keys_common)} (OLD={len(out_old)}, NEW={len(out_new)})")
    max_pred = 0.0
    max_attn = 0.0
    max_zvar = 0.0
    for g in keys_common:
        max_pred = max(max_pred, abs(out_old[g]["prediction"] - out_new[g]["prediction"]))
        import numpy as np
        a, b = np.asarray(out_old[g]["attn_weights"]), np.asarray(out_new[g]["attn_weights"])
        max_attn = max(max_attn, float(np.max(np.abs(a - b))))
        a, b = np.asarray(out_old[g]["z_var"]), np.asarray(out_new[g]["z_var"])
        max_zvar = max(max_zvar, float(np.max(np.abs(a - b))))
    print(f"  forward pred max diff: {max_pred:.3e}")
    print(f"  forward attn max diff: {max_attn:.3e}")
    print(f"  forward z_var max diff: {max_zvar:.3e}")


if __name__ == "__main__":
    pop, seed = "nfe", 42

    lm_old, loaders_old = load_old_path(pop, seed)
    print()
    lm_new, loaders_new = load_new_path(pop, seed)
    print()

    print("=== state_dict comparison (OLD vs NEW Lightning module) ===")
    sd_old = lm_old.state_dict()
    sd_new = lm_new.state_dict()
    compare_state_dicts(sd_old, sd_new)
    print()

    print("=== state_dict comparison (OLD.model vs NEW.model, the nn.Module Varformer) ===")
    sd_old_inner = lm_old.model.state_dict()
    sd_new_inner = lm_new.model.state_dict()
    compare_state_dicts(sd_old_inner, sd_new_inner)
    print()

    print("=== forward pass on first pfam batch ===")
    compare_first_batch_forward(lm_old, lm_new, loaders_old, loaders_new)
