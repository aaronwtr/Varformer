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


def compare_batch_contents(loaders_old, loaders_new, loader_name="pfam"):
    """Compare the first batch's gene_names + tensor values between OLD and NEW loaders."""
    import numpy as np
    loader_old = loaders_old[loader_name]
    loader_new = loaders_new[loader_name]

    batches_old = list(iter(loader_old))
    batches_new = list(iter(loader_new))
    print(f"  batches in {loader_name}: OLD={len(batches_old)}, NEW={len(batches_new)}")

    for i in range(min(2, len(batches_old), len(batches_new))):
        bo = batches_old[i]
        bn = batches_new[i]
        gn_o = bo["pvc"]["gene_name"] if "pvc" in bo else None
        gn_n = bn["pvc"]["gene_name"] if "pvc" in bn else None
        print(f"  batch {i} gene_names same order: {gn_o == gn_n}")
        if gn_o != gn_n:
            print(f"    OLD first 5: {gn_o[:5] if gn_o else None}")
            print(f"    NEW first 5: {gn_n[:5] if gn_n else None}")
            return
        # Compare each tensor in the batch
        for mod_key in ["gc", "go", "pvc"]:
            ov = bo[mod_key]
            nv = bn[mod_key]
            if isinstance(ov, dict):
                for k in ov:
                    if hasattr(ov[k], "shape"):
                        diff = (ov[k].float() - nv[k].float()).abs().max().item()
                        if diff > 0:
                            print(f"    batch {i} {mod_key}.{k}: shape={tuple(ov[k].shape)} max_diff={diff:.3e}")
            elif isinstance(ov, (list, tuple)):
                for j in range(len(ov)):
                    if hasattr(ov[j], "shape"):
                        diff = (ov[j].float() - nv[j].float()).abs().max().item()
                        if diff > 0:
                            print(f"    batch {i} {mod_key}[{j}]: shape={tuple(ov[j].shape)} max_diff={diff:.3e}")


def compare_first_batch_forward(lm_old, lm_new, loaders_old, loaders_new):
    """Run nn.Module forward (NOT _common_step which needs trainer) on first pfam batch."""
    import numpy as np
    lm_old.eval()
    lm_new.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_old = lm_old.to(device)
    lm_new = lm_new.to(device)

    loader_old = loaders_old["pfam"]
    loader_new = loaders_new["pfam"]

    batch_old = next(iter(loader_old))
    batch_new = next(iter(loader_new))

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

    # Build the model input dict + mask the same way _common_step does
    mask_old = batch_old["pvc"]["mask"]
    mask_new = batch_new["pvc"]["mask"]
    inp_old = {"gc": batch_old["gc"], "go": batch_old["go"], "pvc": batch_old["pvc"]}
    inp_new = {"gc": batch_new["gc"], "go": batch_new["go"], "pvc": batch_new["pvc"]}

    with torch.no_grad():
        out_old = lm_old.model(inp_old, mask_old)
        out_new = lm_new.model(inp_new, mask_new)

    # out is (logits, probas, bin_preds, z_var, attn_weights) when return_attn=True
    for i, name in enumerate(["logits", "probas", "bin_preds", "z_var", "attn_weights"]):
        if i >= len(out_old) or out_old[i] is None or out_new[i] is None:
            print(f"  {name}: skipped (None)")
            continue
        diff = (out_old[i].float() - out_new[i].float()).abs().max().item()
        rel = diff / max(out_old[i].float().abs().max().item(), 1e-12)
        print(f"  {name}: max abs diff = {diff:.3e} (rel {rel:.3e})")


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

    print("=== batch contents (pfam loader) ===")
    compare_batch_contents(loaders_old, loaders_new, "pfam")
    print()

    print("=== forward pass on first pfam batch (nn.Module directly) ===")
    compare_first_batch_forward(lm_old, lm_new, loaders_old, loaders_new)
