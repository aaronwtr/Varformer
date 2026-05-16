"""Compare current package predictions against frozen benchmark/reference/ tensors.

Run after every refactor phase. Pass = all (pop, seed) pairs match within tolerance.
Exit 0 on pass, 1 on fail.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_lightning import Trainer

REPO = Path(__file__).resolve().parents[1]

# Tolerances per design spec §2
TOL_PRED_ABS = 1e-5
TOL_ATTN_ABS = 1e-5
TOL_ZVAR_REL = 1e-3

CHECKPOINTS = {
    "nfe": [42, 85, 482, 589, 612],
    "elgh": [7, 32, 57, 64, 482],
}


def _run_inference(population: str, seed: int) -> dict:
    """Run inference with whatever code is current.

    Phases 0..6: imports from src/. Phase 7+: imports from varformer/.
    """
    try:
        # SDK path (Phase 7+)
        from varformer import Varformer
        input_genes = (REPO / "benchmark" / "inputs" / f"{population}_genes.txt").read_text().splitlines()
        model = Varformer.from_pretrained(population, seed=seed)
        return model.predict(genes=input_genes, return_attention=True)
    except (ImportError, AttributeError):
        pass

    # src/-based path (Phases 0..6)
    sys.path.insert(0, str(REPO / "src"))
    from dataloader import ModuleDataProcessor
    from models.lightning import MultiModalLightningTargetIdentifier
    from preprocessing import ModelPreprocessorInference

    config_path = REPO / "src" / f"cluster_config_{population}.yml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["hyperparameters"]["population"] = population
    config["hyperparameters"]["return_attn"] = True

    data = ModuleDataProcessor(gc=True, go=True, pvc=True, psc=False, config=config).process()
    input_genes = set((REPO / "benchmark" / "inputs" / f"{population}_genes.txt").read_text().splitlines())

    import pandas as pd
    splits = data if isinstance(data, list) else [data]
    first = splits[0]

    # Inference mode: each split partitions the same common_genes into train+test.
    # First split's train+test reconstructs the full gene set.
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

    with open(config["paths"]["MISSENSE_MAP"], "rb") as f:
        missense_map = pickle.load(f)
    num_mutations = len(missense_map)
    num_genes = len(first["labels"])  # full common_genes count
    num_features_gc = first["train"]["gc"].shape[1] - (1 if "target" in first["train"]["gc"].columns else 0)
    num_features_go = first["train"]["go"].shape[1] - (1 if "target" in first["train"]["go"].columns else 0)

    ckpt_dir = Path(config["paths"]["CKPT_PATH"]) / population
    ckpt_path = list(ckpt_dir.glob(f"seed{seed}-epoch=*-val_spearman=*.ckpt"))[0]

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
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    results: dict[str, dict] = {}
    for loader in test_loaders.values():
        for batch in trainer.predict(model=model, dataloaders=loader):
            for gid, payload in batch.items():
                if gid in input_genes:
                    results[gid] = payload
    return results


def _compare(reference: dict, candidate: dict) -> tuple[bool, str]:
    missing = set(reference) - set(candidate)
    extra = set(candidate) - set(reference)
    if missing or extra:
        return False, f"missing={len(missing)} extra={len(extra)}"

    max_pred = 0.0
    max_attn = 0.0
    max_zvar_rel = 0.0
    cls_mismatches = 0

    for gid, ref in reference.items():
        cand = candidate[gid]
        max_pred = max(max_pred, abs(ref["prediction"] - cand["prediction"]))
        if ref["classification"] != cand["classification"]:
            cls_mismatches += 1
        a, b = np.asarray(ref["attn_weights"]), np.asarray(cand["attn_weights"])
        max_attn = max(max_attn, float(np.max(np.abs(a - b))))
        a, b = np.asarray(ref["z_var"]), np.asarray(cand["z_var"])
        rel = np.max(np.abs(a - b) / (np.abs(a) + 1e-12))
        max_zvar_rel = max(max_zvar_rel, float(rel))

    ok = (
        max_pred < TOL_PRED_ABS
        and cls_mismatches == 0
        and max_attn < TOL_ATTN_ABS
        and max_zvar_rel < TOL_ZVAR_REL
    )
    msg = f"pred={max_pred:.2e} cls_mismatches={cls_mismatches} attn={max_attn:.2e} z_var_rel={max_zvar_rel:.2e}"
    return ok, msg


def main(populations: list) -> int:
    all_pass = True
    print(f"{'pop':<6} {'seed':<6} {'status':<6} details")
    print("-" * 70)
    for pop in populations:
        for seed in CHECKPOINTS[pop]:
            ref_path = REPO / "benchmark" / "reference" / pop / f"seed{seed}.pkl"
            with ref_path.open("rb") as f:
                reference = pickle.load(f)
            candidate = _run_inference(pop, seed)
            ok, msg = _compare(reference, candidate)
            status = "PASS" if ok else "FAIL"
            print(f"{pop:<6} {seed:<6} {status:<6} {msg}")
            all_pass &= ok
    print("-" * 70)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--populations", nargs="+", default=["nfe", "elgh"])
    args = parser.parse_args()
    sys.exit(main(args.populations))
