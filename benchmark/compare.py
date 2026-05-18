"""Compare current package predictions against frozen benchmark/reference/ tensors.

Run after every refactor phase. Pass = all (pop, seed) pairs match within tolerance.
Exit 0 on pass, 1 on fail.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Tolerances per design spec §2
TOL_PRED_ABS = 1e-5
TOL_ATTN_ABS = 1e-5
TOL_ZVAR_REL = 1e-3

CHECKPOINTS = {
    "nfe": [42, 85, 482, 589, 612],
    "sas": [7, 32, 57, 64, 482],
}


def _run_inference(population: str, seed: int) -> dict:
    """Run inference via the public SDK and return {gene_id: payload}."""
    from varformer import Varformer
    input_genes = (REPO / "benchmark" / "inputs" / f"{population}_genes.txt").read_text().splitlines()
    model = Varformer.from_pretrained(population, seed=seed)
    return model.predict(genes=input_genes, return_attention=True)


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
    parser.add_argument("--populations", nargs="+", default=["nfe", "sas"])
    args = parser.parse_args()
    sys.exit(main(args.populations))
