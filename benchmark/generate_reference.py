"""Run inference on the frozen gene list with each checkpoint and save reference outputs.

Run ONCE at the start of the refactor; never re-run.
Imports from src/ code (pre-refactor semantics — frozen at Phase 0).
WARNING: src/ has been deleted; this script is intentionally frozen and cannot be
re-run. It exists only for audit purposes. Reference outputs are in benchmark/reference/.
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dataloader import ModuleDataProcessor
from models.lightning import MultiModalLightningTargetIdentifier
from preprocessing import ModelPreprocessorInference


CHECKPOINTS = {
    "nfe": [42, 85, 482, 589, 612],
    "elgh": [7, 32, 57, 64, 482],
}


def _find_checkpoint(ckpt_dir: Path, seed: int) -> Path:
    matches = list(ckpt_dir.glob(f"seed{seed}-epoch=*-val_spearman=*.ckpt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 checkpoint for seed {seed} in {ckpt_dir}, got {matches}"
        )
    return matches[0]


def _load_input_genes(population: str) -> set:
    path = REPO / "benchmark" / "inputs" / f"{population}_genes.txt"
    return set(path.read_text().splitlines())


def generate_for_population(population: str) -> None:
    config_path = REPO / "src" / f"cluster_config_{population}.yml"
    with config_path.open() as f:
        config = yaml.safe_load(f)
    config["hyperparameters"]["population"] = population
    config["hyperparameters"]["return_attn"] = True

    print(f"[{population}] loading data...")
    data = ModuleDataProcessor(gc=True, go=True, pvc=True, psc=False, config=config).process()
    input_genes = _load_input_genes(population)

    # process() in mode='inference' returns a LIST of split dicts; each split has
    # train/test partition of the same common_genes. Concatenating one split's
    # train+test reconstructs the full gene set for the loader.
    import pandas as pd

    splits = data if isinstance(data, list) else [data]
    first = splits[0]

    consolidated_data = {
        "gc": pd.concat([first["train"]["gc"], first["test_data"]["gc"]]),
        "go": pd.concat([first["train"]["go"], first["test_data"]["go"]]),
    }
    consolidated_pvc = {**first["train"]["pvc"], **first["test_data"]["pvc"]}
    consolidated_pvc.pop("labels", None)  # 'labels' is sibling metadata, not a gene entry

    test_loaders = ModelPreprocessorInference.create_test_loaders(
        config=config,
        consolidated_data=consolidated_data,
        pvc_data=consolidated_pvc,
        torch_dtype=config["hyperparameters"]["precision"],
    )

    # Dimensions for model loading
    with open(config["paths"]["MISSENSE_MAP"], "rb") as f:
        missense_map = pickle.load(f)
    num_mutations = len(missense_map)

    # In inference mode, "labels" is the full {gene: label} map of common_genes
    num_genes = len(first["labels"])
    num_features_gc = first["train"]["gc"].shape[1] - (1 if "target" in first["train"]["gc"].columns else 0)
    num_features_go = first["train"]["go"].shape[1] - (1 if "target" in first["train"]["go"].columns else 0)

    ckpt_dir = Path(config["paths"]["CKPT_PATH"]) / population
    output_dir = REPO / "benchmark" / "reference" / population
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in CHECKPOINTS[population]:
        ckpt_path = _find_checkpoint(ckpt_dir, seed)
        print(f"[{population} seed={seed}] loading {ckpt_path.name}")

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
        for loader_name, loader in test_loaders.items():
            batch_results = trainer.predict(model=model, dataloaders=loader)
            for batch in batch_results:
                for gid, payload in batch.items():
                    if gid in input_genes:
                        results[gid] = payload

        out_path = output_dir / f"seed{seed}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(results, f)
        print(f"[{population} seed={seed}] wrote {len(results)} predictions → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--populations", nargs="+", default=["nfe", "elgh"])
    args = parser.parse_args()
    for pop in args.populations:
        generate_for_population(pop)
