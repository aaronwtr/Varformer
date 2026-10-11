"""Label-loading and frozen-partition helpers for drug-target splits."""
from __future__ import annotations

import hashlib
import json
import pickle as pkl
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pandas as pd

from varformer.config import Config


def gene_set_sha256(genes: Sequence[str]) -> str:
    """Return the order-independent digest used by training split manifests."""
    payload = "\n".join(sorted(map(str, genes))) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a split source without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_frozen_split(
    path: str | Path,
    *,
    population: str,
    seed: int,
    available_train_genes: Sequence[str],
    test_genes_by_task: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[str], dict]:
    """Load an exact donor split and prove it fits the materialized data.

    This is intentionally stricter than reconstructing a split from a numeric
    seed. A matched-control run must consume the donor's exact train and
    validation gene lists, while the already materialized benchmark tasks must
    have the same genes. Any population, seed, universe, task, count, or digest
    mismatch fails before model construction and training.
    """
    source = Path(path)
    payload = json.loads(source.read_text())
    if str(payload.get("population")) != str(population):
        raise ValueError(
            "Frozen split population mismatch: "
            f"expected {population}, observed {payload.get('population')}"
        )
    try:
        observed_seed = int(payload.get("seed"))
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen split has no valid numeric seed") from error
    if observed_seed != int(seed):
        raise ValueError(
            f"Frozen split seed mismatch: expected {seed}, observed {observed_seed}"
        )

    train = list(map(str, payload.get("train_genes", [])))
    validation = list(map(str, payload.get("validation_genes", [])))
    frozen_tests_raw = payload.get("test_genes_by_task")
    if not train or not validation or not isinstance(frozen_tests_raw, dict):
        raise ValueError("Frozen split is missing train, validation, or test genes")
    frozen_tests = {
        str(task): list(map(str, genes))
        for task, genes in frozen_tests_raw.items()
    }
    current_tests = {
        str(task): list(map(str, genes))
        for task, genes in test_genes_by_task.items()
    }
    if set(frozen_tests) != set(current_tests):
        raise ValueError(
            "Frozen split benchmark-task mismatch: "
            f"expected {sorted(current_tests)}, observed {sorted(frozen_tests)}"
        )

    named_lists = {"train": train, "validation": validation, **{
        f"test/{task}": genes for task, genes in frozen_tests.items()
    }}
    duplicates = {
        name: len(genes) - len(set(genes))
        for name, genes in named_lists.items()
        if len(genes) != len(set(genes))
    }
    if duplicates:
        raise ValueError(f"Frozen split contains duplicate genes: {duplicates}")

    train_set = set(train)
    validation_set = set(validation)
    test_union = set().union(*(set(genes) for genes in frozen_tests.values()))
    intersections = {
        "train_validation": train_set & validation_set,
        "train_test": train_set & test_union,
        "validation_test": validation_set & test_union,
    }
    if any(intersections.values()):
        raise ValueError(
            "Frozen split contains gene-level leakage: "
            + ", ".join(f"{key}={len(value)}" for key, value in intersections.items())
        )

    available_set = set(map(str, available_train_genes))
    frozen_train_validation = train_set | validation_set
    if frozen_train_validation != available_set:
        raise ValueError(
            "Frozen split train/validation universe mismatch: "
            f"missing={len(available_set-frozen_train_validation)}, "
            f"extra={len(frozen_train_validation-available_set)}"
        )
    task_differences = {
        task: {
            "missing": len(set(current_tests[task]) - set(frozen_tests[task])),
            "extra": len(set(frozen_tests[task]) - set(current_tests[task])),
        }
        for task in sorted(current_tests)
        if set(frozen_tests[task]) != set(current_tests[task])
    }
    if task_differences:
        raise ValueError(f"Frozen split benchmark-gene mismatch: {task_differences}")

    expected_hashes = payload.get("set_sha256")
    observed_hashes = {
        "train": gene_set_sha256(train),
        "validation": gene_set_sha256(validation),
        "test_union": gene_set_sha256(test_union),
        "test_by_task": {
            task: gene_set_sha256(genes) for task, genes in frozen_tests.items()
        },
    }
    if expected_hashes != observed_hashes:
        raise ValueError("Frozen split stored set hashes do not match its gene lists")

    expected_counts = payload.get("counts")
    observed_counts = {
        "train": len(train),
        "validation": len(validation),
        "test_union": len(test_union),
        "test_by_task": {task: len(genes) for task, genes in frozen_tests.items()},
    }
    if expected_counts != observed_counts:
        raise ValueError("Frozen split stored counts do not match its gene lists")

    provenance = {
        "path": str(source.resolve()),
        "sha256": file_sha256(source),
        "set_sha256": observed_hashes,
        "population": str(population),
        "seed": int(seed),
        "exact_reuse_verified": True,
    }
    return train, validation, provenance


def load_fda_labels(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the FDA-approved drug-target sheet.

    By default the location is derived from the active configuration profile;
    callers may pass an explicit path for a different data release.
    """
    labels_path = path or Config.load()["paths"]["FDA_LABELS"]
    return pd.read_excel(labels_path)


def load_combined_labels(ot_targets: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Load HPA/manual FDA and Citeline labels, combine with Open Targets ChEMBL data.

    Parameters
    ----------
    ot_targets:
        Open Targets target table containing a 'targetId' and 'target' column.
    config:
        Config dict with ``config['paths']['CITELINE_LABELS']`` pointing to the
        pickled Citeline labels file.

    Returns
    -------
    pd.DataFrame
        Combined label table with columns ``Ensembl`` and ``Status``.
    """
    with open(config['paths']['CITELINE_LABELS'], "rb") as f:
        labels = pkl.load(f)
    labels = labels.drop(columns=["Gene"])
    if not config["hyperparameters"].get("include_open_targets_labels", True):
        return labels
    labels_ensembl = labels["Ensembl"].tolist()
    new_labels = pd.DataFrame(
        [
            {"Ensembl": target, "Status": "Launched"}
            for target in ot_targets[ot_targets['target'] == 1]['targetId']
            if target not in labels_ensembl
        ]
    )
    labels = pd.concat([labels, new_labels], ignore_index=True)
    return labels


def get_labels(gene_names: list, target: pd.DataFrame) -> dict:
    """Build a binary label dict: 1 if the gene is a known drug target, 0 otherwise.

    Parameters
    ----------
    gene_names:
        List of Ensembl gene IDs to label.
    target:
        DataFrame with an ``Ensembl`` column listing positive (target) genes.

    Returns
    -------
    dict
        Mapping ``{gene_id: 0 | 1}``.
    """
    target_genes = list(target["Ensembl"])
    labels = {}
    for gene in gene_names:
        labels[gene] = 1 if gene in target_genes else 0
    return labels


def combine_features_and_labels(
    gene_names: pd.Series,
    features: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Add a binary 'target' column to *features* based on membership in *target*.

    Parameters
    ----------
    gene_names:
        Series of Ensembl gene IDs aligned with ``features``.
    features:
        Feature DataFrame to annotate in-place.
    target:
        DataFrame with an ``Ensembl`` column listing positive genes.

    Returns
    -------
    pd.DataFrame
        The annotated *features* DataFrame (modified in-place and returned).
    """
    target_genes = list(target["Ensembl"])
    features["target"] = 0
    features.loc[gene_names.isin(target_genes), "target"] = 1
    return features
