"""Label-loading helpers for drug-target splits."""
from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Optional

import pandas as pd

from varformer.config import Config


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
