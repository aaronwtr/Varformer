"""Label-loading helpers for drug-target splits.

Extracted from src/utils/utils.py (Phase 6 refactor).
"""
import pickle as pkl

import pandas as pd


def load_fda_labels() -> pd.DataFrame:
    """Load the FDA-approved drug-target Excel sheet."""
    return pd.read_excel("../data/FDA_approved_drug_targets_2023_Q3.xlsx")


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
