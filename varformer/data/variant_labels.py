"""Per-variant frequency-proxy labels, aligned to each gene's token order."""
import hashlib
import os
import pickle as pkl

import pandas as pd
import torch

from varformer.data.features.variants import resolve_variant_features_dir

ENV_PATH = "VARFORMER_VARIANT_LABELS"
STRATA = ("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
PROBE_FRACTION = 0.15
PROBE_SALT = "variant-head-probe-v1"


def resolve_path(hparams=None):
    path = (hparams or {}).get("variant_labels_path") or os.environ.get(ENV_PATH)
    return path or None


def is_probe_gene(gene, fraction=PROBE_FRACTION, salt=PROBE_SALT):
    """Stable across arms, seeds and runs so every model sees the same probe set."""
    digest = hashlib.sha256(f"{salt}:{gene}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def load_variant_labels(config, population):
    """{gene: (labels, in_loss, probe, strata)} row-aligned with the gene's tokens.

    ``in_loss`` marks labelled tokens the variant loss may use. ``probe`` marks
    labelled tokens on withheld genes, used for evaluation only. The two are
    disjoint by construction.
    """
    try:
        hparams = config["hyperparameters"]
    except (KeyError, TypeError) as error:
        raise ValueError("Config lacks hyperparameters") from error
    path = resolve_path(hparams)
    if path is None:
        return None

    features_dir = resolve_variant_features_dir(config, population)
    with open(f"{features_dir}/gene_loc_var_map.pkl", "rb") as handle:
        gene_map = pkl.load(handle)

    frame = pd.read_parquet(path, columns=["variant_id", "label", "observed", "stratum"])
    frame = frame[frame.observed.astype(bool)]
    codes = {name: index for index, name in enumerate(STRATA)}
    lookup = {
        v: (float(lab), codes.get(str(st), -1))
        for v, lab, st in zip(frame.variant_id.to_numpy(), frame.label.to_numpy(float),
                              frame.stratum.to_numpy())
    }

    aligned = {}
    for gene, variants in gene_map.items():
        hits = [lookup.get(v) for v in variants]
        labelled = torch.tensor([h is not None for h in hits], dtype=torch.bool)
        labels = torch.tensor([0.0 if h is None else h[0] for h in hits])
        strata = torch.tensor([-1 if h is None else h[1] for h in hits], dtype=torch.long)
        probe = labelled if is_probe_gene(gene) else torch.zeros_like(labelled)
        in_loss = labelled & ~probe
        aligned[gene] = (labels, in_loss, probe, strata)
    return aligned
