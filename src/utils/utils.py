"""Shim — utilities split into varformer.utils.* and varformer.data.splits.

Delete in Phase 8.
"""
from varformer.utils.seeding import random_seed_context, set_seed  # noqa: F401
from varformer.utils.aa_codes import aa_to_idx, three_letter_aa_to_idx, aa1_to_aa3  # noqa: F401
from varformer.utils.gene_id import map_gene_names, get_protein_length  # noqa: F401
from varformer.data.splits import (  # noqa: F401
    load_fda_labels,
    load_combined_labels,
    get_labels,
    combine_features_and_labels,
)
