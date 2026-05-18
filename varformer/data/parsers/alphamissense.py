"""AlphaMissense data merging with population exome data."""
from __future__ import annotations

from typing import Optional

import pandas as pd


def merge_am_data(
    pop_df: pd.DataFrame,
    pop: str,
    *,
    am_path_iso: Optional[str] = None,
    am_path_can: Optional[str] = None,
    config=None,
) -> pd.DataFrame:
    """Merge AlphaMissense pathogenicity scores into a population variant DataFrame.

    Paths are resolved in this priority order:
    1. Explicit ``am_path_iso`` / ``am_path_can`` keyword arguments.
    2. ``config['paths']['AM_PATH_ISO']`` / ``config['paths']['AM_PATH_CAN']``
       (any mapping that supports key access, e.g. a ``varformer.config.Config``).
    3. Relative fall-back ``../data/alphamissense/...`` for backwards-compat.

    Args:
        pop_df: Population exome DataFrame (columns: CHROM, POS, REF, ALT,
                Amino_acids, Protein_position, …).
        pop: Population identifier, e.g. ``"sas"``, ``"nfe"``, ``"afr"``, ``"amr"``.
        am_path_iso: Path to AlphaMissense isoforms TSV
                     (``AlphaMissense_isoforms_hg38.tsv``).
        am_path_can: Path to AlphaMissense canonical TSV
                     (``AlphaMissense_hg38.tsv``).
        config: Optional config object whose ``config['paths']['AM_PATH_ISO']``
                and ``config['paths']['AM_PATH_CAN']`` entries are used when the
                explicit path arguments are not provided.

    Returns:
        Merged DataFrame with ``am_pathogenicity`` column, filtered to rows
        where the score is not NaN.
    """
    if am_path_iso is None and config is not None:
        am_path_iso = config['paths']['AM_PATH_ISO']
    if am_path_can is None and config is not None:
        am_path_can = config['paths']['AM_PATH_CAN']

    if am_path_iso is None:
        am_path_iso = '../data/alphamissense/AlphaMissense_isoforms_hg38.tsv'
    if am_path_can is None:
        am_path_can = '../data/alphamissense/AlphaMissense_hg38.tsv'

    # Load
    am_iso = pd.read_csv(am_path_iso, sep='\t', skiprows=3)
    am_can = pd.read_csv(am_path_can, sep='\t')

    # Construct variant_id
    for df in [am_iso, am_can]:
        df['variant_id'] = (df['#CHROM'] + '_' + df['POS'].astype(str) + '_' +
                            df['REF'] + '_' + df['ALT'] + '_' + df['protein_variant'])

    # Reduce columns and tag source
    am_iso = am_iso[['am_pathogenicity', 'variant_id']].copy()
    am_iso['variant_type'] = "non-canonical isoform"

    am_can = am_can[['am_pathogenicity', 'variant_id']].copy()
    am_can['variant_type'] = "canonical isoform"

    # Filter only new rows from canonical set
    new_rows = am_can[~am_can['variant_id'].isin(am_iso['variant_id'])]

    # Concatenate
    am = pd.concat([am_iso, new_rows], ignore_index=True)

    pop_df[['ref_aa', 'alt_aa']] = pop_df['Amino_acids'].str.split('/', expand=True)
    pop_df['protein_variant'] = (pop_df['ref_aa'] + pop_df['Protein_position'].astype(str) +
                                 pop_df['alt_aa'])
    pop_df['variant_id'] = (pop_df['CHROM'] + '_' + pop_df['POS'].astype(str) + '_' + pop_df['REF'] + '_' +
                            pop_df['ALT'] + '_' + pop_df['protein_variant'])
    pop_df = pop_df.drop_duplicates(subset=['variant_id'])
    pop_df = pop_df.merge(am, on='variant_id', how='left')
    pop_df = pop_df[pop_df['am_pathogenicity'].notna()]

    return pop_df
