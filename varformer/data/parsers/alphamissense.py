"""AlphaMissense data merging with population exome data."""
from __future__ import annotations

from typing import Optional

import pandas as pd


AM_COLUMNS = ['#CHROM', 'POS', 'REF', 'ALT', 'protein_variant', 'am_pathogenicity']


def _matching_alphamissense_rows(path: str, variant_ids: set[str], variant_type: str):
    """Stream an AlphaMissense TSV and retain only population variants."""
    matches = []
    for chunk in pd.read_csv(
        path,
        sep='\t',
        skiprows=3,
        usecols=AM_COLUMNS,
        chunksize=1_000_000,
        dtype={'#CHROM': 'string', 'REF': 'string', 'ALT': 'string',
               'protein_variant': 'string', 'am_pathogenicity': 'float32'},
    ):
        chunk['variant_id'] = (
            chunk['#CHROM'] + '_' + chunk['POS'].astype(str) + '_' +
            chunk['REF'] + '_' + chunk['ALT'] + '_' + chunk['protein_variant']
        )
        chunk = chunk.loc[
            chunk['variant_id'].isin(variant_ids), ['am_pathogenicity', 'variant_id']
        ]
        if not chunk.empty:
            chunk['variant_type'] = variant_type
            matches.append(chunk)
    if not matches:
        return pd.DataFrame(columns=['am_pathogenicity', 'variant_id', 'variant_type'])
    return pd.concat(matches, ignore_index=True)


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
    3. Relative fall-back ``../data/alphamissense/...`` when neither explicit
       paths nor a config are provided.

    Args:
        pop_df: Population exome DataFrame (columns: CHROM, POS, REF, ALT,
                Amino_acids, Protein_position, …).
        pop: Population identifier, e.g. ``"sas"``, ``"nfe"``, ``"afr"``, ``"amr"``.
        am_path_iso: Path to AlphaMissense isoforms TSV
                     (``AlphaMissense_isoforms_hg38.tsv.gz``).
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
        am_path_iso = '../data/alphamissense/AlphaMissense_isoforms_hg38.tsv.gz'
    if am_path_can is None:
        am_path_can = '../data/alphamissense/AlphaMissense_hg38.tsv.gz'

    pop_df[['ref_aa', 'alt_aa']] = pop_df['Amino_acids'].str.split('/', expand=True)
    pop_df['protein_variant'] = (pop_df['ref_aa'] + pop_df['Protein_position'].astype(str) +
                                 pop_df['alt_aa'])
    pop_df['variant_id'] = (pop_df['CHROM'] + '_' + pop_df['POS'].astype(str) + '_' + pop_df['REF'] + '_' +
                            pop_df['ALT'] + '_' + pop_df['protein_variant'])
    pop_df = pop_df.drop_duplicates(subset=['variant_id'])
    variant_ids = set(pop_df['variant_id'])

    # AlphaMissense's isoform file takes precedence, matching the historical
    # behavior. Chunking keeps memory proportional to the population extract.
    am_iso = _matching_alphamissense_rows(
        am_path_iso, variant_ids, "non-canonical isoform"
    )
    am_can = _matching_alphamissense_rows(
        am_path_can, variant_ids, "canonical isoform"
    )
    am = pd.concat([am_iso, am_can], ignore_index=True).drop_duplicates(
        subset=['variant_id'], keep='first'
    )
    pop_df = pop_df.merge(am, on='variant_id', how='left')
    pop_df = pop_df[pop_df['am_pathogenicity'].notna()]

    return pop_df
