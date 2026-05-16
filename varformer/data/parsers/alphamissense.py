"""merge_am_data — moved from src/utils/merge_am_data.py (Phase 4A)."""
import pandas as pd
import polars as pol


def merge_am_data(pop_df: pd.DataFrame, pop: str):
    # TODO: make this compatible with config as to not hardcode paths
    # path_1 = '/data/scratch/bty174/genomic-drug-targeting/data/alphamissense/AlphaMissense_isoforms_hg38.tsv'
    # path_2 = '/data/scratch/bty174/genomic-drug-targeting/data/alphamissense/AlphaMissense_hg38.tsv'

    path_1 = '../data/alphamissense/AlphaMissense_isoforms_hg38.tsv'
    path_2 = '../data/alphamissense/AlphaMissense_hg38.tsv'

    # Load
    am_iso = pd.read_csv(path_1, sep='\t', skiprows=3)
    am_can = pd.read_csv(path_2, sep='\t')

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

    # pop_dir = f'/data/scratch/bty174/genomic-drug-targeting/data/alphamissense/{pop}'
    # if not os.path.exists(pop_dir):
    #     os.makedirs(pop_dir)
    # pop_df.to_parquet(pop_dir + f'{pop}_am_data_full.parquet', engine='pyarrow', compression='snappy')

    return pop_df
