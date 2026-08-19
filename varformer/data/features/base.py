"""Shared population-data + holdout-gene loading, factored out of GeneCharacterisationPreprocessor.

GCP inherits BaseFeatures; GOP and PVCP compose a GCP instance (no inheritance).
"""
from __future__ import annotations

import os
import pickle as pkl
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pol


class BaseFeatures:
    """Population data + holdout genes — shared by GC/GO/PVC."""

    def __init__(self, config: Any):
        self.config = config
        self.population = self.config['hyperparameters']['population']

        # Get all holdout_genes
        self.get_holdout_genes()

        # check if features dir exists; if not, create it
        if not os.path.exists(f"{self.config['paths']['FEATURES_DIR']}/{self.population}/"):
            os.makedirs(f"{self.config['paths']['FEATURES_DIR']}/{self.population}/")

        # Load population exome data
        self.pop_data = self.load_pop_data()
        if isinstance(self.pop_data, pol.DataFrame):
            self.pop_data = self.pop_data.to_pandas()

        # check if the SWISSPROT and TREMBL columns are present, if they are, skip this bit
        if 'SWISSPROT' in self.pop_data.columns or 'TREMBL' in self.pop_data.columns:
            self.pop_data["UNIPROT"] = self.pop_data["SWISSPROT"].fillna(self.pop_data["TREMBL"])
            self.pop_data = self.pop_data.drop(["SWISSPROT", "TREMBL"], axis=1)
        self.pop_data[['ref_aa', 'alt_aa']] = self.pop_data['Amino_acids'].str.split('/', expand=True)
        self.pop_data['protein_variant'] = (self.pop_data['ref_aa'] + self.pop_data['Protein_position'].astype(str) +
                                            self.pop_data['alt_aa'])
        self.pop_data['variant_id'] = (self.pop_data['CHROM'] + '_' + self.pop_data['POS'].astype(str) + '_' +
                                       self.pop_data['REF'] + '_' + self.pop_data['ALT'] + '_' +
                                       self.pop_data['protein_variant'])

        self.pop_data = self.pop_data.drop_duplicates(subset=['variant_id'])

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_pop_data(self):
        """
        Load population-exome data. Assumed the data is stored as <pop_id>_exomes_filtered.pkl.
        pop_ids that are supported are: 'sas', 'amr', 'afr' and 'nfe'.
        """
        assert self.population in ['sas', 'amr', 'afr', 'nfe'], (
            "Population must be one of: 'sas', 'amr', 'afr', or 'nfe'.")
        pop_path = self.config['paths']['POP_DATA'] + f'{self.population}_exomes_filtered.pkl'
        if not os.path.exists(pop_path):
            pop_data = self.filter_raw_exomes()
            return pop_data
        else:
            with open(pop_path, "rb") as f:
                pop_data = pkl.load(f)
            return pop_data

    def filter_raw_exomes(self):
        """
        Load the normalized gnomAD extract for a population.

        Current extracts already contain the stable Varformer interchange
        schema, so this path deliberately does not infer columns from the
        Genes & Health data (which is SAS-specific).
        """
        root = Path(self.config['paths']['GNOMAD_DATA'])
        candidates = [
            root / f'gnomad_exomes_{self.population}.parquet',
            root / 'v4.1' / f'gnomad_exomes_{self.population}.parquet',
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No gnomAD extract found for {self.population}; checked {candidates}"
            )

        parquet_source = str(path / '*.parquet') if path.is_dir() else str(path)
        variants = pol.read_parquet(parquet_source)

        rename_mapping = {
            'chrom': 'CHROM',
            'pos': 'POS',
            'ref': 'REF',
            'alt': 'ALT'
        }

        existing_renames = {k: v for k, v in rename_mapping.items() if k in variants.columns}

        if existing_renames:
            variants = variants.rename(existing_renames)

        variants = variants.rename(
            {col: col.replace('vep_', '') for col in variants.columns if col.startswith('vep_')})

        required = [
            'CHROM', 'POS', 'REF', 'ALT', 'Gene', 'SYMBOL', 'Consequence',
            'Protein_position', 'Amino_acids', 'HGVSp', 'CANONICAL',
            'MANE_SELECT', 'population_AF', 'population_AC', 'population_AN',
        ]
        missing = [column for column in required if column not in variants.columns]
        if missing:
            raise ValueError(f"gnomAD extract is missing required columns: {missing}")
        variants = variants.select(required)

        return variants

    def get_holdout_genes(self):
        """Load holdout gene sets (Pfam, FDA-approved, Pharos)."""
        # Pfam targets
        pfam_raw = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='pfam_drgbl')
        ensg_pfam = pfam_raw['ENSG'].tolist()
        self.drgbl_targets_pfam = ensg_pfam

        # Recently approved targets
        rcnt_app_raw = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='rcnt_app_targets')
        rcnt_app_genes = rcnt_app_raw['ENSG'].tolist()
        self.rcnt_targets_fda = rcnt_app_genes

        # Pharos targets
        chem_targets_pharos = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='chem_targets')
        chem_targets_pharos_genes = chem_targets_pharos['ENSG'].tolist()
        self.chem_targets_pharos = chem_targets_pharos_genes
