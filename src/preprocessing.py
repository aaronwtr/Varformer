import os
import gc
import warnings
import argparse

import matplotlib.pyplot as plt
import torch
import utils
import requests
import time
import esm
import bz2

import pytorch_lightning as pl
import polars as pol
import pickle as pkl
import gzip
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import dataloader as dl

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO
from torch.utils.data import DataLoader
from shutil import copyfileobj

from utils.utils import (load_combined_labels, combine_features_and_labels, aa_to_idx, three_letter_aa_to_idx,
                         map_gene_names)
from utils.merge_am_data import merge_am_data
# from utils.preprocessing import featurise
# from models.target_identifier import MultiModalTargetIdentifier
from models.lightning import MultiModalLightningTargetIdentifier


# data preprocessing
class GeneCharacterisationPreprocessor:
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self, config):
        print("Gene Characterisation Preprocessor is booting up...")
        self.config = config
        self.population = self.config['hyperparameters']['population']
        self.files_and_dirs = os.listdir(self.config['paths']['DATA_DIR'])
        self.data_name_mapping = {
            "CTD_chem_gene_ixns.csv": "CTD Chemical-Gene Interactions",
            "gnomad.exomes.v2.1.1.lof_metrics.by_gene.csv": "gnomAD Exomes Loss-of-Function Metrics",
            "9606.protein.links.full.v12.0.txt": "STRING Protein-Protein Interactions",
            "part-00000-31eba8be-aff8-492e-9edb-4b5e8c821237-c000.snappy.parquet": "Mouse Knockout Phenotypes"
        }
        self.files = self._get_files()
        self.datasets = self.load_data()

        self.chem_features = None
        self.gnomad_features = None
        self.mouse_ko_features = None
        self.gene_essentiality_features = None
        self.ppi_features = None

        self.drgbl_targets_pfam = None
        self.rcnt_targets_fda = None
        self.chem_targets_pharos = None

        # Get all holdout_genes
        self.get_holdout_genes()

        features_dir = self.config['paths']['FEATURES_DIR']
        population = self.config['hyperparameters']['population']

        # check if features dir exists if not create it
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

        # Load raw G&H missense variant data
        if self.population == 'elgh':
            if not os.path.exists(self.config['paths']['RAW_GH']):
                miva_feature_matrix = self.pop_data[self.pop_data['Consequence'] == 'missense_variant']
                miva_feature_matrix = miva_feature_matrix[["Gene", "UNIPROT", "variant_id"]]
                miva_feature_matrix = miva_feature_matrix.rename(columns={"Gene": "ENSG"})
                miva_feature_matrix = miva_feature_matrix.drop_duplicates(subset="ENSG")
                miva_feature_matrix.to_pickle(self.config['paths']['RAW_GH'])

        feature_extractors = {
            #   'chem_features.pkl': self.chem_feature_extractor,
            #   'gnomad_features.pkl': self.gnomad_feature_extractor,
            #   'mouse_ko_features.pkl': self.mouse_knockout_feature_extractor,
            'gene_essentiality_features.pkl': self.gene_essentiality_feature_extractor,
            'ppi_features.pkl': self.ppi_feature_extractor,
        }

        self.features = self.load_opentargets_features()

        self.features = self.features[self.features['targetId'].isin(self.pop_data['Gene'])]
        nan_percentages = self.features.isna().mean() * 100
        high_nan_features = nan_percentages[nan_percentages > 99].index.tolist()
        if high_nan_features:
            print(f"Removing features with only NaN values: {high_nan_features}")
            self.features = self.features.drop(columns=high_nan_features)

        # check feature statistics in the features attribute here

        tissue_columns = [col for col in self.features.columns if 'tissueDistribution' in col]
        for col in tissue_columns:
            self.features[col] = self.features[col].fillna(-0.5)

        self.features = self.features.fillna(0)

        self.ensg_ids = self.features["targetId"]
        # self.features, self.ensg_ids, self.uniprot_ids = featurise(ensg_features)
        # self.norm = True

        # Ground truth
        self.features = self.features.rename(columns={'maxClinicalTrialPhase': 'target'})
        self.features['target'] = self.features['target'].apply(lambda x: 1.0 if x >= 0.75 else 0.0)
        self.features = self.features[[col for col in self.features if col != 'target'] + ['target']]
        self.ot_targets = self.features[['targetId', 'target']]
        self.target = load_combined_labels(self.ot_targets, self.config)

        # Combine features and target
        self.labels_dict = utils.utils.get_labels(self.ensg_ids, self.target)
        self.full_data = combine_features_and_labels(self.ensg_ids, self.features, self.target)
        self.full_data.set_index('targetId', inplace=True)
        # feature statistics can be checked here!

        self.ce_data = self.full_data
        self.num_features = len(self.full_data.columns) - 1

        self.data = self.full_data
        self.labels = self.labels_dict
        # ensg_to_symb_df = self.pop_data[['Gene', 'SYMBOL']]
        # ensg_to_symb_dict = ensg_to_symb_df.set_index('Gene')['SYMBOL'].to_dict()
        # label_df = pd.DataFrame.from_dict(self.labels, orient='index', columns=['label'])
        # label_df['symbol'] = label_df.index.map(ensg_to_symb_dict)
        # label_df.index.name = 'ensg_id'
        # label_df = label_df[['symbol', 'label']]
        # label_df.to_pickle("../data/labels/processed_labels.pkl")
        # print('break')

        # Create population directory if it doesn't exist
        os.makedirs(f'{features_dir}/{population}', exist_ok=True)

        # Save the GC features (self.data contains the final feature matrix)
        gc_features_path = f'{features_dir}/{population}/gene_characterisation_features.pkl'
        with open(gc_features_path, 'wb') as f:
            pkl.dump(self.data, f)

    def _get_files(self):
        """
        Get the files from the data directory.
        """
        files = []
        populations = ['elgh', 'amr', 'nfe']
        populations = [pop for pop in populations if pop != self.population]
        exclude = ['.DS_Store', 'elgh', 'clinvar', 'VariPred', 'string_data_counts.pkl', 'gnomad_data'] + populations
        data_dir = self.config['paths']['DATA_DIR']

        for file in self.files_and_dirs:
            if "." in file and file not in exclude:
                file_path = f"{data_dir}/{file}"
                # Check if path contains population1/population2 pattern
                if not self._contains_population_pattern(file_path, populations):
                    files.append(file_path)
            elif file not in exclude:
                file_path = f"{data_dir}/{file}"
                # Check if the directory path itself contains population pattern before parsing
                if not self._contains_population_pattern(file_path, populations):
                    _file = self._safe_dir_parser(file_path)
                    # Only process if dir_parser returned a valid file path
                    if _file is not None:
                        # Check if parsed file path contains population1/population2 pattern
                        if not self._contains_population_pattern(_file, populations):
                            files.append(_file)

        return files

    def _contains_population_pattern(self, file_path, populations):
        """
        Check if file path contains any population1/population2 pattern.

        Args:
            file_path (str): The file path to check
            populations (list): List of population codes to check against

        Returns:
            bool: True if path contains population1/population2 pattern, False otherwise
        """
        # Handle None file_path
        if file_path is None:
            return False

        # Get all population codes including the current one
        all_populations = ['elgh', 'amr', 'nfe']

        # Check for any combination of population1/population2 in the path
        for pop1 in all_populations:
            for pop2 in all_populations:
                if pop1 != pop2:  # Don't check pop1/pop1 patterns
                    pattern = f"{pop1}/{pop2}"
                    if pattern in file_path:
                        return True

        return False

    def _safe_dir_parser(self, path):
        """
        Modified directory parser that avoids paths with population1/population2 patterns.

        Args:
            path (str): Directory path to parse

        Returns:
            str or None: File path if valid, None otherwise
        """
        import os

        # Check if current path contains population pattern
        if self._contains_population_pattern(path, []):
            return None

        if not os.path.exists(path):
            return None

        if os.path.isfile(path):
            return path

        try:
            subfiles = os.listdir(path)
            for subfile in subfiles:
                subpath = os.path.join(path, subfile)

                # Skip if subpath would create a population pattern
                if self._contains_population_pattern(subpath, []):
                    continue

                if os.path.isfile(subpath):
                    return subpath
                elif os.path.isdir(subpath):
                    result = self._safe_dir_parser(subpath)
                    if result is not None:
                        return result
        except (OSError, PermissionError):
            # Handle cases where directory cannot be accessed
            pass

        return None

    def _dir_parser(self, path):
        """
        Recursive algorithm to parse the directory to get the files and their paths.
        """
        exclude = ['.DS_Store', 'elgh', 'archive']
        subfiles = os.listdir(path)
        for subfile in subfiles:
            excl = '\t'.join(exclude)
            if "." in subfile and subfile not in excl:
                path = f"{path}/{subfile}"
                return f"{path}"
            elif subfile not in excl:
                path = f"{path}/{subfile}"
                self._dir_parser(path)

    def load_data(self):
        """
        Load the data from the files.
        """
        datasets = {}
        data_dir = self.config['paths']['DATA_DIR']
        if "datasets.pkl" in os.listdir(self.config['paths']['DATA_DIR']):
            with open(f'{data_dir}/datasets.pkl', 'rb') as fp:
                datasets = pkl.load(fp)
            return datasets
        else:
            for file in self.files:
                file_name = file.split("/")[-1]
                if file_name in self.data_name_mapping.keys():
                    file_id = self.data_name_mapping[file_name]
                    if any(word in file for word in ["csv", "txt"]):
                        if '9606' not in file:
                            datasets[file_id] = pd.read_csv(file)
                        else:
                            datasets[file_id] = pd.read_csv(file, sep=" ")
                    elif any(word in file for word in ["xlsx", "xlsb"]):
                        datasets[file_id] = pd.read_excel(file)
                    elif "parquet" in file:
                        datasets[file_id] = pd.read_parquet(file)
                    else:
                        raise ValueError(
                            "The file format is not supported. Make sure data is .csv, .txt, Excel, or parquet.")
            with open(f'{data_dir}/datasets.pkl', 'wb') as fp:
                pkl.dump(datasets, fp)
            return datasets

    def load_pop_data(self):
        """
        Load population-exome data. Assumed the data is stored as <pop_id>_exomes_filtered.pkl.
        pop_ids that are supported are: 'elgh', 'amr', 'afr' and 'nfe'.
        """
        assert self.population in ['elgh', 'amr', 'afr', 'nfe'], ("Population must be one of: 'elgh', 'amr', 'afr', or "
                                                                  "'nfe'.")
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
        Filters raw exome variant data for a specific population and saves the processed data.

        This function reads raw exome variant data from a specified Parquet file, processes
        the data by renaming columns, selecting relevant columns based on specific criteria,
        removing unnecessary prefixes, and filtering based on population-specific allele
        frequency. The processed data is then saved as a serialized Pickle file.

        :raises KeyError: If required configuration keys are missing in the `config` attribute.

        :param self: An instance of the class containing this method. The following attributes
            of the instance are used:

            - `config`: A dictionary containing configuration settings, including file paths
              under the "paths" key.
            - `population`: A string representing the population name to filter the data for.
            - `gh_data`: A DataFrame containing data against which the columns of the variant
              data are matched.

        :rtype: pandas.DataFrame
        :return: A Pandas DataFrame containing the filtered variant data.
        """
        # path = f"{self.config['paths']['GNOMAD_DATA']}gnomad_{self.population}_variants/gnomad_exomes_{self.population}.parquet"
        path = f"{self.config['paths']['GNOMAD_DATA']}gnomad_exomes_{self.population}.parquet"
        gh_data = pd.read_pickle(self.config['paths']['GH_CSQ'])

        variants = pol.read_parquet(path)

        rename_mapping = {
            'chrom': 'CHROM',
            'pos': 'POS',
            'ref': 'REF',
            'alt': 'ALT'
        }

        # Get columns that actually exist and need renaming
        existing_renames = {k: v for k, v in rename_mapping.items() if k in variants.columns}

        if existing_renames:
            variants = variants.rename(existing_renames)

        gh_columns = set(gh_data.columns.tolist())
        columns = variants.columns

        # Find all columns that either match exactly or have partial overlap
        selected_columns = []
        for col in columns:
            # Exact match or partial match
            if ('AF' or 'AC' or 'AN') in col:
                continue  # Skip AF-related columns except 'AF_<population>'
            elif (col in gh_columns or
                  any(gh_col in col or col in gh_col for gh_col in gh_columns)):
                selected_columns.append(col)
            else:
                continue

        if f'AF_{self.population}' not in selected_columns:
            selected_columns.append(f'AF_{self.population}')

        variants = variants.select(selected_columns)

        variants = variants.rename(
            {col: col.replace('vep_', '') for col in variants.columns if col.startswith('vep_')})
        variants.head()

        variants = variants.to_pandas()
        with open(f"{self.config['paths']['POP_DATA']}{self.population}_exomes_filtered.pkl", "wb") as f:
            pkl.dump(variants, f)

        return variants

    def load_ground_truth(self):
        """
        Load the ground truth data.
        """
        return self.datasets["FDA Approved Drug Targets"]

    def get_holdout_genes(self):
        # ACMG actionable genes
        # columns = ['disease', 'gene']
        # acmg_raw = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name=0)  # sheet 0 is acmg genes
        # acmg_raw.columns = columns
        # acmg_raw['gene'] = acmg_raw['gene'].apply(lambda x: x.replace(u'\xa0', u' '))
        # genes = acmg_raw['gene'].tolist()
        # genes = [gene.split(' ')[0] for gene in genes]
        # ensg_gene_map = utils.map_gene_names(genes, 'symb', 'ensg')
        # ensg_genes = list(ensg_gene_map.values())
        # ensg_genes = list(set(ensg_genes))
        # self.acmg_genes = ensg_genes

        # Pfam targets
        pfam_raw = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='pfam_drgbl')

        ensg_pfam = pfam_raw['ENSG'].tolist()
        self.drgbl_targets_pfam = ensg_pfam

        # Check held out test set for common essential genes
        # common_essentials = pd.read_csv(self.config['paths']['COMMON_ESSENTIALS_PATH'])
        # common_essentials = common_essentials.rename(columns={common_essentials.columns[0]: 'gene_name'})
        # common_essentials['gene_name'] = common_essentials['gene_name'].str.split(' ').str[0]
        # common_essentials_pfam = common_essentials[common_essentials['gene_name'].isin(ensg_pfam)]
        # print(f"There are {len(common_essentials_pfam)} common essential genes in the Pfam dataset. They are: "
        #       f"{common_essentials_pfam['gene_name'].tolist()}")

        # Recently approved targets
        rcnt_app_raw = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='rcnt_app_targets')
        rcnt_app_genes = rcnt_app_raw['ENSG'].tolist()
        self.rcnt_targets_fda = rcnt_app_genes

        # Pharos targets
        chem_targets_pharos = pd.read_excel(self.config['paths']['TEST_GENES_PATH'], sheet_name='chem_targets')
        chem_targets_pharos_genes = chem_targets_pharos['ENSG'].tolist()
        self.chem_targets_pharos = chem_targets_pharos_genes

    def load_opentargets_features(self):
        feature_path = self.config['paths']['OT_PATH']
        ot_df = pd.read_pickle(feature_path)
        cols_rm = ["isInMembrane", "isSecreted", "isCancerDriverGene", "tissueSpecificity"]
        cols = ot_df.columns
        cols_to_keep = [col for col in cols if col not in cols_rm]
        ot_df = ot_df[cols_to_keep]
        return ot_df

    def chem_feature_extractor(self):
        """
        Extract chemical features from the CTD dataset. We count the number of known chemical interactions for each gene
        Note: we can further disentangle this data based on interaction type, e.g. increasing or decreasing action of
        target. This is not yet implemented.
        """
        keys = list(self.datasets.keys())
        chem_data = self.datasets[keys[0]]
        chem_features = chem_data[["GeneSymbol", "# ChemicalName", "Organism", "InteractionActions"]]
        chem_features = chem_features[chem_features["Organism"] == "Homo sapiens"]
        gene_counts = chem_features["GeneSymbol"].value_counts()
        chem_features = pd.DataFrame({
            "symbol": gene_counts.index,
            "count": gene_counts.values,
        })

        gene_names = list(chem_features["symbol"])
        mapped_names = map_gene_names(gene_names, 'symb', 'ensg')
        chem_features['symbol'] = chem_features['symbol'].map(mapped_names)
        chem_features = chem_features[chem_features['symbol'] != 'N/A']

        # Transform DataFrame into a dictionary
        chem_features = chem_features.set_index('symbol')['count'].to_dict()
        self.chem_features = chem_features

        # NOTE: normalize AFTER train test split
        # scaler = MinMaxScaler()
        # chem_features["count"] = scaler.fit_transform(chem_features[["count"]])
        # chem_features = chem_features.set_index("symbol")["count"].to_dict()

    def gnomad_feature_extractor(self):
        """
        Extract target conservation scores from gnomAD data. Note that pLI measures the probability of a gene being
        loss-of-function intolerant for a particular variant. There are more potential features we can extract from the
        gnomAD data.
        """
        keys = list(self.datasets.keys())
        gnom_data = self.datasets[keys[1]]
        gnom_data_raw = gnom_data[["gene", "pLI"]]
        # fill the nans with 0.0 in gnoma_data_raw
        gnom_data_raw["pLI"] = gnom_data_raw["pLI"].fillna(0.0)
        gnom_data = gnom_data_raw
        gene_names = list(gnom_data["gene"])
        mapped_names = map_gene_names(gene_names, 'symb', 'ensg')
        gnom_data['gene'] = gnom_data['gene'].map(mapped_names)
        gnom_data = gnom_data[gnom_data['gene'] != 'N/A']
        gnom_data = gnom_data.set_index("gene")["pLI"].to_dict()

        self.gnomad_features = gnom_data

    def ppi_feature_extractor(self):
        """
        Featurise PPI data, i.e. count and normalize the PPIs for each PPI that is experimentally validated.
        """
        protein_info = []  # To store the parsed data
        ppi_data = self.config["paths"]["PPI_FEATURES"]
        with open(ppi_data, "r") as file:
            for line in file:
                fields = line.strip().split('\t')
                protein_info.append(fields)
        protein_info = pd.DataFrame(protein_info)
        protein_info.columns = protein_info.iloc[0]

        keys = list(self.datasets.keys())
        string_data_raw = self.datasets[keys[2]]
        string_data_raw = string_data_raw[["protein1", "protein2", "experiments"]]
        string_data_raw['experiments'] = string_data_raw['experiments'].astype(int)
        string_data_raw = string_data_raw[string_data_raw['experiments'] > 0]

        protein_names_1 = string_data_raw['protein1'].tolist()
        protein_names_2 = string_data_raw['protein2'].tolist()
        protein_names_1 = [protein.split('.')[1] for protein in protein_names_1]
        protein_names_2 = [protein.split('.')[1] for protein in protein_names_2]

        string_data_raw['protein1'] = protein_names_1
        string_data_raw['protein2'] = protein_names_2
        protein_names = list(set(protein_names_1))

        mapped_names = map_gene_names(protein_names, 'ensp', 'ensg')

        string_data_raw['protein1'] = string_data_raw['protein1'].map(mapped_names)
        string_data_raw['protein2'] = string_data_raw['protein2'].map(mapped_names)
        string_data_raw = string_data_raw[string_data_raw['protein1'] != 'N/A']
        string_data_raw = string_data_raw[string_data_raw['protein2'] != 'N/A']

        protein_counts = {}

        # note we only need to do this for one column, since both columns contain the same proteins
        for protein in string_data_raw['protein1']:
            if protein not in protein_counts:
                protein_counts[protein] = 1
            else:
                protein_counts[protein] += 1

        self.ppi_features = protein_counts

        # NOTE: normalize AFTER train test split
        # scaler = MinMaxScaler(feature_range=(0, 1))
        #
        # protein_counts = pd.DataFrame.from_dict(protein_counts, orient='index', columns=['count'])
        # protein_counts['count'] = scaler.fit_transform(protein_counts['count'].values.reshape(-1, 1))
        # protein_counts = protein_counts.to_dict()['count']

        # NOTE: We don't weight the counts by experimental evidence as this would magnify bias in studied proteins.
        # string_data_raw['experiments'] = scaler.fit_transform(string_data_raw['experiments'].values.reshape(-1, 1))
        # for protein in tqdm(protein_counts):
        #     protein_counts[protein] *= string_data_raw[string_data_raw['protein1'] == protein]['experiments'].mean()
        # print(protein_counts)

    def mouse_knockout_feature_extractor(self):
        keys = list(self.datasets.keys())
        df = self.datasets[keys[4]]
        target_counts = {}
        for target in df['targetInModel']:
            if target.upper() in target_counts:
                target_counts[target.upper()] += 1
            else:
                target_counts[target.upper()] = 1

        gene_names = list(target_counts.keys())
        mapped_names = map_gene_names(gene_names, 'symb', 'ensg')
        for gene_name in gene_names:
            if gene_name in mapped_names:
                target_counts[mapped_names[gene_name]] = target_counts.pop(gene_name)

        target_counts = {k: v for k, v in target_counts.items() if k != 'N/A'}

        self.mouse_ko_features = target_counts

        # NOTE: normalize AFTER train test split
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # target_freqs = pd.DataFrame.from_dict(target_counts, orient='index', columns=['count'])
        # target_freqs['count'] = scaler.fit_transform(target_freqs['count'].values.reshape(-1, 1))
        # target_freqs = target_freqs.to_dict()['count']

    def gene_essentiality_feature_extractor(self):
        raw_data = pd.read_csv(self.config['paths']['COMMON_ESSENTIALS_PATH'])
        raw_data = raw_data.rename(columns={raw_data.columns[0]: 'gene_name'})
        raw_data['gene_name'] = raw_data['gene_name'].str.split(' ').str[0]
        gene_names = list(raw_data['gene_name'])
        mapped_names = map_gene_names(gene_names, 'symb', 'ensg')
        gene_list = list(mapped_names.values())
        gene_essentiality = {gene: 1 for gene in gene_list}

        self.gene_essentiality_features = gene_essentiality

    ################################################ ARCHIVED FEATURES ################################################

    # Tractability features were used in the OpenTargets proof-of-concept model. They are not used in our model.
    def _tractability_feature_extractor(self):
        """
        DEPRECATED: This function is deprecated. Function was used to load OpenTargets tractability data.
        """
        keys = list(self.datasets.keys())
        tract_data_raw = self.datasets[keys[3]]
        sym_col = ['symbol']
        sm_cols = tract_data_raw.filter(regex='(SM_B)').columns.tolist()[3:]
        sm_cols = sym_col + sm_cols
        ab_cols = tract_data_raw.filter(regex='(AB_B)').columns.tolist()[3:]
        ab_cols = sym_col + ab_cols
        pr_cols = tract_data_raw.filter(regex='(PR_B)').columns.tolist()[3:]
        pr_cols = sym_col + pr_cols

        tract_data_sm = tract_data_raw.loc[:, sm_cols]
        tract_data_ab = tract_data_raw.loc[:, ab_cols]
        tract_data_pr = tract_data_raw.loc[:, pr_cols]

        return tract_data_sm, tract_data_ab, tract_data_pr

    def __tractability_feature_calculator(self):
        """
        DEPRECATED: This function is deprecated. Function was used to calculate tractability scores for the OpenTargets
        proof-of-concept model.
        """
        with open('data/Tractability/ab_shap_values.pkl', 'rb') as f:
            ab_shap_values = pkl.load(f)
        with open('data/Tractability/sm_shap_values.pkl', 'rb') as f:
            sm_shap_values = pkl.load(f)

        tract_sm = self.bin_tract_features[0]
        tract_ab = self.bin_tract_features[1]

        tract_sm_float = tract_sm.iloc[:, 1:].astype(float)
        tract_ab_float = tract_ab.iloc[:, 1:].astype(float)

        ab_mean_shap = np.abs(np.mean(ab_shap_values, axis=0))
        sm_mean_shap = np.abs(np.mean(sm_shap_values[:, 1:], axis=0))

        weighted_tract_sm = tract_sm_float * sm_mean_shap
        weighted_tract_ab = tract_ab_float * ab_mean_shap

        tract_score_sm = weighted_tract_sm.sum(axis=1)
        tract_score_ab = weighted_tract_ab.sum(axis=1)

        tract_sm['tractability_score'] = tract_score_sm
        tract_ab['tractability_score'] = tract_score_ab

        return tract_sm, tract_ab

    def __ground_truth_extractor(self):
        """
        DEPRECATED: This function is deprecated. Function was used to load OpenTargets tractability ground truth data.
        """
        keys = list(self.datasets.keys())
        tract_data_raw = self.datasets[keys[3]]
        sm_cols = tract_data_raw.filter(regex='(SM_B)').columns.tolist()[0]
        ab_cols = tract_data_raw.filter(regex='(AB_B)').columns.tolist()[0]
        pr_cols = tract_data_raw.filter(regex='(PR_B)').columns.tolist()[0]

        ground_truth_sm = tract_data_raw.loc[:, sm_cols]
        ground_truth_ab = tract_data_raw.loc[:, ab_cols]
        ground_truth_pr = tract_data_raw.loc[:, pr_cols]

        return ground_truth_sm, ground_truth_ab, ground_truth_pr

    def __query_alphafold_api(self):
        """
        DEPRECATED: This function is deprecated. Function was used to query the AlphaFold API to get the pLDDT scores
        for each protein in our dataset.
        """
        uniprot_data = self.pop_data[["SWISSPROT", "TREMBL", "varipred_id"]]
        uniprot_data["uniprot_id"] = uniprot_data["SWISSPROT"].fillna(uniprot_data["TREMBL"])
        uniprot_data = uniprot_data.drop(["SWISSPROT", "TREMBL"], axis=1).rename(columns={"uniprot_id": "UNIPROT"})
        uniprot_ids = uniprot_data["UNIPROT"].unique().tolist()
        extracted_values = {}
        for qualifier in tqdm(uniprot_ids):
            extracted_values[qualifier] = {}
            cif_file_path = f"{self.config['paths']['AF_PATH']}AF-{qualifier}-F1-model_v4.cif"
            target_format_mean = "_ma_qa_metric_global.metric_value"
            target_format_max = "_ma_qa_metric_local.ordinal_id"
            extract = False
            values_list = []
        base_url = "https://alphafold.ebi.ac.uk/api/uniprot"
        api_key = "AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"

        url = f"{base_url}/{qualifier}.json?key={api_key}"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            mean_value = float(data["structures"][0]["summary"]["confidence_avg_local_score"])
            extracted_values[qualifier]['mean'] = mean_value
        else:
            print(f"\nError: Unable to fetch data for {qualifier}. Status code: {response.status_code}. "
                  f"Inserting 0.0.")
            extracted_values[qualifier]['mean'] = 0.0
        return extracted_values

    def __gene_drug_evidence_feature_extractor(self):
        """
        DEPRECATED: We can not use the clinical annotations because they explictly encode FDA labels, which forms our
        positive class. This would lead to data leakage.
        """
        raw_data = pd.read_csv(self.config['paths']['GENE_DRUG_EVIDENCE_PATH'], sep='\t')
        efficacy_data = raw_data[raw_data['type'].str.contains('Efficacy')]
        toxicity_data = raw_data[raw_data['type'].str.contains('Toxicity')]

    def __binding_affinity_feature_extractor(self):
        """
        DEPRECATED: Data is too sparse. Either K_i or IC50 is present, not both and either is too sparse to use on its
        own.
        """
        binding_affinity_data = pd.read_csv(self.config['paths']['BINDING_AFFINITY_PATH'], on_bad_lines='warn',
                                            sep='\t')
        binding_affinity_human = binding_affinity_data[binding_affinity_data[('Target Source Organism According to '
                                                                              'Curator or DataSource')] == ('Homo '
                                                                                                            'sapiens')]


class GeneOntologyPreprocessor(GeneCharacterisationPreprocessor):
    """
    This class processes gene ontology data, specifically it extracts and processes data from the Human Protein Atlas:
    biological processes, molecular functions, subcellular locations and tissue specificity.
    """
    gene_ontology_features: pd.DataFrame
    data: pd.DataFrame

    def __init__(self, config, gcp=None):
        self.hpa_tissue_specificity_features = None
        self.gtex_tissue_specificity_features = None
        self.protein_atlas_feature_names = None
        if not gcp:
            super(GeneOntologyPreprocessor, self).__init__(config)
            self.gcp_data = self.data
            self.full_gcp_data = self.full_data
            self.gcp_pfam_pos = self.pfam_pos_data
            self.gcp_rcnt_pos = self.rcnt_pos_data
            self.gcp_pharos_pos = self.pharos_pos_data
            self.gcp_pfam_neg = self.pfam_neg_data
            self.gcp_rcnt_neg = self.rcnt_neg_data
            self.gcp_pharos_neg = self.pharos_neg_data
            self.gcp_population = self.population
            # self.gcp_acmg = gcp.acmg_data
        else:
            self.gcp = gcp
            self.pop_data = gcp.pop_data
            self.target = gcp.target
            self.gcp_data = gcp.data
            self.full_gcp_data = gcp.full_data
            self.population = gcp.population
            # self.gcp_pfam_pos = gcp.pfam_pos_data
            # self.gcp_rcnt_pos = gcp.rcnt_pos_data
            # self.gcp_pharos_pos = gcp.pharos_pos_data
            # self.gcp_pfam_neg = gcp.pfam_neg_data
            # self.gcp_rcnt_neg = gcp.rcnt_neg_data
            # self.gcp_pharos_neg = gcp.pharos_neg_data
            # self.pfam_ids = gcp.pfam_ids
            # self.rcnt_ids = gcp.rcnt_ids
            # self.pharos_ids = gcp.pharos_ids
            # self.pfam_neg_data = gcp.pfam_neg_data
            # self.rcnt_neg_data = gcp.rcnt_neg_data
            # self.pharos_neg_data = gcp.pharos_neg_data
            # self.pfam_ids_all = gcp.pfam_ids_all
            # self.rcnt_ids_all = gcp.rcnt_ids_all
            # self.pharos_ids_all = gcp.pharos_ids_all
            # self.drgbl_targets_pfam = gcp.drgbl_targets_pfam
            # self.gcp_acmg = gcp.acmg_data

        print("Gene Ontology Preprocessor booting up...")
        self.config = config

        # reset data variables
        self.pfam_data = None
        self.rcnt_data = None
        self.pharos_data = None

        print("Extracting protein atlas features...")
        self.protein_atlas_features = None
        self.protein_atlas_feature_extractor()

        self.tissue_specificity_features = None
        self.hpa_tissue_expression_feature_extractor()
        self.gtex_tissue_expression_feature_extractor()

        print("Combining gene ontology features...")
        self.gene_ontology_features = None
        features_dir = f"{self.config['paths']['FEATURES_DIR']}/{self.population}"
        if os.path.exists(f'{features_dir}/gene_ontology_features.pkl'):
            with open(f'{features_dir}/gene_ontology_features.pkl', 'rb') as f:
                self.gene_ontology_features = pkl.load(f)
        else:
            self.combine_go_features()

        self.data = self.gene_ontology_features
        self.data = self.data.set_index('ENSG')
        self.data.index.name = 'targetId'

        self.num_features = len(self.data.columns) - 1  # subtract 1 for the target column

    def protein_atlas_feature_extractor(self):
        features_dir = f"{self.config['paths']['FEATURES_DIR']}/{self.population}"
        if os.path.exists(f'{features_dir}/protein_atlas_features.pkl'):
            with open(f'{features_dir}/protein_atlas_features.pkl', 'rb') as f:
                self.protein_atlas_features = pkl.load(f)
            with open(f'{features_dir}/protein_atlas_feature_names.pkl', 'rb') as f:
                self.protein_atlas_feature_names = pkl.load(f)
        else:
            protein_atlas_features = pd.read_csv(self.config['paths']['PROTEIN_ATLAS_FEATURES'], sep='\t')
            protein_atlas_features = protein_atlas_features[['Ensembl', 'Biological process', 'Molecular function',
                                                             'Subcellular location']]

            all_features_bio_proc = set()
            all_features_mol_func = set()
            all_features_sub_loc = set()

            for feature_list in protein_atlas_features['Biological process'].values:
                if isinstance(feature_list, str):
                    all_features_bio_proc.update([f.strip() for f in feature_list.split(",")])

            for feature_list in protein_atlas_features['Molecular function'].values:
                if isinstance(feature_list, str):
                    all_features_mol_func.update([f.strip() for f in feature_list.split(",")])

            for feature_list in protein_atlas_features['Subcellular location'].values:
                if isinstance(feature_list, str):
                    all_features_sub_loc.update([f.strip() for f in feature_list.split(",")])

            all_features_bio_proc = sorted(list(all_features_bio_proc))
            all_features_mol_func = sorted(list(all_features_mol_func))
            all_features_sub_loc = sorted(list(all_features_sub_loc))

            feature_dict_bio_proc = {ensg: [0] * len(all_features_bio_proc) for ensg in
                                     protein_atlas_features['Ensembl']}
            feature_dict_mol_func = {ensg: [0] * len(all_features_mol_func) for ensg in
                                     protein_atlas_features['Ensembl']}
            feature_dict_sub_loc = {ensg: [0] * len(all_features_sub_loc) for ensg in protein_atlas_features['Ensembl']}

            bio_proc_feature_names = [f'biological_process__{feature}' for feature in all_features_bio_proc]
            mol_func_feature_names = [f'molecular_function__{feature}' for feature in all_features_mol_func]
            sub_loc_feature_names = [f'subcellular_location__{feature}' for feature in all_features_sub_loc]

            bio_proc_feature_names = [feature.replace(' ', '_') for feature in bio_proc_feature_names]
            mol_func_feature_names = [feature.replace(' ', '_') for feature in mol_func_feature_names]
            sub_loc_feature_names = [feature.replace(' ', '_') for feature in sub_loc_feature_names]

            self.protein_atlas_feature_names = {
                'biological_processes': bio_proc_feature_names,
                'molecular_functions': mol_func_feature_names,
                'subcellular_locations': sub_loc_feature_names
            }

            for index, row in protein_atlas_features.iterrows():
                ensg = row['Ensembl']
                bio_proc_features = row['Biological process'].split(",") if isinstance(row['Biological process'],
                                                                                       str) else []
                mol_func_features = row['Molecular function'].split(",") if isinstance(row['Molecular function'],
                                                                                       str) else []
                sub_loc_features = row['Subcellular location'].split(",") if isinstance(row['Subcellular location'],
                                                                                        str) else []

                for feature in bio_proc_features:
                    feature_index = all_features_bio_proc.index(feature.strip())
                    feature_dict_bio_proc[ensg][feature_index] = 1
                for feature in mol_func_features:
                    feature_index = all_features_mol_func.index(feature.strip())
                    feature_dict_mol_func[ensg][feature_index] = 1
                for feature in sub_loc_features:
                    feature_index = all_features_sub_loc.index(feature.strip())
                    feature_dict_sub_loc[ensg][feature_index] = 1

            self.protein_atlas_features = {
                'biological_processes': feature_dict_bio_proc,
                'molecular_processes': feature_dict_mol_func,
                'subcellular_locations': feature_dict_sub_loc
            }

            # save the protein atlas features and feature names
            with open(f'{features_dir}/protein_atlas_features.pkl', 'wb') as f:
                pkl.dump(self.protein_atlas_features, f)

            with open(f'{features_dir}/protein_atlas_feature_names.pkl', 'wb') as f:
                pkl.dump(self.protein_atlas_feature_names, f)

    def hpa_tissue_expression_feature_extractor(self):
        # check if the tissue expression data has already been processed
        features_dir = f"{self.config['paths']['FEATURES_DIR']}/{self.population}"
        if os.path.exists(f'{features_dir}/hpa_tissue_specificity_features.pkl'):
            with open(f'{features_dir}/hpa_tissue_specificity_features.pkl', 'rb') as f:
                self.hpa_tissue_specificity_features = pkl.load(f)
        else:
            tissue_expression = pd.read_csv(self.config['paths']['TISSUE_EXPRESSION_HPA'], sep='\t')  # 1,197,500
            tissue_expression = tissue_expression[tissue_expression['Reliability'] != 'Uncertain']  # 1,014,693
            tissue_expression = tissue_expression[tissue_expression['Level'] != 'Uncertain']  # 1,014,693

            gene_tissue_dict = {}
            for index, row in tissue_expression.iterrows():
                gene = row['Gene']
                tissue = row['Tissue']
                if gene in gene_tissue_dict:
                    gene_tissue_dict[gene].append(tissue)
                else:
                    gene_tissue_dict[gene] = [tissue]

            self.hpa_tissue_specificity_features = gene_tissue_dict
            with open(f'{features_dir}/hpa_tissue_specificity_features.pkl', 'wb') as f:
                pkl.dump(gene_tissue_dict, f)

    def gtex_tissue_expression_feature_extractor(self):
        features_dir = f"{self.config['paths']['FEATURES_DIR']}/{self.population}"
        if os.path.exists(f'{features_dir}/gtex_tissue_specificity_features.pkl'):
            with open(f'{features_dir}/gtex_tissue_specificity_features.pkl', 'rb') as f:
                self.gtex_tissue_specificity_features = pkl.load(f)
        else:
            with gzip.open(self.config['paths']['TISSUE_EXPRESSION_GTEX'], 'rt') as f:
                next(f)
                dim_line = next(f)
                rows, cols = map(int, dim_line.strip().split())
                header_line = next(f)
                column_names = header_line.strip().split('\t')
                data = []
                for line in f:
                    data.append(line.strip().split('\t'))

            gtex_data = pd.DataFrame(data)
            gtex_data.columns = column_names
            gtex_data = gtex_data.set_index(gtex_data.columns[0])

            for col in gtex_data.columns[1:]:
                gtex_data[col] = gtex_data[col].astype(float)

            if 'Description' in gtex_data.columns:
                gtex_data = gtex_data.drop(columns=['Description'])

            gtex_data = (gtex_data > 0).astype(float)

            gtex_data.index = gtex_data.index.str.split('.').str[0]

            if gtex_data.index.duplicated().any():
                gtex_data = gtex_data.groupby(gtex_data.index).max()

            # Create dictionary mapping gene IDs to list of expressed tissues
            gene_tissue_dict = {}
            for gene_id in gtex_data.index:
                expressed_tissues = gtex_data.columns[gtex_data.loc[gene_id] == 1.0].tolist()

                gene_tissue_dict[gene_id] = expressed_tissues

            # Save the tissue dictionary
            with open(f'{features_dir}/gtex_tissue_specificity_features.pkl', 'wb') as f:
                pkl.dump(gene_tissue_dict, f)

            self.gtex_tissue_specificity_features = gene_tissue_dict

    def combine_go_features(self):
        ensg_features = {
            "tissue_specificity_hpa": self.hpa_tissue_specificity_features,
            "tissue_specificity_gtex": self.gtex_tissue_specificity_features,
            "biological_processes": self.protein_atlas_features['biological_processes'],
            "molecular_functions": self.protein_atlas_features['molecular_processes'],
            "subcellular_locations": self.protein_atlas_features['subcellular_locations']
        }

        feature_matrix = pd.DataFrame({"ENSG": self.pop_data["Gene"]})
        feature_matrix = feature_matrix.drop_duplicates(subset=["ENSG"]).reset_index(drop=True)

        for feature, values in ensg_features.items():
            feature_matrix[feature] = feature_matrix["ENSG"].map(values)

        sub_df = feature_matrix.copy()
        sub_df = sub_df.iloc[:, -3:]

        bio_proc_list = []
        mol_func_list = []
        sub_loc_list = []

        for index, row in tqdm(sub_df.iterrows(), total=sub_df.shape[0]):
            bio_proc = row['biological_processes']
            mol_func = row['molecular_functions']
            sub_loc = row['subcellular_locations']

            if isinstance(bio_proc, float):
                bio_proc = [0] * len(self.protein_atlas_feature_names['biological_processes'])
            if isinstance(mol_func, float):
                mol_func = [0] * len(self.protein_atlas_feature_names['molecular_functions'])
            if isinstance(sub_loc, float):
                sub_loc = [0] * len(self.protein_atlas_feature_names['subcellular_locations'])

            bio_proc_array = np.array(bio_proc).reshape(-1, len(bio_proc))
            mol_func_array = np.array(mol_func).reshape(-1, len(mol_func))
            sub_loc_array = np.array(sub_loc).reshape(-1, len(sub_loc))

            bio_proc_list.append(bio_proc_array[0].tolist())
            mol_func_list.append(mol_func_array[0].tolist())
            sub_loc_list.append(sub_loc_array[0].tolist())

        bio_proc_feature_names = self.protein_atlas_feature_names['biological_processes']
        mol_func_feature_names = self.protein_atlas_feature_names['molecular_functions']
        sub_loc_feature_names = self.protein_atlas_feature_names['subcellular_locations']

        bio_proc_df = pd.DataFrame(np.array(bio_proc_list).reshape(-1, len(bio_proc_feature_names)),
                                   columns=bio_proc_feature_names)
        mol_func_df = pd.DataFrame(np.array(mol_func_list).reshape(-1, len(mol_func_feature_names)),
                                   columns=mol_func_feature_names)
        sub_loc_df = pd.DataFrame(np.array(sub_loc_list).reshape(-1, len(sub_loc_feature_names)),
                                  columns=sub_loc_feature_names)

        feature_matrix = feature_matrix.drop(['biological_processes', 'molecular_functions', 'subcellular_locations'],
                                             axis=1)

        feature_matrix.reset_index(drop=True, inplace=True)
        bio_proc_df.reset_index(drop=True, inplace=True)
        mol_func_df.reset_index(drop=True, inplace=True)
        sub_loc_df.reset_index(drop=True, inplace=True)

        feature_matrix = pd.concat([feature_matrix, bio_proc_df, mol_func_df, sub_loc_df], axis=1)

        feature_matrix = feature_matrix.fillna(0)

        for col in ['tissue_specificity_hpa', 'tissue_specificity_gtex']:
            feature_matrix[col] = feature_matrix[col].apply(lambda x: x if isinstance(x, list) else [])

        # Get unique tissues from both data sources
        unique_hpa_tissues = set(tissue for tissues_list in feature_matrix['tissue_specificity_hpa']
                                 for tissue in tissues_list)
        unique_gtex_tissues = set(tissue for tissues_list in feature_matrix['tissue_specificity_gtex']
                                  for tissue in tissues_list)

        # Create new columns for each unique HPA tissue
        for tissue in unique_hpa_tissues:
            tissue_col = f'tissue_hpa_{tissue.replace(" ", "_")}'
            feature_matrix[tissue_col] = feature_matrix['tissue_specificity_hpa'].apply(
                lambda x: 1 if tissue in x else 0)

        # Create new columns for each unique GTEx tissue
        for tissue in unique_gtex_tissues:
            tissue_col = f'tissue_gtex_{tissue.replace(" ", "_")}'
            feature_matrix[tissue_col] = feature_matrix['tissue_specificity_gtex'].apply(
                lambda x: 1 if tissue in x else 0)
            feature_matrix = feature_matrix.copy()

        feature_matrix = feature_matrix.drop(['tissue_specificity_hpa', 'tissue_specificity_gtex'], axis=1)
        feature_matrix = feature_matrix.loc[:, (feature_matrix != 0).any(axis=0)]

        # Fill any remaining NaN values with 0
        self.gene_ontology_features = feature_matrix.fillna(0)

        features_dir = self.config['paths']['FEATURES_DIR']
        with open(f'{features_dir}/{self.population}/gene_ontology_features.pkl', 'wb') as f:
            pkl.dump(self.gene_ontology_features, f)


class PopulationVariantPreprocessor(GeneCharacterisationPreprocessor):
    """
    This class processes protein variant information, specifically it obtains and processes amino acid sequence embeddings
    and missense variant pathogenicity embeddings, and it processes protein structure confidence scores, in particular
    it generates and processes embeddings of AlphaFold's residue-wise pLDDT score.
    """

    def __init__(self, config, gcp=None):
        if not gcp:
            super().__init__(config)
            self.gcp_data = self.data
            self.full_gcp_data = self.full_data
            self.gcp_pfam_pos = self.pfam_pos_data
            self.gcp_rcnt_pos = self.rcnt_pos_data
            self.gcp_pharos_pos = self.pharos_pos_data
            self.gcp_pfam_neg = self.pfam_neg_data
            self.gcp_rcnt_neg = self.rcnt_neg_data
            self.gcp_pharos_neg = self.pharos_neg_data
            self.gcp_ce_data = self.ce_data
            self.gcp_population = self.population
            # self.gcp_acmg = gcp.acmg_data
        else:
            self.gcp = gcp
            self.pop_data = gcp.pop_data
            self.target = gcp.target
            self.gcp_data = gcp.data
            self.full_gcp_data = gcp.full_data
            self.population = gcp.population
            # self.gcp_pfam_pos = gcp.pfam_pos_data
            # self.gcp_rcnt_pos = gcp.rcnt_pos_data
            # self.gcp_pharos_pos = gcp.pharos_pos_data
            # self.gcp_pfam_neg = gcp.pfam_neg_data
            # self.gcp_rcnt_neg = gcp.rcnt_neg_data
            # self.gcp_pharos_neg = gcp.pharos_neg_data
            # self.drgbl_targets_pfam = gcp.drgbl_targets_pfam
            # self.rcnt_targets_fda = gcp.rcnt_targets_fda
            # self.chem_targets_pharos = gcp.chem_targets_pharos
            self.ensg_ids = gcp.ensg_ids
            self.gcp_ce_data = gcp.ce_data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_features = config['hyperparameters']['max_seq_len']

        features_dir = f"{config['paths']['FEATURES_DIR']}/{self.population}"
        print("Obtaining AlphaMissense pathogenicity embeddings...")
        if not os.path.exists(f'{features_dir}/var_pat_features.pkl'):
            print("Preparing variant features...")
            self.variant_gh_data(config)
            self.var_pat_features, self.var_gene_map = self.varformer_pathogenicity_input()
            with open(f'{features_dir}/var_pat_features.pkl', 'wb') as file:
                pkl.dump(self.var_pat_features, file)
        else:
            with open(f'{features_dir}/var_pat_features.pkl', 'rb') as f:
                self.var_pat_features = pkl.load(f)
            with open(self.gcp.config['paths']['GENE_VAR_MAP'], 'rb') as f:
                self.var_gene_map = pkl.load(f)

        self.norm = False

        # Ground truth
        self.target = self.full_gcp_data['target']
        self.labels = {key: 1 if key in self.target.tolist() else 0 for key in self.var_pat_features.keys()}

        self.data = {key: value for key, value in self.var_pat_features.items()}
        self.data = {key: value for key, value in self.var_pat_features.items()}
        self.data['labels'] = self.labels

    def variant_gh_data(self, config):
        print("Preparing GH data for variant-level embeddings...")
        data_dir = config['paths']['FEATURES_DIR']
        if not os.path.exists(f'{data_dir}/{self.population}/am_pop_merge.pkl'):
            am = merge_am_data(self.pop_data, self.population)
            if hasattr(am, 'to_pandas') and not isinstance(am, pd.DataFrame):
                self.pop_data = am.to_pandas()
            elif isinstance(am, pd.DataFrame):
                self.pop_data = am
            else:
                # Handle case where am doesn't have to_pandas() method
                raise ValueError(f"Cannot convert {type(am)} to pandas DataFrame")
            # if 'AC' not in self.gh_data_am.columns:
            #     self.pop_am_merge = self.pop_am_merge[
            #         self.pop_am_merge['variant_id'].isin(self.pop_data['variant_id'].unique())]
            #     self.pop_am_merge['AC'] = self.pop_am_merge['variant_id'].map(self.pop_data.set_index('variant_id')['AC'])
            # elif 'AN' not in self.gh_data_am.columns:
            #     self.pop_am_merge['AN'] = self.pop_am_merge['variant_id'].map(self.pop_data.set_index('variant_id')['AN'])
            # self.pop_data = self.pop_am_merge
            self.variant_sharding(config)
        else:
            self.pop_data = pd.read_pickle(f'{data_dir}/{self.population}/am_pop_merge.pkl')
            max_pos = self.pop_data['Protein_pos_shard'].max()
            if max_pos + 1 != config['hyperparameters']['max_seq_len']:
                print("Max sequence len dimension has been changed, reprocessing GH data...")
                self.variant_sharding(config)
                if os.path.exists(f'{data_dir}/{self.population}/var_pat_features.pkl'):
                    os.remove(f'{data_dir}/{self.population}/var_pat_features.pkl')

    def variant_sharding(self, config):
        self.pop_data['ALT'] = self.pop_data['ALT'].str.split(',')
        self.pop_data = self.pop_data.explode('ALT')
        self.pop_data = self.pop_data[(self.pop_data['ALT'].str.len() == 1) & (self.pop_data['REF'].str.len() == 1)]
        self.pop_data = self.pop_data[self.pop_data['Consequence'] == 'missense_variant']

        self.pop_data['Protein_position'] = self.pop_data['Protein_position'].astype(int)
        max_seq_len = config['hyperparameters']['max_seq_len']
        self.pop_data.loc[:, 'Protein_pos_shard'] = self.pop_data['Protein_position'].apply(
            lambda x: x % max_seq_len)
        cols = self.pop_data.columns.tolist()
        pp_idx = cols.index('Protein_position')
        cols = cols[:pp_idx] + [cols[-1]] + cols[pp_idx:-1]
        self.pop_data = self.pop_data[cols]
        data_dir = config['paths']['FEATURES_DIR']
        self.pop_data.to_pickle(f'{data_dir}/{self.population}/am_pop_merge.pkl')

    def missense_mutation_map(self):
        mutation_map = {'UNK': 0}
        mut_id = 1

        all_aas = list(set(self.pop_data['AA_ref'].unique().tolist() + self.pop_data['AA_alt'].unique().tolist()))

        for ref in all_aas:
            for alt in all_aas:
                if ref != alt:
                    mutation = f"{ref}>{alt}"
                    mutation_map[mutation] = mut_id
                    mut_id += 1
        return mutation_map

    def varformer_pathogenicity_input(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and prepare for input into the Varformer model. We need
        to prepare data for 1) pathogenicity embedding, 2) positional embedding and 3) missense mutation identity
        embedding.
        """
        other_columns = [col for col in self.pop_data.columns if col not in ['variant_id', 'am_pathogenicity']]

        agg_dict = {col: 'first' for col in other_columns}
        agg_dict['am_pathogenicity'] = 'mean'

        self.pop_data = self.pop_data.groupby('variant_id', as_index=False).agg(agg_dict)

        self.config = self.gcp.config

        sym_list = self.pop_data['SYMBOL'].tolist()
        prot_pos = self.pop_data['Protein_position'].tolist()
        aas = self.pop_data['Amino_acids'].tolist()
        hgvsp = self.pop_data['HGVSp'].tolist()
        prot_pos = [str(pos) for pos in prot_pos]
        hgvsp = [str(hgvsp.split('.')[1].split(':')[0]) for hgvsp in hgvsp]
        isoform_id_list = [f"{sym}_{prot_pos}_{aas.split('/')[0]}_{aas.split('/')[1]}_{hgvsp}" for
                           sym, prot_pos, aas, hgvsp in zip(sym_list, prot_pos, aas, hgvsp)]
        self.pop_data['isoform'] = isoform_id_list

        components = self.pop_data['isoform'].str.extract(
            r'(?P<gene_symbol>\w+)_(?P<prot_position>\d+)_(?P<ref_aa>\w+)_(?P<alt_aa>\w+)_(?P<isoform>\d+)')
        # combined = components['gene_symbol'] + '_' + components['prot_position'].astype(str) + '_' + components[
        #     'ref_aa'] + '_' + components['alt_aa']
        # gh['combined'] = combined
        self.pop_data['AA_ref'] = components['ref_aa']
        self.pop_data['AA_alt'] = components['alt_aa']
        # gh = gh.loc[components['isoform'] == '1'].drop_duplicates(subset=['combined'])
        # gh = gh.drop(columns=['combined'])

        mutation_map = self.missense_mutation_map()
        data_dir = self.config['paths']['DATA_DIR']

        # save the mutation_map to a .pkl file
        # with open(f'{data_dir}/elgh/missense_mutation_map.pkl', 'wb') as f:
        #     pkl.dump(mutation_map, f)

        gene_map = {gene: i for i, gene in enumerate(self.pop_data['Gene'].unique())}
        with open(self.config['paths']['VAR_MAP'], 'rb') as file:
            variant_map = pkl.load(file)
        var_pat_features = {}
        gene_var_map = {}
        for index, row in tqdm(self.pop_data.iterrows(), total=self.pop_data.shape[0]):
            gene = row['Gene']
            variant_id = row['variant_id']
            # rs_id = variant_map.get(variant_id, None)
            # if rs_id is not None:
            #     variant_id = rs_id
            ref_aa = row['AA_ref']
            alt_aa = row['AA_alt']
            mutation = f"{ref_aa}>{alt_aa}"
            mut = mutation_map.get(mutation, mutation_map['UNK'])
            pos = row['Protein_pos_shard']  # 0-indexed
            pat_value = row['am_pathogenicity']
            if np.isnan(pat_value):
                continue
            if gene not in var_pat_features.keys():
                gene_var_map[gene] = [variant_id]
                var_pat_features[gene] = [[pat_value, pos, mut, gene_map[gene]]]
            else:
                if variant_id in gene_var_map[gene]:
                    continue
                else:
                    var_pat_features[gene].append([pat_value, pos, mut, gene_map[gene]])
                    gene_var_map[gene].append(variant_id)

        var_features = {}
        max_seq_len = self.config['hyperparameters']['max_seq_len']

        for gene, features in var_pat_features.items():
            zipped = list(zip(features, gene_var_map[gene]))  # [(feature_list, variant_id), ...]

            zipped.sort(key=lambda x: x[0][0], reverse=True)
            top_zipped = zipped[:max_seq_len]
            top_features, top_variant_ids = zip(*top_zipped) if top_zipped else ([], [])
            var_features[gene] = torch.tensor(top_features)
            gene_var_map[gene] = list(top_variant_ids)

            if len(gene_var_map[gene]) > max_seq_len:
                print(f"ERROR: Gene {gene} has {len(gene_var_map[gene])} variants, max_seq_len is {max_seq_len}")

        with open(f'{data_dir}/features/{self.population}/gene_loc_var_map.pkl', 'wb') as f:
            pkl.dump(gene_var_map, f)
        return var_features, gene_var_map

    def variant_structure_input(self):
        af_data = self.alphafold_extractor()

        if not os.path.exists('../data/features/var_stc_features.pkl.gz'):
            var_stc_features = {}

            num_genes = len(af_data)

            for gene, gene_data in tqdm(af_data.items(), total=num_genes):
                coords = gene_data['coordinates']
                plddt = gene_data['res_plddt']
                sequence = gene_data['sequence']
                seq_length = gene_data['protein_len']
                matrix = sparse.lil_matrix((4, 2700 * 20), dtype=np.float32)

                for i, (coords, c, amino_acid) in enumerate(zip(coords, plddt, sequence)):
                    amino_acid_idx = three_letter_aa_to_idx(amino_acid)
                    x, y, z = coords
                    values = [x, y, z, c]

                    for row_num in range(4):
                        matrix_index = i * 20 + amino_acid_idx
                        matrix[row_num, matrix_index] = values[row_num]

                    matrix[:, (matrix_index + 1):] = 0.0
                var_stc_features[gene] = matrix.tocsr()

            with gzip.open('../data/features/var_stc_features.pkl.gz', 'wb') as f:
                pkl.dump(var_stc_features, f)
            return var_stc_features

        else:
            with gzip.open('../data/features/var_stc_features.pkl.gz', 'rb') as f:
                return pkl.load(f)

    def variant_sequence_input(self):
        if not os.path.exists('../data/features/var_seq_features.pkl.gz'):
            # Load the latest checkpoint if it exists
            checkpoint_path = '../data/features/var_seq_features_ckpt.pkl.gz'
            if os.path.exists(checkpoint_path):
                with gzip.open(checkpoint_path, 'rb') as f:
                    var_seq_features_ckpt = pkl.load(f)
                var_seq_features = var_seq_features_ckpt
            else:
                var_seq_features = {}
                var_seq_features_ckpt = {}

            var_seq_data = self.pop_data.drop_duplicates(subset=['Feature'])

            # var_seq_data_split = np.array_split(var_seq_data, 8)
            # var_seq_data = var_seq_data_split[0]

            transcript_ids = var_seq_data['Feature'].tolist()
            ensg_ids = var_seq_data['Gene'].tolist()
            prot_pos = var_seq_data['Protein_position'].tolist()
            # prot_pos_shard = var_seq_data['Protein_pos_shard'].tolist()
            amino_acids = var_seq_data['Amino_acids'].tolist()
            ref_aas = [aa.split('/')[0] for aa in amino_acids]
            mt_aas = [aa.split('/')[1] for aa in amino_acids]
            wildtype_sequences = {}
            gene_counters = {}

            num_genes = len(transcript_ids)
            ref_aa_count = 0
            iso_mismatch_count = 0
            no_enst_seq_count = 0
            counter = 0
            for i, (transcript_id, ensg_id) in tqdm(enumerate(zip(transcript_ids, ensg_ids)), total=num_genes):
                if ensg_id in var_seq_features_ckpt.keys():
                    continue

                var_seq, wt_seq, wildtype_sequences = self.get_protein_sequence(transcript_id, ensg_id,
                                                                                wildtype_sequences)
                if var_seq is None or wt_seq is None:
                    no_enst_seq_count += 1
                    continue
                if prot_pos[i] - 1 >= len(var_seq):
                    print("Skipping because of isoform mismatch. . .")
                    iso_mismatch_count += 1
                    continue
                if ref_aas[i] != var_seq[prot_pos[i] - 1]:
                    print("Skipping because of reference amino acid mismatch. . .")
                    ref_aa_count += 1
                    continue

                var_seq = list(var_seq)
                var_seq[prot_pos[i] - 1] = mt_aas[i]
                var_seq = ''.join(var_seq)

                if len(wt_seq) > 1022:  # split the sequence into chunks of 1022 (ESM-2 limitation)
                    wt_seqs = [wt_seq[i:i + 1022] for i in range(0, len(wt_seq), 1022)]
                    var_seqs = [var_seq[i:i + 1022] for i in range(0, len(var_seq), 1022)]
                    variant_chunk_index = (prot_pos[i] - 1) // 1022
                    for j, (wt_seq, var_seq) in enumerate(zip(wt_seqs, var_seqs)):
                        if j == variant_chunk_index:
                            seqs = [
                                ("wt_seq", wt_seq),
                                ("mt_seq", var_seq)
                            ]
                            wt_emb, mt_emb = self.get_esm_embeddings(seqs)
                else:
                    seqs = [
                        ("wt_seq", wt_seq),
                        ("mt_seq", var_seq)
                    ]
                    wt_emb, mt_emb = self.get_esm_embeddings(seqs)

                var_emb = mt_emb - wt_emb

                row_idx = 4096 // 2
                if ensg_id not in var_seq_features.keys():
                    matrix_shape = (4096, 1280)
                    var_seq_matrix = sparse.lil_matrix(matrix_shape, dtype=np.float32)
                    var_seq_matrix[row_idx, :] = var_emb
                    var_seq_features[ensg_id] = var_seq_matrix.tocsr()
                    gene_counters[ensg_id] = 1  # Initialize counter for this gene
                else:
                    var_seq_matrix = var_seq_features[ensg_id].tolil()
                    gene_counter = gene_counters[ensg_id]
                    offset = (gene_counter + 1) // 2  # Calculate offset from the middle row
                    if gene_counter % 2 == 0:
                        row_idx -= offset
                    else:
                        row_idx += offset

                    var_seq_matrix[row_idx, :] = var_emb
                    var_seq_features[ensg_id] = var_seq_matrix.tocsr()
                    gene_counters[ensg_id] += 1  # Increment counter for this gene

                counter += 1
                if counter >= 10:
                    # Save a checkpoint
                    with gzip.open(checkpoint_path, 'wb') as f:
                        pkl.dump(var_seq_features, f)
                    counter = 0

            print(f"Finished! Following are the statistics for the missing embeddings:'n")
            print(f"Number of reference amino acid mismatches: {ref_aa_count}\n")
            print(f"Number of isoform mismatches: {iso_mismatch_count}\n")
            print(f"Number of missing ENST sequences: {no_enst_seq_count}\n")

            var_seq_flattened = {gene: matrix.toarray().ravel() for gene, matrix in var_seq_features.items()}

            with gzip.open('../data/features/var_seq_features_1.pkl.gz', 'wb') as f:
                pkl.dump(var_seq_flattened, f)
            return var_seq_flattened
        else:
            with gzip.open('../data/features/var_seq_features.pkl.gz', 'rb') as f:
                var_seq = pkl.load(f)
                first_val = list(var_seq.values())[0]
                if len(first_val.shape) == 1:
                    return var_seq
                else:
                    var_seq_flattened = {gene: matrix.toarray().ravel() for gene, matrix in var_seq.items()}
                    return var_seq_flattened

    @staticmethod
    def get_protein_sequence(transcript_id, ensg_id, wildtype_sequences, max_retries=10, retry_delay=5):
        server = "https://rest.ensembl.org"
        ext_variant = f"/sequence/id/{transcript_id}?type=protein"
        ext_wildtype = f"/lookup/id/{ensg_id}?expand=1"

        retry_count = 0

        while retry_count < max_retries:
            try:
                # Check if wildtype sequence is already in the dictionary
                canonical_transcript_id = None
                if ensg_id in list(wildtype_sequences.keys()):
                    wildtype_sequence = wildtype_sequences[ensg_id]
                else:
                    # Fetch gene information from Ensembl to get the canonical transcript
                    response_wildtype = requests.get(server + ext_wildtype,
                                                     headers={"Content-Type": "application/json"})
                    # catch 400 error
                    if response_wildtype.status_code == 400:
                        print(f"Bad request for {ensg_id}")
                        return None, None, wildtype_sequences

                    response_wildtype.raise_for_status()  # Raise an exception for non-2xx status codes

                    gene_data = response_wildtype.json()

                    # Find the canonical transcript
                    canonical_transcript = None
                    for transcript in gene_data["Transcript"]:
                        if transcript["is_canonical"]:
                            canonical_transcript = transcript
                            break

                    if canonical_transcript:
                        canonical_transcript_id = canonical_transcript["id"]

                        # Fetch canonical protein sequence from Ensembl
                        ext_canonical_protein = f"/sequence/id/{canonical_transcript_id}?type=protein"
                        response_canonical_protein = requests.get(server + ext_canonical_protein,
                                                                  headers={"Content-Type": "text/plain"})

                        if response_wildtype.status_code == 400:
                            print(f"Bad request for {transcript_id}")
                            return None, None, wildtype_sequences

                        response_canonical_protein.raise_for_status()  # Raise an exception for non-2xx status codes

                        wildtype_sequence = response_canonical_protein.text
                        wildtype_sequence = wildtype_sequence.replace("*", "")
                        wildtype_sequences[ensg_id] = wildtype_sequence
                    else:
                        print(f"No canonical transcript found for {ensg_id}")
                        wildtype_sequence = None

                if canonical_transcript_id != transcript_id:
                    response_variant = requests.get(server + ext_variant, headers={"Content-Type": "text/plain"})
                    if response_variant.status_code == 400:
                        print(f"Bad request for {transcript_id}")
                        return None, None, wildtype_sequences
                    response_variant.raise_for_status()  # Raise an exception for non-2xx status codes
                    var_sequence = response_variant.text
                    var_sequence = var_sequence.replace("*", "")
                    return var_sequence, wildtype_sequence, wildtype_sequences
                else:
                    variant_sequence = wildtype_sequence
                    return variant_sequence, wildtype_sequence, wildtype_sequences

            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < max_retries:
                    print(
                        f"Error occurred: {e}. Retrying in {retry_delay} seconds... (Attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    raise TimeoutError(f"Maximum number of retries ({max_retries}) reached. Giving up.")

    def get_esm_embeddings(self, seqs):
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(seqs)
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)

        # Generate the embeddings on the CUDA device
        with torch.no_grad():
            results = self.esm_model(batch_tokens.to(self.device), repr_layers=[33])

        # Get the embeddings from the last layer
        token_embeddings = results["representations"][33].cpu()

        # Generate per-sequence representations via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_embeddings[i, 1: tokens_len - 1].mean(0))
        return sequence_representations

    def fetch_pathogenicity_embeddings(self, variant_am_features):
        hparams = self.config['hyperparameters']['pathogenicity_autoencoder']
        if hparams['ae_type'] == "ae":
            model_dir = 'autoencoders/checkpoints/variant_pathogenicity_encoder'
            model_files = os.listdir(model_dir)
            assert len(model_files) == 1
            model_file = model_files[0]
            model = AutoencoderTrainer.load_from_checkpoint(
                model_dir + '/' + model_file, input_dim=hparams['max_seq_len'],
                encoding_dim=hparams['latent_dim'],
                num_layers=hparams['num_layers'], nhead=hparams['nhead'],
                reduction_type=hparams['reduction']
            )
        else:
            model_dir = 'autoencoders/checkpoints/variant_pathogenicity_vae'
            model_files = os.listdir(model_dir)
            assert len(model_files) == 1
            model_file = model_files[0]
            model = VAETrainer.load_from_checkpoint(
                model_dir + '/' + model_file, input_dim=hparams['max_seq_len'],
                latent_dim=hparams['latent_dim']
            )

        model.eval()

        with torch.no_grad():
            variant_pathogenicity = DataLoader(
                dl.VariantPathogenicityData(data_dict=variant_am_features, reduct_dim=hparams['max_seq_len'],
                                            reduction_type=hparams['reduction']),
                collate_fn=ae_training.padding,
                shuffle=False
            )
            gene_names = variant_pathogenicity.dataset.gene_names

            print("Generating embeddings...")

            trainer = pl.Trainer()
            embeddings = trainer.predict(model, dataloaders=variant_pathogenicity)
            embeddings = {gene: embedding.tolist()[0] for gene, embedding in zip(gene_names, embeddings)}

        return embeddings

    def variant_emb_data(self):
        path_embs = pd.DataFrame.from_dict(self.var_pat_embs, orient='index')
        stc_embs = pd.DataFrame.from_dict(self.var_stc_embs, orient='index')
        seq_embs = pd.DataFrame.from_dict(self.var_seq_embs, orient='index')

        combined_data = combined_data.reset_index().rename(columns={'index': 'ENSG'})

        path_genes = list(combined_data['ENSG'])
        gh_genes = list(self.ensg_ids)
        missing_genes = list(set(gh_genes) - set(path_genes))

        emb_vals = combined_data.iloc[:, 1:].values
        mean_embs = emb_vals.mean(axis=0)
        mean_embs_dict = dict(zip(combined_data.columns[1:], mean_embs))
        missing_genes_df = pd.DataFrame(index=missing_genes, columns=combined_data.columns[1:])
        missing_genes_df = missing_genes_df.fillna(mean_embs_dict)

        missing_genes_df['ENSG'] = missing_genes_df.index
        missing_genes_df = missing_genes_df.reset_index(drop=True)
        missing_genes_df = missing_genes_df[['ENSG'] + [col for col in missing_genes_df.columns if col != 'ENSG']]

        combined_data = pd.concat([combined_data, missing_genes_df], axis=0)

        target = self.target
        target['target'] = 1
        target = target.drop(['Gene'], axis=1)
        target = target.rename(columns={'Ensembl': 'ENSG'})
        target_genes = list(target['ENSG'])
        target_genes = list(set(target_genes) & set(gh_genes))
        target_genes = pd.DataFrame(target_genes, columns=['ENSG'])
        target_genes['target'] = 1

        combined_data = combined_data.merge(target_genes, on='ENSG', how='left')
        combined_data = combined_data.fillna(0)

        return combined_data

    def plddt_embedder(self):
        self.alphafold_feature_extractor()
        alphafold_raw = self.alphafold_features

        plddt_raw = {}
        for uniprot_id, data_types in alphafold_raw.items():
            res_pldtt_values = data_types['res_plddt']
            plddt_raw[uniprot_id] = res_pldtt_values

        return 0

    def alphafold_extractor(self):
        """
        Extract AlphaFold features from the AlphaFold API. Specifically, we get the average pLDDT score for the proteins
        in our dataset
        """
        uniprot_ids = self.pop_data["UNIPROT"].unique().tolist()
        uniprot_ids = [uni for uni in uniprot_ids if str(uni) != 'nan']
        uniprot_ids = [uni.split('.')[0] for uni in uniprot_ids]
        if not os.path.exists('data/cache/uniprot_ids.pkl'):
            uni_to_gene = {}
            for index, row in self.pop_data.iterrows():
                gene = row['Gene']
                uni = row['UNIPROT']
                if type(uni) == float:
                    continue
                uni = uni.split('.')[0]
                uni_to_gene[uni] = gene

            # write the uniprot ids to a pkl file
            with open('data/cache/uniprot_ids.pkl', 'wb') as fp:
                pkl.dump(uni_to_gene, fp)
        else:
            with open('data/cache/uniprot_ids.pkl', 'rb') as fp:
                uni_to_gene = pkl.load(fp)

        if not os.path.exists('data/alphafold/alphafold_cifs'):
            self.download_af_cifs()
        extracted_values = {}
        if os.path.exists('data/features/alphafold_features.pkl'):
            with open('data/features/alphafold_features.pkl', 'rb') as fp:
                features = pkl.load(fp)
                return features
        else:
            if os.path.exists('data/alphafold/alphafold_features_temp.pkl'):
                with open('data/alphafold/alphafold_features_temp.pkl', 'rb') as fp:
                    extracted_values = pkl.load(fp)
            else:
                for qualifier in tqdm(uniprot_ids):
                    extracted_values[qualifier] = {}
                    target_format_mean = "_ma_qa_metric_global.metric_value"
                    target_format_max = "_ma_qa_metric_local.ordinal_id"
                    extract = True
                    values_list = []
                    aas = []
                    cif_file_path = f"{self.config['paths']['AF_PATH']}AF-{qualifier}-F1-model_v4.cif"
                    try:
                        with open(cif_file_path, "r") as cif_file:
                            for line in cif_file:
                                if line.startswith(target_format_mean):
                                    mean_value = line[len(target_format_mean):].strip()
                                    extracted_values[qualifier]['mean'] = float(mean_value)

                                if line.startswith(target_format_max):
                                    extract = True
                                    continue

                                if line == '#\n':
                                    extract = False
                                    continue

                                if extract:
                                    parts = line.split()
                                    line_str = ' '.join(parts)
                                    if line_str.startswith('<?xml') and '<Error>' in line_str:
                                        continue
                                    elif len(parts) >= 5:
                                        plddt = float(parts[4])
                                        values_list.append(plddt)
                                        aas.append(parts[1])

                            if len(values_list) != 0:
                                protein_len = len(values_list)
                                max_value = max(values_list)
                                extracted_values[qualifier]['max'] = max_value
                                extracted_values[qualifier]['sequence'] = aas
                                extracted_values[qualifier]['res_plddt'] = values_list
                                # we extract the len for later experiments
                                extracted_values[qualifier]['protein_len'] = protein_len
                            else:
                                print(f"\nError: Unable to fetch data for {qualifier}. Inserting 0.0.")
                                extracted_values[qualifier]['mean'] = 0.0
                                extracted_values[qualifier]['max'] = 0.0
                                extracted_values[qualifier]['protein_len'] = np.nan
                                extracted_values[qualifier]['sequence'] = []
                                extracted_values[qualifier]['res_plddt'] = []
                                extracted_values[qualifier]['coordinates'] = []
                        cif_file.close()
                    except FileNotFoundError as e:
                        print(f"\nError: Unable to fetch data for {qualifier}. Inserting 0.0.")
                        extracted_values[qualifier]['mean'] = 0.0
                        extracted_values[qualifier]['max'] = 0.0
                        extracted_values[qualifier]['protein_len'] = np.nan
                        extracted_values[qualifier]['sequence'] = []
                        extracted_values[qualifier]['res_plddt'] = []
                        extracted_values[qualifier]['coordinates'] = []

                # write the extracted values to a pkl file
                with open('data/alphafold/temp_extracted_values.pkl', 'wb') as fp:
                    pkl.dump(extracted_values, fp)

                for uni, info in tqdm(extracted_values.items(), total=len(extracted_values)):
                    seq = info['sequence']
                    if not seq:
                        continue
                    cif_file_path = f"data/alphafold/alphafold_cifs/AF-{uni}-F1-model_v4.cif"
                    coords = []
                    curr_aa_id = 1
                    with open(cif_file_path, "r") as cif_file:
                        for line in cif_file:
                            if line.startswith('ATOM'):
                                aa_id = int(line.split()[23])
                                atom_type = line.split()[3]
                                if aa_id == curr_aa_id and atom_type == 'CA':
                                    x = float(line.split()[10])
                                    y = float(line.split()[11])
                                    z = float(line.split()[12])
                                    coords.append((x, y, z))
                                    curr_aa_id += 1
                    extracted_values[uni]['coordinates'] = coords

            extracted_values = {uni_to_gene[uniprot]: values for uniprot, values in extracted_values.items()}
            with open('data/features/alphafold_features.pkl', 'wb') as fp:
                pkl.dump(extracted_values, fp)
            return extracted_values

    def download_af_cifs(self):
        uniprot_ids = self.pop_data["UNIPROT"].unique().tolist()
        uniprot_ids = [uni for uni in uniprot_ids if str(uni) != 'nan']
        uniprot_ids = [uni.split('.')[0] for uni in uniprot_ids]

        url = "https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.cif"

        folder = "data/alphafold/alphafold_cifs"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for uni_id in tqdm(uniprot_ids):
            # check if uni_id occurs in swissprot_cif_v4 folder, if so copy it to alphafold_cifs
            if os.path.exists(f"data/alphafold/alphafold_cifs/AF-{uni_id}-F1-model_v4.cif"):
                continue
            file_url = url.format(id=uni_id)
            file_name = os.path.basename(file_url)

            response = requests.get(file_url)
            with open(os.path.join(folder, file_name), "wb") as f:
                f.write(response.content)

    # DEPRECATED

    def __variant_pathogenicity_input(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and prepare the input for embedding.
        Deprecated because we switched to VarFormer for pathogenicity embedding.
        """
        # check if gh_am_data.pkl exists yet
        print("Combining AM and GH data...")
        if not os.path.exists("data/features/var_pat_features.pkl.gz"):
            am = pd.read_csv("data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')
            am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']

            am = am[['am_pathogenicity', 'variant_id']]
            print("Combining AlphaMissense data with GH data...")
            self.pop_data = self.pop_data.merge(am, on='variant_id', how='left')
            # save gh_data
            with open("data/alphamissense/gh_am_data_full.pkl", 'wb') as f:
                pkl.dump(self.pop_data, f)

            sym_list = self.pop_data['SYMBOL'].tolist()
            prot_pos = self.pop_data['Protein_pos_shard'].tolist()
            aas = self.pop_data['Amino_acids'].tolist()
            hgvsp = self.pop_data['HGVSp'].tolist()
            prot_pos = [str(pos) for pos in prot_pos]
            hgvsp = [str(hgvsp.split('.')[1].split(':')[0]) for hgvsp in hgvsp]
            isoform_id_list = [f"{sym}_{prot_pos}_{aas.split('/')[0]}_{aas.split('/')[1]}_{hgvsp}" for
                               sym, prot_pos, aas, hgvsp in zip(sym_list, prot_pos, aas, hgvsp)]
            self.pop_data['isoform'] = isoform_id_list

            components = self.pop_data['isoform'].str.extract(
                r'(?P<gene_symbol>\w+)_(?P<prot_position>\d+)_(?P<ref_aa>\w+)_(?P<alt_aa>\w+)_(?P<isoform>\d+)')
            combined = components['gene_symbol'] + '_' + components['prot_position'].astype(str) + '_' + components[
                'ref_aa'] + '_' + components['alt_aa']
            self.pop_data['combined'] = combined
            self.pop_data['AA_ref'] = components['ref_aa']
            self.pop_data['AA_alt'] = components['alt_aa']
            self.pop_data = self.pop_data.loc[components['isoform'] == '1'].drop_duplicates(subset=['combined'])
            self.pop_data = self.pop_data.drop(columns=['combined'])

            var_pat_features = {}
            for index, row in tqdm(self.pop_data.iterrows(), total=self.pop_data.shape[0]):
                gene = row['Gene']
                ref_aa = row['AA_ref']
                alt_aa = row['AA_alt']
                ref_idx = aa_to_idx(ref_aa)
                alt_idx = aa_to_idx(alt_aa)
                pos = row['Protein_pos_shard']
                value = row['am_pathogenicity'] * row['AF']
                if np.isnan(value):
                    continue

                max_seq_len = self.config['hyperparameters']['pathogenicity_embedding']['max_seq_len']
                if gene not in var_pat_features.keys():
                    matrix_shape = (21 * 21, max_seq_len)
                    var_pat_matrix = np.zeros(matrix_shape, dtype=np.float32)
                    matrix_index = ref_idx * 21 + alt_idx
                    var_pat_matrix[matrix_index, pos - 1] = value
                    var_pat_features[gene] = var_pat_matrix
                else:
                    var_pat_matrix = var_pat_features[gene]
                    matrix_index = ref_idx * 21 + alt_idx
                    if var_pat_matrix[matrix_index, pos - 1] != 0:
                        print(f"Warning: Overwriting value at position {pos} for gene {gene}")
                    var_pat_matrix[matrix_index, pos - 1] = value
                    var_pat_features[gene] = var_pat_matrix
            #
            # # Convert all sparse matrices to CSR format
            # for gene, var_pat_matrix in var_pat_features.items():
            #     var_pat_features[gene] = var_pat_matrix.tocsr()

            with gzip.open('data/features/var_pat_features.pkl.gz', 'wb') as f:
                pkl.dump(var_pat_features, f)
            return var_pat_features
        else:
            with gzip.open('data/features/var_pat_features.pkl.gz', 'rb') as f:
                return pkl.load(f)

    def __pathogenicity_feature_extractor(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and average to gene-level using population
        statistics.
        """
        self.pop_data["uniprot_id"] = self.pop_data["SWISSPROT"].fillna(self.pop_data["TREMBL"])
        self.pop_data["uniprot_id"] = self.pop_data["SWISSPROT"].fillna(self.pop_data["TREMBL"])

        am = pd.read_csv("data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')

        am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']
        self.pop_data['ALT'] = self.pop_data['ALT'].str.split(',')
        self.pop_data = self.pop_data.explode('ALT')
        self.pop_data = self.pop_data[(self.pop_data['ALT'].str.len() == 1) & (self.pop_data['REF'].str.len() == 1)]
        self.pop_data = self.pop_data[self.pop_data['Consequence'] == 'missense_variant']
        pos_list = self.pop_data['POS'].tolist()
        chrom_list = self.pop_data['#CHROM'].tolist()
        ref_list = self.pop_data['REF'].tolist()
        alt_list = self.pop_data['ALT'].tolist()

        pos_list = [str(pos) for pos in pos_list]
        variant_id_list = [chrom + '_' + pos + '_' + ref + '_' + alt for chrom, pos, ref, alt in
                           zip(chrom_list, pos_list, ref_list, alt_list)]
        self.pop_data['variant_id'] = variant_id_list
        am = am[['am_pathogenicity', 'variant_id']]

        self.pop_data = self.pop_data.merge(am, on='variant_id', how='left')

        if not os.path.exists("../data/alphamissense/gh_am_data.pkl"):
            self.pop_data.to_pickle('../data/alphamissense/gh_am_data.pkl')

        variant_am_features = {}
        for index, row in tqdm(self.pop_data.iterrows()):
            gene = row['Gene']
            gh_af = row['AF_ELGH']
            am_score = row['am_pathogenicity']
            if gene not in variant_am_features.keys():
                variant_am_features[gene] = [np.nan_to_num(am_score * gh_af)]
            else:
                variant_am_features[gene].append(np.nan_to_num(am_score * gh_af))

        gene_am_features = {}
        for ensg, probs in variant_am_features.items():
            gene_am_features[ensg] = sum(probs) / len(probs)

        self.pathogenicity_features = gene_am_features


# Model preprocessing
class ModelPreprocessorEval:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.gc_data = data['train']['gc']
        self.go_data = data['train']['go']
        self.pvc_data = data['train'].get('pvc', None)
        if self.pvc_data is not None:
            self.pvc_data = self.pvc_data.copy()
        self.labels = data["labels"]
        self.test_labels = data["test_labels"]
        self.test_labels_per_source = data["test_labels_per_source"]
        self.genes = data['genes']
        self.num_features = data['num_features']
        self.test_data = data['test_data']
        self.test_genes = data['test_genes']
        self.train_genes, self.val_genes = train_test_split(self.genes, test_size=0.2,
                                                            random_state=config['hyperparameters']['seed'])
        self.class_prior = data['class_prior']
        self.torch_dtype = torch.bfloat16 if config['hyperparameters']['precision'] == 'bf16-mixed' else torch.float32
        torch.set_default_dtype(self.torch_dtype)

    def model_init(self):
        gc_train_raw = self.gc_data.loc[self.train_genes, :]
        gc_val_raw = self.gc_data.loc[self.val_genes, :]

        go_train_raw = self.go_data.loc[self.train_genes, :]
        go_val_raw = self.go_data.loc[self.val_genes, :]

        train_raw = {
            'gc': gc_train_raw,
            'go': go_train_raw,
        }

        val_raw = {
            'gc': gc_val_raw,
            'go': go_val_raw,
        }

        if self.pvc_data is not None:
            pvc_train_raw = {k: v for k, v in self.pvc_data.items() if k in self.train_genes}
            pvc_val_raw = {k: v for k, v in self.pvc_data.items() if k in self.val_genes}
            train_raw['pvc'] = pvc_train_raw
            val_raw['pvc'] = pvc_val_raw

        model, train_combined, val_combined, test_combined, hyperparameters, accelerator = self.initialise_model(
            train_raw,
            val_raw,
            self.labels,
            self.test_labels,
            self.train_genes,
            self.val_genes,
            self.test_labels_per_source,
            self.test_data,
            self.torch_dtype,
            self.config
        )

        return model, train_combined, val_combined, test_combined, hyperparameters, accelerator

    def initialise_model(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test,
                         torch_dtype, config):
        hyperparams = config['hyperparameters']
        (train_combined, val_combined, test_combined,
         num_samples_per_class) = self.normalise_data(train_raw, val_raw, labels, test_labels, train_genes, val_genes,
                                                      test_genes, test, torch_dtype, config)

        use_pvc = 'pvc' in train_raw

        if use_pvc:
            max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in train_raw['pvc'].keys()])
            with open(config['paths']['MISSENSE_MAP'], "rb") as f:
                missense_map = pkl.load(f)
            num_mutations = len(missense_map)
        else:
            max_genes_pvc = 0
            num_mutations = 0

        gc_features_dim = train_raw['gc'].shape[1]
        go_features_dim = train_raw['go'].shape[1]

        model = MultiModalLightningTargetIdentifier(
            config=config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=hyperparams['max_seq_len'],
            num_genes=max_genes_pvc,
            num_samples_per_class=num_samples_per_class,
            class_prior=self.class_prior,
            use_pvc=use_pvc
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        hyperparameters = dict(
            depth=hyperparams['depth_cls_head'],
            lr=hyperparams['lr_start'],
            batch_size=hyperparams['batch_size'],
            optimizer=hyperparams['optimizer'],
            epochs=hyperparams['epochs'],
            dropout=hyperparams['dropout'],
            gc_width=hyperparams['gc_width'],
            go_width=hyperparams['go_width'],
            weight_decay=hyperparams['weight_decay']
        )

        return model, train_combined, val_combined, test_combined, hyperparameters, accelerator

    @staticmethod
    def normalise_data(train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes, test_raw,
                       torch_dtype, config):
        hparams = config['hyperparameters']

        train_datasets = {}
        val_datasets = {}
        test_datasets = {key: {} for key in test_raw.keys()}
        scalers = {}

        for module_str, train_data in train_raw.items():
            if module_str != "pvc":
                val_norm = val_raw[module_str].values
                train_norm = train_data.values

                # scaler = MinMaxScaler()
                # train_norm = scaler.fit_transform(train_norm)
                # val_norm = scaler.transform(val_norm)
                # scalers[module_str] = scaler

                train_norm = {gene: train_norm[i] for i, gene in enumerate(train_genes)}
                val_norm = {gene: val_norm[i] for i, gene in enumerate(val_genes)}

                train_datasets[module_str] = dl.MultiModalData(
                    data=train_norm,
                    labels=labels,
                    gene_names=train_genes,
                    dtype=torch_dtype
                )

                val_datasets[module_str] = dl.MultiModalData(
                    data=val_norm,
                    labels=labels,
                    gene_names=val_genes,
                    dtype=torch_dtype
                )

                for key, modalities in test_raw.items():
                    # normed = scaler.transform(modalities[module_str].values)
                    # normed = {gene: normed[i] for i, gene in enumerate(test_genes[key][module_str])}
                    test_data = {gene: modalities[module_str].values[i] for i, gene in enumerate(test_genes[key])}
                    test_datasets[key][module_str] = dl.MultiModalData(
                        data=test_data,
                        labels=test_labels,
                        gene_names=test_genes[key],
                        dtype=torch_dtype,
                        test_source=key
                    )
            else:
                train_datasets[module_str] = dl.MultiModalData(
                    data=None,
                    labels=None,
                    gene_names=train_genes,
                    dtype=torch_dtype,
                    variant_data={'data': train_data, 'labels': labels},
                    max_variants=hparams['max_seq_len']
                )

                val_datasets[module_str] = dl.MultiModalData(
                    data=None,
                    labels=None,
                    gene_names=val_genes,
                    dtype=torch_dtype,
                    variant_data={'data': val_raw[module_str], 'labels': labels},
                    max_variants=hparams['max_seq_len']
                )

                for key, modalities in test_raw.items():
                    test_datasets[key][module_str] = dl.MultiModalData(
                        data=None,
                        labels=None,
                        gene_names=test_genes[key],
                        dtype=torch_dtype,
                        variant_data={
                            'data': modalities[module_str],
                            'labels': test_labels,
                            'test_source': key
                        },
                        max_variants=hparams['max_seq_len'],
                        test_source=key
                    )

        #get from the test_datasets['pharos']['gc'] the gene names and the labels
        test_gene_names = test_datasets['pharos']['gc'].gene_names
        test_labels = test_datasets['pharos']['gc'].labels
        subset_labels_test = {gene: test_labels[gene] for gene in test_gene_names if gene in test_labels}
        subset_labels = {gene: labels[gene] for gene in test_gene_names if gene in labels}

        train_loader = dl.MultiModalDataLoader(
            datasets=train_datasets,
            batch_size=hparams['batch_size'],
            shuffle=True
        )

        val_loader = dl.MultiModalDataLoader(
            datasets=val_datasets,
            batch_size=hparams['batch_size'],
            shuffle=False
        )

        test_loaders = {}
        for key in test_raw.keys():
            if len(next(iter(test_datasets[key].values()))) <= 1000:
                test_loaders[key] = dl.MultiModalDataLoader(
                    datasets=test_datasets[key],
                    batch_size=len(next(iter(test_datasets[key].values()))),
                    shuffle=False
                )
            else:
                test_loaders[key] = dl.MultiModalDataLoader(
                    datasets=test_datasets[key],
                    batch_size=hparams['batch_size'],
                    shuffle=False
                )

        num_maj_samples, num_min_samples = next(iter(train_datasets.values())).samples_per_class()

        return train_loader, val_loader, test_loaders, (num_maj_samples, num_min_samples)


class ModelPreprocessorInference:
    def __init__(self, config, consolidated_data, pvc_data, gene_names):
        self.config = config
        self.data_raw = consolidated_data
        self.pvc_data = pvc_data
        self.gene_names = gene_names
        self.torch_dtype = torch.bfloat16 if config['hyperparameters']['precision'] == 'bf16-mixed' else torch.float32
        torch.set_default_dtype(self.torch_dtype)

    def model_init(self):
        # Create the unlabeled loader
        unlabeled_loader, num_samples = self.create_unlabeled_loader(
            self.data_raw,
            self.pvc_data,
            self.gene_names,
            self.torch_dtype,
            self.config
        )

        test_loaders = self.create_test_loaders(
            self.data_raw,
            self.pvc_data,
            self.torch_dtype,
            self.config
        )

        # Initialize model dimensions
        gc_features_dim = self.data_raw['gc'].shape[1] - 1 if 'target' in self.data_raw['gc'].columns else self.data_raw['gc'].shape[1]
        go_features_dim = self.data_raw['go'].shape[1] - 1 if 'target' in self.data_raw['go'].columns else self.data_raw['go'].shape[1]

        with open(self.config['paths']['MISSENSE_MAP'], 'rb') as f:
            missense_map = pkl.load(f)
        num_mutations = len(missense_map)

        model = MultiModalLightningTargetIdentifier(
            config=self.config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=self.config['hyperparameters']['max_seq_len'],
            num_genes=len(self.gene_names),
            num_samples_per_class=(len(self.gene_names), 0),  # all unlabeled = class 0
            class_prior={0: 1.0, 1: 0.0}
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        return unlabeled_loader, test_loaders, gc_features_dim, go_features_dim, num_genes, num_mutations

    def initialise_model(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes_dict,
                         test_data_dict, torch_dtype, config):
        hyperparams = config['hyperparameters']
        unlabeled_loader, num_samples_per_class = ModelPreprocessorInference.create_unlabeled_loader(
            train_raw, val_raw, labels, test_labels, train_genes, val_genes,
            test_genes_dict, test_data_dict, torch_dtype, config
        )

        # Determine max_genes_pvc (max variants per gene in training set for PVC)
        # Ensure pvc data in train_raw is not empty and genes have data
        max_genes_pvc = 0
        if 'pvc' in train_raw and train_raw['pvc']:
            non_empty_pvc_genes = [gene for gene in train_raw['pvc'] if train_raw['pvc'][gene].nelement() > 0]
            if non_empty_pvc_genes:
                max_genes_pvc = max([train_raw['pvc'][gene].shape[0] for gene in non_empty_pvc_genes])

        with open(config['paths']['MISSENSE_MAP'], "rb") as f:
            missense_map = pkl.load(f)
        num_mutations = len(missense_map)

        if 'target' in train_raw['gc'].columns:
            gc_features_dim = train_raw['gc'].shape[1] - 1  # -1 for target column if present before normalise
        else:
            gc_features_dim = train_raw['gc'].shape[1]

        if 'target' in train_raw['go'].columns:
            go_features_dim = train_raw['go'].shape[1] - 1
        else:
            go_features_dim = train_raw['go'].shape[1]

        model = MultiModalLightningTargetIdentifier(
            config=config,
            num_features_gc=gc_features_dim,
            num_features_go=go_features_dim,
            num_mutations=num_mutations,
            max_seq_len=hyperparams['max_seq_len'],
            num_genes=max_genes_pvc,  # This might need to be num_unique_gene_ids for embedding
            num_samples_per_class=num_samples_per_class,
            class_prior=self.class_prior  # Use actual class_prior from data_split
        )

        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

        hyperparameters_log = dict(
            depth=hyperparams['depth_cls_head'],
            lr=hyperparams['lr_start'],
            batch_size=hyperparams['batch_size'],
            optimizer=hyperparams['optimizer'],
            epochs=hyperparams['epochs'],
            dropout=hyperparams['dropout'],
            gc_width=hyperparams['gc_width'],
            go_width=hyperparams['go_width'],
            weight_decay=hyperparams['weight_decay']
        )

        return model, unlabeled_loader, hyperparameters_log, accelerator

    @staticmethod
    def create_unlabeled_loader(consolidated_data, pvc_data, gene_names, torch_dtype, config):
        hparams = config['hyperparameters']

        unlabeled_datasets = {}

        # gc and go modalities
        for modality in ['gc', 'go']:
            data_dict = {
                gene: consolidated_data[modality].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in gene_names
            }

            unlabeled_datasets[modality] = dl.MultiModalData(
                data=data_dict,
                labels={gene: 0 for gene in gene_names},
                gene_names=gene_names,
                dtype=torch_dtype
            )

        # pvc modality
        unlabeled_datasets['pvc'] = dl.MultiModalData(
            data=None,
            labels=None,
            gene_names=gene_names,
            dtype=torch_dtype,
            variant_data={'data': {gene: pvc_data[gene] for gene in gene_names},
                          'labels': {gene: 0 for gene in gene_names}},
            max_variants=hparams['max_seq_len']
        )

        dataset_size = len(gene_names)

        unlabeled_loader = dl.MultiModalDataLoader(
            datasets=unlabeled_datasets,
            batch_size=config['hyperparameters']['batch_size'],  # all at once
            shuffle=False
        )

        return unlabeled_loader, dataset_size

    @staticmethod
    def create_test_loaders(config, consolidated_data, pvc_data, torch_dtype):
        """
        Create dataloaders for test genes (approved targets) from pfam, rcnt, and pharos test sets.

        Args:
            config: Configuration dictionary
            consolidated_data: Dictionary with 'gc', 'go' dataframes containing ALL genes (train + test)
            pvc_data: Dictionary mapping gene_id -> variant tensor
            torch_dtype: Torch data type string ('bf16-mixed' or 'float32')

        Returns:
            Dictionary of test loaders: {'pfam': loader, 'rcnt': loader, 'pharos': loader}
        """
        import dataloader as dl
        import pickle

        # Convert dtype string to torch dtype
        if torch_dtype == 'bf16-mixed':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        test_loaders = {}

        # Load test gene IDs from pickle file
        test_labels_file = config['paths'].get('TEST_LABELS_FILE')

        try:
            with open(test_labels_file, 'rb') as f:
                test_gene_ids = pickle.load(f)
            print(f"  Loaded test gene IDs from {test_labels_file}")
        except FileNotFoundError:
            print(f"  ⚠️ Test labels file not found: {test_labels_file}")
            return {}

        # Create loader for each test set
        for test_name in ['pfam', 'rcnt', 'pharos']:
            if test_name not in test_gene_ids:
                print(f"  Skipping {test_name}: not in test labels file")
                continue

            test_genes = test_gene_ids[test_name]

            # Filter to genes available in all modalities
            available_genes = [
                gene for gene in test_genes
                if gene in consolidated_data['gc'].index and
                   gene in consolidated_data['go'].index and
                   gene in pvc_data
            ]

            if len(available_genes) == 0:
                print(f"  Skipping {test_name}: no genes available in data ({len(test_genes)} total)")
                continue

            print(f"  Creating {test_name} loader: {len(available_genes)}/{len(test_genes)} genes")

            # Create datasets (same pattern as unlabeled)
            gc_data_dict = {
                gene: consolidated_data['gc'].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in available_genes
            }

            go_data_dict = {
                gene: consolidated_data['go'].loc[gene].drop(labels=['target'], errors='ignore').values.flatten()
                for gene in available_genes
            }

            # Labels are all 1 (positive/approved targets)
            labels = {gene: 1 for gene in available_genes}

            gc_dataset = dl.MultiModalData(
                data=gc_data_dict,
                labels=labels,
                gene_names=available_genes,
                dtype=dtype
            )

            go_dataset = dl.MultiModalData(
                data=go_data_dict,
                labels=labels,
                gene_names=available_genes,
                dtype=dtype
            )

            pvc_dataset = dl.MultiModalData(
                data=None,
                labels=None,
                gene_names=available_genes,
                dtype=dtype,
                variant_data={
                    'data': {gene: pvc_data[gene] for gene in available_genes},
                    'labels': labels
                },
                max_variants=config['hyperparameters']['max_seq_len']
            )

            # Create loader
            test_loaders[test_name] = dl.MultiModalDataLoader(
                datasets={'gc': gc_dataset, 'go': go_dataset, 'pvc': pvc_dataset},
                batch_size=min(32, len(available_genes)),
                shuffle=False
            )

        print(f"✓ Created {len(test_loaders)} test loaders: {list(test_loaders.keys())}")
        return test_loaders


def extract_pvc_features(gene, pvc_data, max_variants=100):
    """
    Extract fixed-length features from variable-length protein variant call data for logistic regression.

    Args:
        gene: Gene identifier
        pvc_data: Dictionary mapping genes to torch tensors of shape (N_g, 4) where N_g varies per gene
                  The 4 features are: pathogenicity, position, variant identity, gene identity
        max_variants: Maximum number of variants to consider (for padding/truncation)

    Returns:
        numpy array of fixed-length features derived from the PVC data
    """
    # If gene not in pvc_data, return zeros
    if gene not in pvc_data:
        return np.zeros(10)  # Return zeros for all features

    # Get the tensor for this gene
    tensor = pvc_data[gene]

    # Convert to numpy for easier manipulation
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    # Handle empty tensor
    if tensor.shape[0] == 0:
        return np.zeros(10)

    # Extract each feature column
    pathogenicity = tensor[:, 0]
    positions = tensor[:, 1]
    variant_ids = tensor[:, 2]
    gene_ids = tensor[:, 3]

    # Statistical features (10 total)
    features = []

    # 1. Basic count statistics
    num_variants = len(pathogenicity)
    features.append(num_variants)  # Total number of variants

    # 2. Pathogenicity statistics
    features.append(np.mean(pathogenicity))  # Mean pathogenicity
    features.append(np.std(pathogenicity) if num_variants > 1 else 0)  # Std dev of pathogenicity
    features.append(np.max(pathogenicity) if num_variants > 0 else 0)  # Max pathogenicity
    features.append(np.min(pathogenicity) if num_variants > 0 else 0)  # Min pathogenicity

    # 3. Position statistics
    features.append(np.mean(positions))  # Mean position
    features.append(np.std(positions) if num_variants > 1 else 0)  # Std dev of positions
    features.append(len(np.unique(positions)) / max(1, num_variants))  # Position diversity ratio

    # 4. Position distribution features
    if num_variants > 1:
        # Calculate distance between consecutive positions
        sorted_positions = np.sort(positions)
        position_diffs = np.diff(sorted_positions)
        features.append(np.mean(position_diffs))  # Mean distance between variants
        features.append(np.std(position_diffs))  # Std dev of distances between variants
    else:
        features.extend([0, 0])  # Placeholder values when not enough variants

    return np.array(features)


class LogisticRegressionPreprocessor:
    def __init__(self, config, data):
        """
        Initialize the preprocessor for logistic regression model

        Args:
            config: Configuration dictionary
            data: Data dictionary containing train, test data and other metadata
        """
        self.config = config
        self.data = data
        self.gc_data = data['train']['gc']
        self.go_data = data['train']['go']
        self.pvc_data = data['train'].get('pvc', None)
        if self.pvc_data is not None:
            self.pvc_data = self.pvc_data.copy()
        self.labels = self.data["labels"]
        self.test_labels = data["test_labels"]
        self.genes = data['genes']
        self.num_features = data['num_features']
        self.test_data = data['test_data']
        self.test_genes = data['test_genes']
        self.train_genes, self.val_genes = train_test_split(self.genes, test_size=0.2,
                                                            random_state=config['hyperparameters']['seed'])
        self.class_prior = data['class_prior']
        self.scalers = {}
        self.max_variants = config['hyperparameters'].get('max_seq_len', 100)

    def prepare_features(self):
        """
        Prepare features for logistic regression by processing and combining
        gene-centric (gc), gene ontology (go), and protein variant calls (pvc) data

        Returns:
            Dictionary containing processed train, validation, and test data
            with features and labels
        """
        print("Preparing features for logistic regression...")

        # Split data into train and validation sets
        gc_train_raw = self.gc_data.loc[self.train_genes, :]
        gc_val_raw = self.gc_data.loc[self.val_genes, :]

        go_train_raw = self.go_data.loc[self.train_genes, :]
        go_val_raw = self.go_data.loc[self.val_genes, :]

        train_raw = {
            'gc': gc_train_raw,
            'go': go_train_raw,
        }

        val_raw = {
            'gc': gc_val_raw,
            'go': go_val_raw,
        }

        if self.pvc_data is not None:
            pvc_train_raw = {k: v for k, v in self.pvc_data.items() if k in self.train_genes}
            pvc_val_raw = {k: v for k, v in self.pvc_data.items() if k in self.val_genes}
            train_raw['pvc'] = pvc_train_raw
            val_raw['pvc'] = pvc_val_raw

        # Process and combine the features
        processed_data = self.process_and_combine_features(
            train_raw,
            val_raw,
            self.labels,
            self.test_labels,
            self.train_genes,
            self.val_genes,
            self.test_genes,
            self.test_data
        )

        return processed_data

    def process_and_combine_features(self, train_raw, val_raw, labels, test_labels, train_genes, val_genes, test_genes,
                                     test_raw):
        """
        Process and combine features from different modalities for logistic regression

        Args:
            train_raw: Raw training data for different modalities
            val_raw: Raw validation data for different modalities
            labels: Training and validation labels
            test_labels: Test labels
            train_genes: List of genes in training set
            val_genes: List of genes in validation set
            test_genes: Dictionary of genes in test sets
            test_raw: Raw test data for different modalities

        Returns:
            Dictionary containing processed data for training, validation, and testing
        """
        # Initialize dictionaries to store processed data
        train_features = {}
        train_labels_list = []
        val_features = {}
        val_labels_list = []
        test_features = {test_set: {} for test_set in test_raw.keys()}
        test_labels_dict = {test_set: [] for test_set in test_raw.keys()}

        # Process gene-centric (gc) and gene ontology (go) features
        for module_str in ['gc', 'go']:
            # Scale the features
            # scaler = MinMaxScaler()
            # train_norm = scaler.fit_transform(train_raw[module_str].values)
            # val_norm = scaler.transform(val_raw[module_str].values)
            # self.scalers[module_str] = scaler
            train_feat = train_raw[module_str].values
            val_feat = val_raw[module_str].values

            train_features[module_str] = {train_genes[i]: train_feat[i] for i in range(len(train_genes))}
            val_features[module_str] = {val_genes[i]: val_feat[i] for i in range(len(val_genes))}

            # Process test data for each test set
            for test_set, modalities in test_raw.items():
                test_feat_df = modalities[module_str]
                test_feat = test_feat_df.values
                test_feat_genes = test_feat_df.index.tolist()
                test_features[test_set][module_str] = {test_feat_genes[i]: test_feat[i] for i in
                                                       range(len(test_feat_genes))}

        # Process protein variant calls (pvc) features using the dedicated function
        print("Processing PVC features...")
        train_features['pvc'] = self.process_pvc_batch(train_genes, train_raw['pvc'], self.max_variants)
        val_features['pvc'] = self.process_pvc_batch(val_genes, val_raw['pvc'], self.max_variants)

        for test_set, modalities in test_raw.items():
            test_genes_list = list(test_features[test_set]['gc'].keys())  # Using gc gene list for reference
            test_features[test_set]['pvc'] = self.process_pvc_batch(test_genes_list, modalities['pvc'],
                                                                    self.max_variants)

        # Combine features from all modalities into a single feature vector for each gene
        print("Combining features from all modalities...")
        combined_train_features = {}
        combined_val_features = {}
        combined_test_features = {test_set: {} for test_set in test_raw.keys()}

        # Create feature arrays ensuring all features are available
        for gene in train_genes:
            if gene in labels:
                gc_feat = train_features['gc'].get(gene, np.zeros(train_raw['gc'].shape[1]))
                go_feat = train_features['go'].get(gene, np.zeros(train_raw['go'].shape[1]))
                pvc_feat = train_features['pvc'].get(gene, np.zeros(10))  # 10 PVC features
                combined_train_features[gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                train_labels_list.append((gene, labels[gene]))

        for gene in val_genes:
            if gene in labels:
                gc_feat = val_features['gc'].get(gene, np.zeros(val_raw['gc'].shape[1]))
                go_feat = val_features['go'].get(gene, np.zeros(val_raw['go'].shape[1]))
                pvc_feat = val_features['pvc'].get(gene, np.zeros(10))
                combined_val_features[gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                val_labels_list.append((gene, labels[gene]))

        for test_set in test_raw.keys():
            test_gc_shape = test_raw[test_set]['gc'].shape[1]
            test_go_shape = test_raw[test_set]['go'].shape[1]

            for gene in list(test_features[test_set]['gc'].keys()):
                if gene in test_labels:
                    gc_feat = test_features[test_set]['gc'].get(gene, np.zeros(test_gc_shape))
                    go_feat = test_features[test_set]['go'].get(gene, np.zeros(test_go_shape))
                    pvc_feat = test_features[test_set]['pvc'].get(gene, np.zeros(10))
                    combined_test_features[test_set][gene] = np.concatenate([gc_feat, go_feat, pvc_feat])
                    test_labels_dict[test_set].append((gene, test_labels[gene]))

        # Convert to numpy arrays for scikit-learn
        X_train = np.array([combined_train_features[gene[0]] for gene in train_labels_list])
        y_train = np.array([label[1] for label in train_labels_list])

        X_val = np.array([combined_val_features[gene[0]] for gene in val_labels_list])
        y_val = np.array([label[1] for label in val_labels_list])

        X_test = {}
        y_test = {}
        test_gene_lists = {}
        for test_set in test_raw.keys():
            X_test[test_set] = np.array(
                [combined_test_features[test_set][gene[0]] for gene in test_labels_dict[test_set]])
            y_test[test_set] = np.array([label[1] for label in test_labels_dict[test_set]])
            test_gene_lists[test_set] = [gene[0] for gene in test_labels_dict[test_set]]

        # Handle missing values
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)
        for test_set in X_test:
            X_test[test_set] = np.nan_to_num(X_test[test_set])

        # Log feature dimensions
        print(f"Features dimensions - Train: {X_train.shape}, Val: {X_val.shape}")
        for test_set in X_test:
            print(f"Test {test_set}: {X_test[test_set].shape}")

        # Document the feature set composition for interpretability
        feature_composition = {
            'gc_features': train_raw['gc'].shape[1],
            'go_features': train_raw['go'].shape[1],
            'pvc_features': 10,
            'total_features': X_train.shape[1]
        }
        print(f"Feature composition: {feature_composition}")

        return {
            'train': {'X': X_train, 'y': y_train, 'genes': [gene[0] for gene in train_labels_list]},
            'val': {'X': X_val, 'y': y_val, 'genes': [gene[0] for gene in val_labels_list]},
            'test': {test_set: {'X': X_test[test_set], 'y': y_test[test_set], 'genes': test_gene_lists[test_set]}
                     for test_set in test_raw.keys()},
            'feature_composition': feature_composition
        }

    @staticmethod
    def process_pvc_batch(genes, pvc_data, max_variants=100):
        """
        Process a batch of genes to extract PVC features.

        Args:
            genes: List of gene identifiers
            pvc_data: Dictionary mapping genes to variant data tensors
            max_variants: Maximum number of variants to consider

        Returns:
            Dictionary mapping genes to extracted feature vectors
        """
        features = {}
        for gene in genes:
            features[gene] = extract_pvc_features(gene, pvc_data, max_variants)
        return features


# Deprecated
class __WildtypeLoader:
    """
    This class is deprecatated. Wildtype gets loaded in the VariantLoader class. This class is kept for reference and
    archive purposes.
    """

    def __init__(self, uniparc_path, msa_output):
        """
        :param uniparc_path: path to the UNIPARC dataset.
        :param msa_output: path to where the MSA .fasta file will be saved.
        """
        self.uniparc_path = uniparc_path
        self.msa_output = msa_output
        self.msa_file_name = msa_output.split(os.path.sep)[-1]
        self.uniparc_col_name = "UNIPARC"
        self.gene_col_name = "SYMBOL"
        self._init_verification()

    def _init_verification(self):
        """
        Check whether the path points to a valid .csv file. Also check whether the msa output ends in ".fasta".
        """
        if not self.uniparc_path.endswith(".csv"):
            raise TypeError("The path to the UNIPARC dataset must point to a .csv file.")
        if not self.msa_output.endswith(".fasta"):
            raise TypeError("The output file must be of type .fasta.")

    def data_reader(self):
        """
        Read in the raw .csv data and check whether the data contains the required columns.
        :return: raw_data as a pandas dataframe.
        """
        _raw_data = pd.read_csv(self.uniparc_path, sep="\t")
        if self.uniparc_col_name not in _raw_data.columns:
            raise TypeError(f"Change the column name of the column containing the uniparc ids to "
                            f"'{self.uniparc_col_name}'.")
        if self.gene_col_name not in _raw_data.columns:
            raise TypeError(f"Change the column name of the column containing the gene names to "
                            f"'{self.gene_col_name}'.")
        return _raw_data

    def parse_data(self, data):
        """
        Get UNIPARC ids and gene symbols from the dataset and put them in a dictionary.
        """
        uniprot_ids = data[self.uniparc_col_name].tolist()
        gene_names = data[self.gene_col_name].tolist()
        uniprot_ids_dict = {}
        for i in range(len(uniprot_ids)):
            if gene_names[i] not in uniprot_ids_dict:
                uniprot_ids_dict[gene_names[i]] = [uniprot_ids[i]]
            else:
                uniprot_ids_dict[gene_names[i]].append(uniprot_ids[i])
        return self._get_msa(uniprot_ids_dict)

    def _get_msa(self, uniprot_ids_dict):
        """
        Get the multiple sequence alignment from the UNIPROT database.
        :param uniprot_ids_dict: dictionary of UNIPROT IDs and gene names.
        :return: processed MSA data.
        """
        cont_idx, num_genes = self._file_tracker(uniprot_ids_dict)
        if cont_idx == 0:
            with open(self.msa_output, "w") as f:
                f.write("")
            f.close()
        elif cont_idx == num_genes:
            print("Data preprocessing completed!")
            return list(SeqIO.parse(self.msa_output, "fasta"))
        else:
            print("Data preprocessing incomplete. Continuing from where it left off.")
            with open("preprocessing_log.txt", "r") as f:
                last_processed_gene, last_processed_gene = f.read().split("\t")
            f.close()
            new_uniprot_ids_dict = {}
            found_gene = False
            for gene_name in uniprot_ids_dict:
                if found_gene:
                    new_uniprot_ids_dict[gene_name] = uniprot_ids_dict[gene_name]
                if gene_name == last_processed_gene:
                    new_uniprot_ids_dict[gene_name] = uniprot_ids_dict[gene_name][uniprot_ids_dict[gene_name].index(
                        last_processed_gene) + 1:]
                    found_gene = True
            uniprot_ids_dict = new_uniprot_ids_dict
        print("Parsing the data...")
        with open(self.msa_output, "a") as f:
            for gene_name in tqdm(uniprot_ids_dict):
                for uniprot_id in uniprot_ids_dict[gene_name]:
                    url = f"https://www.uniprot.org/uniparc/{uniprot_id}.fasta"
                    response = requests.get(url)
                    if response.status_code == 200:
                        fasta_msa = response.text
                    elif str(uniprot_id) == "nan":
                        continue
                    else:
                        raise ValueError(f"{response.status_code} Could not retrieve the MSA for gene {gene_name}.")
                    fasta_msa = fasta_msa.replace("status=active", "")
                    fasta_msa = fasta_msa.replace("status=inactive", "")
                    fasta_msa = f"{fasta_msa[0:14]}|{gene_name}{fasta_msa[14:]}"
                    f.write(fasta_msa)
                    with open("preprocessing_log.txt", "w") as log:
                        log.write(f"{gene_name}\t{uniprot_id}")
                    log.close()
        f.close()
        print("Data preprocessing completed!")
        return list(SeqIO.parse(self.msa_output, "fasta"))

    def _file_tracker(self, uniprot_ids_dict):
        """
        Track whether there already exists output and if it is complete or not.
        """
        num_genes = 0
        if self.msa_file_name in os.listdir():
            with open(self.msa_file_name, "r") as f:
                count = f.read().count(">")
            f.close()
            for gene_name in uniprot_ids_dict:
                num_genes += len(uniprot_ids_dict[gene_name])
            if count != num_genes:
                cont_idx = count - 1
            else:
                cont_idx = num_genes
        else:
            cont_idx = 0
        return cont_idx, num_genes


class __MissenseVariantPreprocessor:
    """
    DEPRACATED: This class is deprecated. All the preprocessing is now done in the GeneCharacterisation class. This is
    because we switched away from using VariPred pathogenicity in favour of AlphaMissense pathogenicity.
    """

    def __init__(self, config=None, preprocess=False, train=False, predict=False, evaluation=False):
        self.config = config
        parser = argparse.ArgumentParser(description='Script to process variants')
        parser.add_argument('--data', type=str)
        self.args = parser.parse_args()
        if self.args.data is not None:
            self.elgh_path = self.args.data
        else:
            self.elgh_path = self.config['paths']['ALL_GH'].strip("\n")
        self.genome_path = self.config['paths']['GENOME_PATH']
        self.variant_cols = ["#CHROM", "POS", "REF", "Allele", "SYMBOL", "Gene", "HGVSp", "AF_ELGH", "UNIPARC",
                             "SWISSPROT", "TREMBL", "Protein_position", "Amino_acids"]
        self.variant_data = self.load_gh_data()
        self.variant_data = self.variant_data.rename(columns={'Allele': 'ALT'})
        self.variant_data["uniprot_id"] = self.variant_data["SWISSPROT"].fillna(self.variant_data["TREMBL"])
        self.variant_data["uniprot_id"] = self.variant_data["SWISSPROT"].fillna(self.variant_data["TREMBL"])

        try:
            am = self.load_am_data()
            am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']
            self.variant_data['variant_id'] = self.variant_data['#CHROM'] + '_' + self.variant_data['POS'].astype(str) + \
                                              '_' + self.variant_data['REF'] + '_' + self.variant_data['ALT']
            am = am[['am_pathogenicity', 'variant_id']]
            vp_pretrained_data = pd.read_csv("data/VariPred/varipred_output_data_pretrained.csv", sep="\t")
            vp_pretrained_data = vp_pretrained_data.rename(columns={'target_id': 'varipred_id', 'probability':
                'vp_pathogenicity'})
            # remove the classification column from the pretrained data
            vp_pretrained_data = vp_pretrained_data.drop(columns=['classification'])
            self.variant_data = self.variant_data.merge(am, on='variant_id')
            self.variant_data = self.variant_data.merge(vp_pretrained_data, on='varipred_id')
        except FileNotFoundError:
            print("AlphaMissense not found. Comparison done and deleted data for storage purposes, or data still needs"
                  " to be downloaded.")

        if preprocess:
            self.process_variants_proteomic()
        elif len(os.listdir('data/VariPred/input')) == self.config['varipred']['num_batches']:
            print("Variants already processed.")
        else:
            print('Not all variants are preprocessed yet. Put the preprocess flag to True in the MissenseVariantLoader '
                  'and run again.')
        if train:
            # NOTE: In order to prepare the data for training, first all the datasets generated in evaluation need to be
            # loaded.

            # raw_data here is dummy file, normally should be done on cluster for all batch files

            raw_data = pd.read_csv("data/elgh/train_batch_mivas/variant_data_396.csv")
            train = self.train_test_val_loader(raw_data)
            utils.run_shell_script(self.config['paths']['VP_TRAINING_PATH'])

        if predict:
            if self.args.varipred_input is not None:
                variant_files = os.listdir(self.args.varipred_input)
                data_dir = self.args.varipred_input.split('/')[4]
                variant_files = sorted(variant_files, key=utils.extract_number)
                for file in variant_files:
                    if file.endswith('.csv'):
                        self.predict_pathogenicity(f"{data_dir}/{file[:-4]}")
            else:
                self.predict_pathogenicity()

        varipred_output = utils.preprocess_varipred_output(self.config['paths']['VP_OUTPUT_PATH'])

        if os.path.exists("data/elgh/varipred_elgh_data.csv"):
            self.variant_data = pd.read_csv("data/elgh/varipred_elgh_data.csv", sep="\t")
        else:
            self.variant_data = utils.combine_varipred_elgh(varipred_output, self.variant_data)

        if evaluation:
            clinvar_data = pd.read_csv("data/clinvar/variant_summary.txt", sep="\t")
            utils.varipred_evaluation(self.variant_data, clinvar_data, posthoc=False)

    def load_gh_data(self):
        """
        Load the variant data and the reference genome.
        """
        if self.args.data is not None:
            variant_data = pd.read_csv(self.elgh_path)
        else:
            variant_data = pd.read_csv(self.elgh_path, sep="\t")
        variant_data = variant_data.loc[:, ~variant_data.columns.str.contains('^Unnamed')]
        variant_data = variant_data[self.variant_cols]
        return variant_data

    @staticmethod
    def load_am_data():
        am = pd.read_csv("data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')
        return am

    @staticmethod
    def fetch_amino_acid_sequence(uniparc_id, mt_aa, aa_index):
        url = f"https://www.uniprot.org/uniparc/{uniparc_id}.fasta"
        headers = {"Accept": "text/plain"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            wt_sequence = "".join(response.text.split("\n")[1:])
            var_sequence = wt_sequence[:aa_index] + mt_aa + wt_sequence[aa_index + 1:]

            return wt_sequence, var_sequence
        else:
            raise LookupError(f"Could not find sequence for {uniparc_id}")

    def process_variants_proteomic(self):
        seq_ids = []
        sequence_table = []
        for i in tqdm(range(len(self.variant_data))):
            uniparc_id = str(self.variant_data["UNIPARC"].iloc[i])
            if '-' in str(self.variant_data["Protein_position"].iloc[i]) or uniparc_id == "nan":
                continue
            else:
                aa_index = int(self.variant_data["Protein_position"].iloc[i]) - 1
                wt_aa = self.variant_data["Amino_acids"].iloc[i].split("/")[0]
                if '/' in self.variant_data["Amino_acids"].iloc[i]:
                    mt_aa = self.variant_data["Amino_acids"].iloc[i].split("/")[1]
                else:
                    mt_aa = self.variant_data["Amino_acids"].iloc[i]
                seq_id = f"{self.variant_data['SYMBOL'].iloc[i]}_{aa_index}_{wt_aa}_{mt_aa}"
                gc.collect()
                if len(mt_aa) > 1:
                    continue
                else:
                    if seq_id not in seq_ids:
                        variant_seq, wildtype_seq = self.fetch_amino_acid_sequence(uniparc_id, mt_aa, aa_index)
                        seq_ids.append(seq_id)
                        sequence_table.append([seq_id, aa_index, wt_aa, mt_aa, wildtype_seq, variant_seq])
        sequence_table = pd.DataFrame(sequence_table, columns=["target_id", "aa_index", "wt_aa", "mt_aa", "wt_seq",
                                                               "mt_seq"])
        variants_id = str(self.elgh_path.split("_")[-1].split(".")[0])
        sequence_table.to_csv(f"data/VariPred/input/variants_{variants_id}.csv", index=False)

    def train_test_val_loader(self, data, downsampling=True):
        train_files = os.listdir("data/VariPred/train/")
        if len(train_files) != 1000:
            data = data[["vp_cv_id", "UNIPARC", "Protein_position", "Amino_acids", "ClinSigSimple"]]
            wt_aa = data["Amino_acids"].str.split("/", expand=True)[0]
            mt_aa = data["Amino_acids"].str.split("/", expand=True)[1]
            data["wt_aa"] = wt_aa
            data["mt_aa"] = mt_aa
            data = data.drop(columns=["Amino_acids"])
            data = data.rename(
                columns={"vp_cv_id": "target_id", "Protein_position": "aa_index", "ClinSigSimple": "label"})
            cols = ["target_id", "UNIPARC", "aa_index", "wt_aa", "mt_aa", "label"]
            data = data[cols]
            seq_ids = []
            sequence_table = []
            for i in tqdm(range(len(data))):
                seq_id = data["target_id"].iloc[i]
                if seq_id not in seq_ids:
                    wt_seq, mt_seq = self.fetch_amino_acid_sequence(data["UNIPARC"].iloc[i], data["mt_aa"].iloc[i],
                                                                    data["aa_index"].iloc[i])
                    seq_ids.append(seq_id)
                    sequence_table.append(
                        [seq_id, data["UNIPARC"].iloc[i], data["aa_index"].iloc[i], data["wt_aa"].iloc[i],
                         data["mt_aa"].iloc[i], wt_seq, mt_seq, data["label"].iloc[i]])
            sequence_table = pd.DataFrame(sequence_table,
                                          columns=["target_id", "uniparc_id", "aa_index", "wt_aa", "mt_aa",
                                                   "wt_seq", "mt_seq", "label"])
            variants_id = str(self.elgh_path.split("_")[-1].split(".")[0])
            sequence_table.to_csv(f"data/VariPred/train/variants_{variants_id}.csv", index=False)
        else:
            if not os.path.exists("data/VariPred/all_train.csv"):
                utils.combine_train_files()
            elif not os.path.exists("data/VariPred/train.csv"):
                raw_train = pd.read_csv("data/VariPred/all_train.csv")
                df = raw_train.copy()
                df = df[df.target_id != 'target_id']
                df = df.rename(columns={'target_id': 'seq_id'})
                train, test = train_test_split(df, test_size=0.1, random_state=555, stratify=df['label'])
                train.to_csv("data/VariPred/train.csv", index=False)
                # val.to_csv("data/VariPred/val.csv", index=False)
                test.to_csv("data/VariPred/test.csv", index=False)
                print(f"Train and test data loaded with size: {len(train)} and {len(test)}")
                return train, test
            elif downsampling and not os.path.exists("data/VariPred/train_downsample.csv"):
                train = pd.read_csv("data/VariPred/train.csv")
                test = pd.read_csv("data/VariPred/test.csv")
                # val = pd.read_csv("data/VariPred/test.csv")
                train = utils.downsampler(train)
                test = utils.downsampler(test)
                # val = utils.downsampler(val)
                train.to_csv("data/VariPred/train_downsample.csv", index=False)
                test.to_csv("data/VariPred/test_downsample.csv", index=False)
                # val.to_csv("data/VariPred/val_downsample.csv", index=False)
                print(f"Train and test data downsampled with sizes: {len(train)} {len(test)}")
                return train, test
            elif downsampling:
                train = pd.read_csv("data/VariPred/train_downsample.csv")
                # val = pd.read_csv("data/VariPred/val_downsample.csv")
                test = pd.read_csv("data/VariPred/test_downsample.csv")
                print(f"Downsampled train and test data loaded with sizes: {len(train)}, {len(test)}")
                return train, test
            else:
                train = pd.read_csv("data/VariPred/train.csv")
                val = pd.read_csv("data/VariPred/val.csv")
                test = pd.read_csv("data/VariPred/test.csv")
                print(f"Train and test data loaded with sizes: {len(train)}, {len(val)}, {len(test)}")
                return train, val, test

    def predict_pathogenicity(self, variant_file=None):
        if variant_file is not None:
            # data_folder = self.args.varipred_input.split('/')[3]
            # file = f"{data_folder}/{variant_file}"
            file = f"{variant_file}"
            utils.run_shell_script(self.config['paths']['VP_INFERENCE_PATH'], file)
        else:
            utils.run_shell_script(self.config['paths']['VP_INFERENCE_PATH'])

    def __process_variants_genomic(self):
        warnings.filterwarnings('ignore')
        exons = pd.read_csv("data/exon_variant_locs_unpadded.bed", sep="\t", header=None)
        exons.columns = ["chr", "start", "stop"]

        seq_ids = []

        for i in tqdm(range(len(self.variant_data))):
            aa_index = self.variant_data["POS"].iloc[i] - 1
            wt_aa = self.variant_data["REF"].iloc[i]
            mt_aa = self.variant_data["ALT"].iloc[i]
            seq_id = f"{self.variant_data['SYMBOL'].iloc[i]}_{aa_index}_{wt_aa}_{mt_aa}"
            gc.collect()
            if seq_id not in seq_ids:
                variant_seq, wildtype_seq = self._get_variant(self.variant_data["#CHROM"].iloc[i], aa_index, wt_aa,
                                                              mt_aa, exons)

                variant_aa = utils.translate_sequence(variant_seq)
                wildtype_aa = utils.translate_sequence(wildtype_seq)

                print(variant_aa)
                print(wildtype_aa)

                seq_ids.append(seq_id)

        return 0

    def __process_variants_parallel_genomic(self, num_processes, batch_size):
        warnings.filterwarnings('ignore')
        exons = pd.read_csv("data/exon_variant_locs_unpadded.bed", sep="\t", header=None)
        exons.columns = ["chr", "start", "stop"]

        input_tracker = []

        num_variants = len(self.variant_data)
        num_batches = (num_variants + batch_size - 1) // batch_size
        batch_size = (num_variants + num_processes - 1) // num_processes

        partial_process_batch = partial(self.process_batch)

        # Create a dictionary to store progress bars for each CPU
        progress_bars = {}
        for i in range(num_processes):
            progress_bars[i] = tqdm(total=batch_size, desc=f"CPU {i + 1}", position=i)

        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for cpu_id in range(num_processes):
                start_index = cpu_id * batch_size
                end_index = min((cpu_id + 1) * batch_size, num_variants)
                args = (start_index, end_index, exons, input_tracker, progress_bars[cpu_id])
                future = executor.submit(partial_process_batch, args)
                futures.append(future)

            # Use as_completed to iterate over completed futures
            for future in as_completed(futures):
                future_result = future.result()
                cpu_id = future_result[-1]
                progress_bar = progress_bars[cpu_id]
                progress_bar.update(1)

        # Close and remove progress bars
        for progress_bar in progress_bars.values():
            progress_bar.close()

        return 0

    def __process_batch(self, args):
        start_index, end_index, exons, input_tracker, progress_bar = args
        warnings.filterwarnings('ignore')
        for i in range(start_index, end_index):
            aa_index = self.variant_data["POS"].iloc[i] - 1
            wt_aa = self.variant_data["REF"].iloc[i]
            mt_aa = self.variant_data["ALT"].iloc[i]
            seq_id = f"{self.variant_data['SYMBOL'].iloc[i]}_{aa_index}_{wt_aa}_{mt_aa}"
            gc.collect()
            if seq_id not in input_tracker:
                variant_seq, wildtype_seq = self._get_variant(self.variant_data["#CHROM"].iloc[i], aa_index, wt_aa,
                                                              mt_aa, exons)

                variant_aa = utils.translate_sequence(variant_seq)
                print(i)
                wildtype_aa = utils.translate_sequence(wildtype_seq)
                progress_bar.update(1)
            else:
                input_tracker.append(seq_id)

    def _get_variant(self, chrom, pos, ref, alt, exons, debug=False):
        """
        Get the variant sequence.
        """
        exons_chr = exons.loc[exons['chr'] == chrom]
        exon = exons_chr.loc[(exons_chr['start'] - 100 <= pos) & (pos <= exons_chr['stop'] + 100)]
        if len(exon) > 1:
            exon['start_dist'] = abs(exon['start'] - pos)
            exon['stop_dist'] = abs(exon['stop'] - pos)
            exon = exon.sort_values(['start_dist', 'stop_dist'])
            exon = exon.iloc[0][['start', 'stop']]

        start = exon['start'].item() - 100
        end = exon['stop'].item() + 100
        ref_genome = next(r for r in SeqIO.parse(self.genome_path, "fasta") if r.id == chrom)
        sequence = str(ref_genome.seq)
        if str(sequence[pos]).lower() == str(ref).lower():
            if len(alt) > 1:
                alts = alt.split(',')
                for alt in alts:
                    variant_seq = sequence[start:pos] + alt.upper() + sequence[pos + 1:end]
                    wt_seq = sequence[start:end]
                    if debug:
                        print(sequence[start:pos] + "\033[31m" + alt.upper() + "\033[0m" + sequence[pos + 1:end])
                        print(ref.upper())
                        print('-----------')
                        print(alt.upper())
                        print(sequence[start:pos] + "\033[31m" + alt.upper() + "\033[0m" + sequence[pos + 1:end])
                        print('\n\n')
                    return variant_seq, wt_seq
            else:
                variant_seq = sequence[start:pos] + alt.upper() + sequence[pos + 1:end]
                wt_seq = sequence[start:end]
                if debug:
                    print(sequence[start:pos] + "\033[31m" + ref.upper() + "\033[0m" + sequence[pos + 1:end])
                    print(ref.upper())
                    print('-----------')
                    if len(alt) > 1:
                        print(alt.upper())
                        alts = alt.split(',')
                        for alt in alts:
                            print(sequence[start:pos] + "\033[31m" + alt.upper() + "\033[0m" + sequence[pos + 1:end])
                    else:
                        print(alt.upper())
                        print(sequence[start:pos] + "\033[31m" + alt.upper() + "\033[0m" + sequence[pos + 1:end])
                    print('\n\n')
                return variant_seq, wt_seq
        else:
            raise ValueError(f"The reference allele ({ref}) at position {pos} does not match the specified ref allele "
                             f"({sequence[pos]}).")
