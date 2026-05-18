"""Gene-level feature extraction from OpenTargets and population exome data."""
import os
import gc
import warnings
import argparse

import matplotlib.pyplot as plt
import torch
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

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Bio import SeqIO
from torch.utils.data import DataLoader
from shutil import copyfileobj

from typing import Optional

from varformer.data.splits import load_combined_labels, combine_features_and_labels, get_labels
from varformer.utils.aa_codes import aa_to_idx, three_letter_aa_to_idx
from varformer.utils.gene_id import map_gene_names
from varformer.data.parsers.alphamissense import merge_am_data
from varformer.training.lightning_module import VarformerLightningModule as MultiModalLightningTargetIdentifier
from varformer.data.features.base import BaseFeatures


# data preprocessing
class GeneCharacterisationPreprocessor(BaseFeatures):
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self, config, base: Optional[BaseFeatures] = None):
        print("Gene Characterisation Preprocessor is booting up...")
        if base is not None:
            # Adopt the BaseFeatures state without re-running its __init__
            self.__dict__.update(base.__dict__)
        else:
            super().__init__(config)

        features_dir = self.config['paths']['FEATURES_DIR']
        population = self.config['hyperparameters']['population']

        self.chem_features = None
        self.gnomad_features = None
        self.mouse_ko_features = None
        self.gene_essentiality_features = None
        self.ppi_features = None

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
        self.labels_dict = get_labels(self.ensg_ids, self.target)
        self.full_data = combine_features_and_labels(self.ensg_ids, self.features, self.target)
        self.full_data.set_index('targetId', inplace=True)
        # feature statistics can be checked here!

        self.ce_data = self.full_data
        self.num_features = len(self.full_data.columns) - 1

        self.data = self.full_data
        self.labels = self.labels_dict

        # Create population directory if it doesn't exist
        os.makedirs(f'{features_dir}/{population}', exist_ok=True)

        # Save the GC features (self.data contains the final feature matrix)
        gc_features_path = f'{features_dir}/{population}/gene_characterisation_features.pkl'
        with open(gc_features_path, 'wb') as f:
            pkl.dump(self.data, f)

    def load_ground_truth(self):
        """
        Load the ground truth data.
        """
        return self.datasets["FDA Approved Drug Targets"]

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
