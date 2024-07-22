import os
import gc
import warnings
import argparse
import torch
import utils
import requests
import time
import esm
import bz2

import pytorch_lightning as pl
import pickle as pkl
import gzip
import scipy.sparse as sparse
import pandas as pd
import numpy as np
import src.dataloader as dl

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from torch.utils.data import DataLoader
from shutil import copyfileobj

from autoencoders.ae import AutoencoderTrainer
from autoencoders.vae import VAETrainer
from src.autoencoders import ae_training, vae_training
from src.utils import featurise, load_combined_labels, combine_features_and_labels, aa_to_idx, three_letter_aa_to_idx


class GeneCharacterisationPreprocessor:
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self, config):
        print("Gene Characterisation Preprocessor is booting up...")
        self.config = config
        self.files_and_dirs = os.listdir("../data")
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

        # Load Genes & Health South-Asian Population exome data
        self.gh_data = self.load_gh_data()

        self.gh_data["UNIPROT"] = self.gh_data["SWISSPROT"].fillna(self.gh_data["TREMBL"])
        self.gh_data = self.gh_data.drop(["SWISSPROT", "TREMBL"], axis=1)
        self.gh_data['variant_id'] = self.gh_data['CHROM'].astype(str) + '_' + self.gh_data['POS'].astype(str) + '_' + \
                                     self.gh_data['REF'].astype(str) + '_' + self.gh_data['ALT'].astype(str)

        # TODO: Handle feature matrix generation and deal with different consequences of variants
        #   potentially ask Dan MacArthur for advice on how to handle this?
        #   Another question we should ask him is how to define some negatives for the test datasets (LoF/constraint
        #   would be argued against by his paper on evaluating drug targets through human loss-of-function)

        # Load raw G&H missense variant data
        if not os.path.exists('../data/features/raw_miva_feature_matrix.pkl'):
            miva_feature_matrix = self.gh_data[self.gh_data['Consequence'] == 'missense_variant']
            miva_feature_matrix = miva_feature_matrix[["Gene", "UNIPROT", "variant_id"]]
            miva_feature_matrix = miva_feature_matrix.rename(columns={"Gene": "ENSG"})
            miva_feature_matrix = miva_feature_matrix.drop_duplicates(subset="ENSG")
            miva_feature_matrix.to_pickle('../data/features/raw_miva_feature_matrix.pkl')

        feature_extractors = {
            'chem_features.pkl': self.chem_feature_extractor,
            'gnomad_features.pkl': self.gnomad_feature_extractor,
            'mouse_ko_features.pkl': self.mouse_knockout_feature_extractor,
            'gene_essentiality_features.pkl': self.gene_essentiality_feature_extractor,
            'ppi_features.pkl': self.ppi_feature_extractor,
        }

        for feature_file, feature_extractor in feature_extractors.items():
            if not os.path.isfile(f'../data/features/{feature_file}'):
                print(f"Extracting {feature_file}...")
                feature_extractor()
                with open(f'../data/features/{feature_file}', 'wb') as fp:
                    pkl.dump(getattr(self, feature_file.split('.')[0]), fp)
            else:
                print(f"Loading {feature_file}...")
                with open(f'../data/features/{feature_file}', 'rb') as fp:
                    setattr(self, feature_file.split('.')[0], pkl.load(fp))

        ensg_features = {
            "chemical_interaction_count": self.chem_features,
            "pli_lof_constraint": self.gnomad_features,
            "mouse_ko_effect": self.mouse_ko_features,
            "ppi_count": self.ppi_features,
            "common_essentials": self.gene_essentiality_features,
        }

        self.features, self.ensg_ids, self.uniprot_ids = featurise(ensg_features)
        self.num_features = len(self.features.columns)
        self.norm = True

        # Ground truth
        self.target = load_combined_labels()

        # Combine features and target
        self.full_data = combine_features_and_labels(self.ensg_ids, self.features, self.target)

        # Get test data and remove from train feature matrix
        self.pfam_ids = self.ensg_ids[self.ensg_ids.isin(self.drgbl_targets_pfam)]
        self.pfam_pos_data = self.full_data[self.full_data.index.isin(self.pfam_ids.index)]
        num_pfam_pos = len(self.pfam_pos_data)

        self.rcnt_ids = self.ensg_ids[self.ensg_ids.isin(self.rcnt_targets_fda)]
        self.rcnt_pos_data = self.full_data[self.full_data.index.isin(self.rcnt_ids.index)]
        self.rcnt_pos_data.loc[:, 'target'] = 1
        num_rcnt_pos = len(self.rcnt_pos_data)

        self.pharos_ids = self.ensg_ids[self.ensg_ids.isin(self.chem_targets_pharos)]
        self.pharos_pos_data = self.full_data[self.full_data.index.isin(self.pharos_ids.index)]
        self.pharos_pos_data.loc[:, 'target'] = 1
        num_pharos_pos = len(self.pharos_pos_data)

        self.holdout_ids = pd.concat([self.pfam_ids, self.rcnt_ids, self.pharos_ids])

        self.data_neg = self.full_data[~self.full_data.index.isin(self.holdout_ids.index)]

        common_essentials = self.full_data[self.full_data['common_essentials'] == 1]
        common_essentials = common_essentials[common_essentials['target'] == 0]
        common_essentials = common_essentials[common_essentials['pli_lof_constraint'] > 0.9]
        negative_test_balance = common_essentials.sample(n=len(self.holdout_ids), random_state=42)
        negative_test_ids = self.ensg_ids[self.ensg_ids.index.isin(negative_test_balance.index)]
        num_negs = len(negative_test_ids)

        self.pfam_negs = negative_test_ids.sample(n=num_pfam_pos, random_state=42)
        negative_test_ids = negative_test_ids.drop(self.pfam_negs.index)

        self.rcnt_negs = negative_test_ids.sample(n=num_rcnt_pos, random_state=42)
        negative_test_ids = negative_test_ids.drop(self.rcnt_negs.index)

        self.pharos_negs = negative_test_ids.sample(n=num_pharos_pos, random_state=42)

        self.pfam_ids_all = pd.concat([self.pfam_ids, self.pfam_negs])
        self.rcnt_ids_all = pd.concat([self.rcnt_ids, self.rcnt_negs])
        self.pharos_ids_all = pd.concat([self.pharos_ids, self.pharos_negs])

        self.all_test_ids = pd.concat([self.pfam_ids_all, self.rcnt_ids_all, self.pharos_ids_all])

        self.pfam_neg_data = self.full_data[self.full_data.index.isin(self.pfam_negs.index)]
        self.rcnt_neg_data = self.full_data[self.full_data.index.isin(self.rcnt_negs.index)]
        self.pharos_neg_data = self.full_data[self.full_data.index.isin(self.pharos_negs.index)]

        self.pfam_data = pd.concat([self.pfam_pos_data, self.pfam_neg_data]).sample(frac=1)
        self.rcnt_data = pd.concat([self.rcnt_pos_data, self.rcnt_neg_data]).sample(frac=1)
        self.pharos_data = pd.concat([self.pharos_pos_data, self.pharos_neg_data]).sample(frac=1)

        # # save combination of ids and data for plotting
        # self.pfam_ids_all = self.pfam_ids_all.reindex(self.pfam_data.index)
        # self.pfam_data = pd.concat([self.pfam_ids_all, self.pfam_data], axis=1)
        #
        # self.rcnt_ids_all = self.rcnt_ids_all.reindex(self.rcnt_data.index)
        # self.rcnt_data = pd.concat([self.rcnt_ids_all, self.rcnt_data], axis=1)
        #
        # self.pharos_ids_all = self.pharos_ids_all.reindex(self.pharos_data.index)
        # self.pharos_data = pd.concat([self.pharos_ids_all, self.pharos_data], axis=1)
        #
        # self.pfam_data.to_pickle('../data/test_data/pfam_data.pkl')
        # self.rcnt_data.to_pickle('../data/test_data/rcnt_data.pkl')
        # self.pharos_data.to_pickle('../data/test_data/pharos_data.pkl')

        total_holdout = num_pfam_pos + num_rcnt_pos + num_pharos_pos + num_negs

        num_positives = len(self.full_data[self.full_data['target'] == 1])
        print(f"Number of approved drug targets in the training data: {num_positives}")
        print(f"Number of approved or putative drug targets in the holdout data: {total_holdout}\n")
        print(f"Num positives and negatives:\n\t- Pfam: {num_pfam_pos}\n\t- Recently approved: {num_rcnt_pos}\n\t- "
              f"Pharos: {num_pharos_pos}")

        # Remove holdout data from training data
        self.data = self.full_data[~self.full_data.index.isin(self.all_test_ids.index)]

        # Explore the data
        # plot.umap(self.data)

    def _get_files(self):
        """
        Get the files from the data directory.
        """
        files = []
        exclude = ['.DS_Store', 'elgh', 'clinvar', 'VariPred', 'string_data_counts.pkl']

        for file in self.files_and_dirs:
            if "." in file and file not in exclude:
                files.append(f"../data/{file}")
            elif file not in exclude:
                file_path = f"../data/{file}"
                _file = self._dir_parser(file_path)
                files.append(_file)
        return files

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
        if "datasets.pkl" in os.listdir('../data/'):
            with open('../data/datasets.pkl', 'rb') as fp:
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
            with open('../data/datasets.pkl', 'wb') as fp:
                pkl.dump(datasets, fp)
            return datasets

    def load_gh_data(self):
        """
        Load the Genes & Health variant data.
        """
        # read in pickle file as dataframe
        # check the file type of the data path
        if 'pkl' in self.config['paths']['ALL_GH']:
            variant_data = pd.read_pickle(self.config['paths']['ALL_GH'])
        else:
            variant_data = pd.read_csv(self.config['paths']['ALL_GH'], sep="\t", low_memory=False)
        variant_data = variant_data.loc[:, ~variant_data.columns.str.contains('^Unnamed')]
        return variant_data

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
        mapped_names = utils.map_gene_names(gene_names, 'symb', 'ensg')
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
        mapped_names = utils.map_gene_names(gene_names, 'symb', 'ensg')
        gnom_data['gene'] = gnom_data['gene'].map(mapped_names)
        gnom_data = gnom_data[gnom_data['gene'] != 'N/A']
        gnom_data = gnom_data.set_index("gene")["pLI"].to_dict()

        self.gnomad_features = gnom_data

    def ppi_feature_extractor(self):
        """
        Featurise PPI data, i.e. count and normalize the PPIs for each PPI that is experimentally validated.
        """
        protein_info = []  # To store the parsed data
        with open("../data/9606.protein.links.full.v12.0.txt", "r") as file:
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

        mapped_names = utils.map_gene_names(protein_names, 'ensp', 'ensg')

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
        mapped_names = utils.map_gene_names(gene_names, 'symb', 'ensg')
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
        mapped_names = utils.map_gene_names(gene_names, 'symb', 'ensg')
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
        uniprot_data = self.gh_data[["SWISSPROT", "TREMBL", "varipred_id"]]
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
        self.protein_atlas_feature_names = None
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
            # self.gcp_acmg = gcp.acmg_data
        else:
            self.gcp = gcp
            self.gh_data = gcp.gh_data
            self.target = gcp.target
            self.gcp_data = gcp.data
            self.full_gcp_data = gcp.full_data
            self.gcp_pfam_pos = gcp.pfam_pos_data
            self.gcp_rcnt_pos = gcp.rcnt_pos_data
            self.gcp_pharos_pos = gcp.pharos_pos_data
            self.gcp_pfam_neg = gcp.pfam_neg_data
            self.gcp_rcnt_neg = gcp.rcnt_neg_data
            self.gcp_pharos_neg = gcp.pharos_neg_data
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
        self.tissue_expression_feature_extractor()

        print("Combining gene ontology features...")
        self.gene_ontology_features = None
        self.combine_go_features()

        self.data = self.gene_ontology_features

        # self.acmg_data = self.data[self.data['ENSG'].isin(self.acmg_ids)]
        # self.acmg_data = self.acmg_data.drop(columns=['ENSG'])
        # self.acmg_data = self.acmg_data.set_index(self.gcp_acmg.index)
        # self.acmg_data['target'] = self.gcp_acmg['target']

        self.pfam_data = self.data[self.data['ENSG'].isin(self.pfam_ids)]
        self.pfam_data = self.pfam_data.drop(columns=['ENSG'])
        self.pfam_pos_data = self.pfam_data.set_index(self.gcp_pfam_pos.index)
        self.pfam_pos_data['target'] = self.gcp_pfam_pos['target']

        self.rcnt_data = self.data[self.data['ENSG'].isin(self.rcnt_ids)]
        self.rcnt_data = self.rcnt_data.drop(columns=['ENSG'])
        self.rcnt_pos_data = self.rcnt_data.set_index(self.gcp_rcnt_pos.index)
        self.rcnt_pos_data['target'] = self.gcp_rcnt_pos['target']

        self.pharos_data = self.data[self.data['ENSG'].isin(self.pharos_ids)]
        self.pharos_data = self.pharos_data.drop(columns=['ENSG'])
        self.pharos_pos_data = self.pharos_data.set_index(self.gcp_pharos_pos.index)
        self.pharos_pos_data['target'] = self.gcp_pharos_pos['target']

        neg_data = self.data.set_index(self.full_gcp_data.index)
        pfam_neg_data = neg_data[neg_data.index.isin(self.pfam_neg_data.index)]
        pfam_neg_data = pfam_neg_data.drop(columns=['ENSG'])
        pfam_neg_data['target'] = 0

        rcnt_neg_data = neg_data[neg_data.index.isin(self.rcnt_neg_data.index)]
        rcnt_neg_data = rcnt_neg_data.drop(columns=['ENSG'])
        rcnt_neg_data['target'] = 0

        pharos_neg_data = neg_data[neg_data.index.isin(self.pharos_neg_data.index)]
        pharos_neg_data = pharos_neg_data.drop(columns=['ENSG'])
        pharos_neg_data['target'] = 0

        self.pfam_data = pd.concat([self.pfam_pos_data, pfam_neg_data]).sample(frac=1)
        self.rcnt_data = pd.concat([self.rcnt_pos_data, rcnt_neg_data]).sample(frac=1)
        self.pharos_data = pd.concat([self.pharos_pos_data, pharos_neg_data]).sample(frac=1)

        self.data = self.data.set_index(self.full_gcp_data.index)
        self.data = self.data[~self.data.index.isin(self.pfam_ids_all.index)]
        self.data = self.data[~self.data.index.isin(self.rcnt_ids_all.index)]
        self.data = self.data[~self.data.index.isin(self.pharos_ids_all.index)]
        self.data = self.data.drop(columns=['ENSG'])

        self.data['target'] = self.gcp_data['target']

        self.num_features = len(self.data.columns) - 1  # subtract 1 for the target column

    def protein_atlas_feature_extractor(self):
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

        feature_dict_bio_proc = {ensg: [0] * len(all_features_bio_proc) for ensg in protein_atlas_features['Ensembl']}
        feature_dict_mol_func = {ensg: [0] * len(all_features_mol_func) for ensg in protein_atlas_features['Ensembl']}
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

    def tissue_expression_feature_extractor(self):
        # check if the tissue expression data has already been processed
        if os.path.exists('../data/features/tissue_specificity_features.pkl'):
            with open('../data/features/tissue_specificity_features.pkl', 'rb') as f:
                self.tissue_specificity_features = pkl.load(f)
            return
        else:
            tissue_expression = pd.read_csv(self.config['paths']['TISSUE_EXPRESSION_PATH'], sep='\t')  # 1,197,500
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

            self.tissue_specificity_features = gene_tissue_dict
            with open('../data/features/tissue_specificity_features.pkl', 'wb') as f:
                pkl.dump(gene_tissue_dict, f)

    def combine_go_features(self):
        ensg_features = {
            "tissue_specificity": self.tissue_specificity_features,
            "biological_processes": self.protein_atlas_features['biological_processes'],
            "molecular_functions": self.protein_atlas_features['molecular_processes'],
            "subcellular_locations": self.protein_atlas_features['subcellular_locations']
        }

        with open('../data/features/raw_miva_feature_matrix.pkl', 'rb') as f:
            feature_matrix = pkl.load(f)

        for feature, values in ensg_features.items():
            feature_matrix[feature] = feature_matrix["ENSG"].map(values)

        feature_matrix = feature_matrix.drop(["UNIPROT"], axis=1)
        feature_matrix = feature_matrix.drop(["variant_id"], axis=1)

        sub_df = feature_matrix.copy()
        sub_df = sub_df.iloc[:, -3:]

        bio_proc_list = []
        mol_func_list = []
        sub_loc_list = []

        for index, row in sub_df.iterrows():
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

        feature_matrix['tissue_specificity'] = feature_matrix['tissue_specificity'].apply(
            lambda x: x if isinstance(x, list) else [])

        # Get the unique tissues
        unique_tissues = set(tissue for tissues_list in feature_matrix['tissue_specificity'] for tissue in tissues_list)

        # Create new columns for each unique tissue
        for tissue in unique_tissues:
            feature_matrix[f'tissue_specificity_{tissue.replace(" ", "_")}'] = feature_matrix[
                'tissue_specificity'].apply(
                lambda x: 1 if tissue in x else 0)

        feature_matrix = feature_matrix.drop('tissue_specificity', axis=1)
        self.gene_ontology_features = feature_matrix.fillna(0)


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
            # self.gcp_acmg = gcp.acmg_data
        else:
            self.gcp = gcp
            self.gh_data = gcp.gh_data
            self.target = gcp.target
            self.gcp_data = gcp.data
            self.full_gcp_data = gcp.full_data
            self.gcp_pfam_pos = gcp.pfam_pos_data
            self.gcp_rcnt_pos = gcp.rcnt_pos_data
            self.gcp_pharos_pos = gcp.pharos_pos_data
            self.gcp_pfam_neg = gcp.pfam_neg_data
            self.gcp_rcnt_neg = gcp.rcnt_neg_data
            self.gcp_pharos_neg = gcp.pharos_neg_data

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Preparing variant features...")
        self.variant_gh_data(config['hyperparameters']['pathogenicity_embedding'])

        print("Obtaining AlphaMissense pathogenicity embeddings...")
        self.var_pat_features = self.variant_pathogenicity_input()

        # print("Obtaining AlphaFold protein structure embeddings")
        # self.var_stc_features = self.variant_structure_input()

        # if self.device == 'cuda':
        #     # self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t48_15B_UR50D()
        #     # self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        #     self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #     self.esm_model = self.esm_model.half()
        # else:
        #     self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # self.esm_model = self.esm_model.to(self.device)
        # self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        # self.esm_model.eval()
        # self.var_seq_features = self.variant_sequence_input()

        if not os.path.exists('../data/cache/variant_pathogenicity_features.pkl'):
            self.var_pat_features, self.pat_ensg_ids, self.pat_uniprot_ids = featurise(self.var_pat_features,
                                                                                       'pathogenicity')
            self.num_pat_features = len(self.var_pat_features.columns)

            # self.var_stc_features, self.stc_ensg_ids, self.stc_uniprot_ids = featurise(self.var_stc_features,
            #                                                                           'structure')
            # self.num_stc_features = len(self.var_stc_features.columns)

            # self.var_seq_features, self.seq_ensg_ids, self.seq_uniprot_ids = featurise(self.var_seq_features,
            #                                                                            'sequence')

            if not os.path.exists('../data/cache/variant_pathogenicity_features.pkl'):
                var_pat_data = [self.var_pat_features, self.pat_ensg_ids, self.pat_uniprot_ids]
                with open('../data/cache/variant_pathogenicity_features.pkl', 'wb') as file:
                    pkl.dump(var_pat_data, file)
            # if not os.path.exists('../data/cache/variant_structure_features.pkl'):
            # self.var_stc_features.to_pickle('../data/cache/variant_structure_features.pkl')
            # if not os.path.exists('../data/cache/variant_sequence_features.pkl'):
            # with bz2.BZ2File('../data/cache/variant_sequence_features.pkl.bz2', 'wb') as f:
            # pkl.dump(self.var_seq_features, f)
        else:
            var_pat_data = pd.read_pickle('../data/cache/variant_pathogenicity_features.pkl')
            self.var_pat_features, self.pat_ensg_ids, self.pat_uniprot_ids = var_pat_data
            # self.var_stc_features = pd.read_pickle('../data/cache/variant_structure_features.pkl')
            # self.var_seq_features = pd.read_pickle('../data/cache/variant_sequence_features.pkl')

        # self.num_seq_features = len(self.var_seq_features.columns)

        self.norm = False

        # Ground truth
        self.target = load_combined_labels()

        common_essentials = self.data[self.data['common_essentials'] == 1]
        common_essentials = common_essentials[common_essentials['target'] == 0]
        common_essentials = common_essentials[common_essentials['pli_lof_constraint'] > 0.9]
        negative_test_balance = common_essentials.sample(n=len(self.holdout_ids), random_state=42)
        negative_test_ids = self.pat_ensg_ids[self.pat_ensg_ids.index.isin(negative_test_balance.index)]
        num_negs = len(negative_test_ids)

        # Combine features and target
        self.features = self.var_pat_features
        self.data = combine_features_and_labels(self.pat_ensg_ids, self.features, self.target)
        self.ensg_ids = self.pat_ensg_ids

        # Get test data and remove from train feature matrix
        self.pfam_ids = self.pat_ensg_ids[self.pat_ensg_ids.isin(self.drgbl_targets_pfam)]
        self.pfam_pos_data = self.data[self.data.index.isin(self.pfam_ids.index)]
        num_pfam_pos = len(self.pfam_pos_data)

        self.rcnt_ids = self.pat_ensg_ids[self.pat_ensg_ids.isin(self.rcnt_targets_fda)]
        self.rcnt_pos_data = self.data[self.data.index.isin(self.rcnt_ids.index)]
        self.rcnt_pos_data.loc[:, 'target'] = 1
        num_rcnt_pos = len(self.rcnt_pos_data)

        self.pharos_ids = self.pat_uniprot_ids[self.pat_ensg_ids.isin(self.chem_targets_pharos)]
        self.pharos_pos_data = self.data[self.data.index.isin(self.pharos_ids.index)]
        self.pharos_pos_data.loc[:, 'target'] = 1
        num_pharos_pos = len(self.pharos_pos_data)

        self.holdout_ids = pd.concat([self.pfam_ids, self.rcnt_ids, self.pharos_ids])

        self.data_neg = self.data[~self.data.index.isin(self.holdout_ids.index)]

        self.pfam_negs = negative_test_ids.sample(n=num_pfam_pos, random_state=42)
        negative_test_ids = negative_test_ids.drop(self.pfam_negs.index)

        self.rcnt_negs = negative_test_ids.sample(n=num_rcnt_pos, random_state=42)
        negative_test_ids = negative_test_ids.drop(self.rcnt_negs.index)

        self.pharos_negs = negative_test_ids.sample(n=num_pharos_pos, random_state=42)

        self.pfam_ids_all = pd.concat([self.pfam_ids, self.pfam_negs])
        self.rcnt_ids_all = pd.concat([self.rcnt_ids, self.rcnt_negs])
        self.pharos_ids_all = pd.concat([self.pharos_ids, self.pharos_negs])

        self.all_test_ids = pd.concat([self.pfam_ids_all, self.rcnt_ids_all, self.pharos_ids_all])

        self.pfam_neg_data = self.data[self.data.index.isin(self.pfam_negs.index)]
        self.rcnt_neg_data = self.data[self.data.index.isin(self.rcnt_negs.index)]
        self.pharos_neg_data = self.data[self.data.index.isin(self.pharos_negs.index)]

        self.pfam_data = pd.concat([self.pfam_pos_data, self.pfam_neg_data]).sample(frac=1)
        self.rcnt_data = pd.concat([self.rcnt_pos_data, self.rcnt_neg_data]).sample(frac=1)
        self.pharos_data = pd.concat([self.pharos_pos_data, self.pharos_neg_data]).sample(frac=1)

    def variant_gh_data(self, config):
        print("Preparing GH data for variant-level embeddings...")
        if not os.path.exists("../data/elgh/gh_miva_data.pkl"):
            self.gh_data['ALT'] = self.gh_data['ALT'].str.split(',')
            self.gh_data = self.gh_data.explode('ALT')
            self.gh_data = self.gh_data[(self.gh_data['ALT'].str.len() == 1) & (self.gh_data['REF'].str.len() == 1)]
            self.gh_data = self.gh_data[self.gh_data['Consequence'] == 'missense_variant']

            self.gh_data['Protein_position'] = self.gh_data['Protein_position'].astype(int)
            selected_data = self.gh_data[self.gh_data['Protein_position'] > config['io_dim']]
            io_dim = config['io_dim']
            self.gh_data.loc[:, 'Protein_pos_shard'] = self.gh_data['Protein_position'].apply(
                lambda x: (x - 1) % io_dim + 1)
            cols = self.gh_data.columns.tolist()
            pp_idx = cols.index('Protein_position')
            cols = cols[:pp_idx] + [cols[-1]] + cols[pp_idx:-1]
            self.gh_data = self.gh_data[cols]

            # self.gh_data = self.gh_data[self.gh_data['SYMBOL'].isin(genes_sharded)]
            self.gh_data.to_pickle('../data/elgh/gh_miva_data.pkl')
        else:
            self.gh_data = pd.read_pickle('../data/elgh/gh_miva_data.pkl')

    def variant_pathogenicity_input(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and prepare the input for embedding
        """
        # check if gh_am_data.pkl exists yet
        print("Combining AM and GH data...")
        if not os.path.exists("../data/features/var_pat_features.pkl.gz"):
            am = pd.read_csv("../data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')
            am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']

            am = am[['am_pathogenicity', 'variant_id']]
            print("Combining AlphaMissense data with GH data...")
            self.gh_data = self.gh_data.merge(am, on='variant_id', how='left')
            # save gh_data
            with open("../data/alphamissense/gh_am_data_full.pkl", 'wb') as f:
                pkl.dump(self.gh_data, f)

            sym_list = self.gh_data['SYMBOL'].tolist()
            prot_pos = self.gh_data['Protein_pos_shard'].tolist()
            aas = self.gh_data['Amino_acids'].tolist()
            hgvsp = self.gh_data['HGVSp'].tolist()
            prot_pos = [str(pos) for pos in prot_pos]
            hgvsp = [str(hgvsp.split('.')[1].split(':')[0]) for hgvsp in hgvsp]
            isoform_id_list = [f"{sym}_{prot_pos}_{aas.split('/')[0]}_{aas.split('/')[1]}_{hgvsp}" for
                               sym, prot_pos, aas, hgvsp in zip(sym_list, prot_pos, aas, hgvsp)]
            self.gh_data['isoform'] = isoform_id_list

            components = self.gh_data['isoform'].str.extract(
                r'(?P<gene_symbol>\w+)_(?P<prot_position>\d+)_(?P<ref_aa>\w+)_(?P<alt_aa>\w+)_(?P<isoform>\d+)')
            combined = components['gene_symbol'] + '_' + components['prot_position'].astype(str) + '_' + components[
                'ref_aa'] + '_' + components['alt_aa']
            self.gh_data['combined'] = combined
            self.gh_data['AA_ref'] = components['ref_aa']
            self.gh_data['AA_alt'] = components['alt_aa']
            self.gh_data = self.gh_data.loc[components['isoform'] == '1'].drop_duplicates(subset=['combined'])
            self.gh_data = self.gh_data.drop(columns=['combined'])

            var_pat_features = {}
            for index, row in tqdm(self.gh_data.iterrows(), total=self.gh_data.shape[0]):
                gene = row['Gene']
                ref_aa = row['AA_ref']
                alt_aa = row['AA_alt']
                ref_idx = aa_to_idx(ref_aa)
                alt_idx = aa_to_idx(alt_aa)
                pos = row['Protein_pos_shard']
                value = row['am_pathogenicity'] * row['AF']

                io_dim = self.config['hyperparameters']['pathogenicity_embedding']['io_dim']
                if gene not in var_pat_features.keys():
                    matrix_shape = (21 * 21, io_dim)
                    var_pat_matrix = sparse.lil_matrix(matrix_shape, dtype=np.float32)
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

            # Convert all sparse matrices to CSR format
            for gene, var_pat_matrix in var_pat_features.items():
                var_pat_features[gene] = var_pat_matrix.tocsr()

            with gzip.open('../data/features/var_pat_features.pkl.gz', 'wb') as f:
                pkl.dump(var_pat_features, f)
            return var_pat_features
        else:
            with gzip.open('../data/features/var_pat_features.pkl.gz', 'rb') as f:
                return pkl.load(f)

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

            var_seq_data = self.gh_data.drop_duplicates(subset=['Feature'])

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
                model_dir + '/' + model_file, input_dim=hparams['io_dim'],
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
                model_dir + '/' + model_file, input_dim=hparams['io_dim'],
                latent_dim=hparams['latent_dim']
            )

        model.eval()

        with torch.no_grad():
            variant_pathogenicity = DataLoader(
                dl.VariantPathogenicityData(data_dict=variant_am_features, reduct_dim=hparams['io_dim'],
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

        # TODO: continue the implementation

        return 0

    def alphafold_extractor(self):
        """
        Extract AlphaFold features from the AlphaFold API. Specifically, we get the average pLDDT score for the proteins
        in our dataset
        """
        uniprot_ids = self.gh_data["UNIPROT"].unique().tolist()
        uniprot_ids = [uni for uni in uniprot_ids if str(uni) != 'nan']
        uniprot_ids = [uni.split('.')[0] for uni in uniprot_ids]
        if not os.path.exists('../data/cache/uniprot_ids.pkl'):
            uni_to_gene = {}
            for index, row in self.gh_data.iterrows():
                gene = row['Gene']
                uni = row['UNIPROT']
                if type(uni) == float:
                    continue
                uni = uni.split('.')[0]
                uni_to_gene[uni] = gene

            # write the uniprot ids to a pkl file
            with open('../data/cache/uniprot_ids.pkl', 'wb') as fp:
                pkl.dump(uni_to_gene, fp)
        else:
            with open('../data/cache/uniprot_ids.pkl', 'rb') as fp:
                uni_to_gene = pkl.load(fp)

        if not os.path.exists('../data/alphafold/alphafold_cifs'):
            self.download_af_cifs()
        extracted_values = {}
        if os.path.exists('../data/features/alphafold_features.pkl'):
            with open('../data/features/alphafold_features.pkl', 'rb') as fp:
                features = pkl.load(fp)
                return features
        else:
            if os.path.exists('../data/alphafold/alphafold_features_temp.pkl'):
                with open('../data/alphafold/alphafold_features_temp.pkl', 'rb') as fp:
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
                with open('../data/alphafold/temp_extracted_values.pkl', 'wb') as fp:
                    pkl.dump(extracted_values, fp)

                for uni, info in tqdm(extracted_values.items(), total=len(extracted_values)):
                    seq = info['sequence']
                    if not seq:
                        continue
                    cif_file_path = f"../data/alphafold/alphafold_cifs/AF-{uni}-F1-model_v4.cif"
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
            with open('../data/features/alphafold_features.pkl', 'wb') as fp:
                pkl.dump(extracted_values, fp)
            return extracted_values

    def download_af_cifs(self):
        uniprot_ids = self.gh_data["UNIPROT"].unique().tolist()
        uniprot_ids = [uni for uni in uniprot_ids if str(uni) != 'nan']
        uniprot_ids = [uni.split('.')[0] for uni in uniprot_ids]

        url = "https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.cif"

        folder = "../data/alphafold/alphafold_cifs"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for uni_id in tqdm(uniprot_ids):
            # check if uni_id occurs in swissprot_cif_v4 folder, if so copy it to alphafold_cifs
            if os.path.exists(f"../data/alphafold/alphafold_cifs/AF-{uni_id}-F1-model_v4.cif"):
                continue
            file_url = url.format(id=uni_id)
            file_name = os.path.basename(file_url)

            response = requests.get(file_url)
            with open(os.path.join(folder, file_name), "wb") as f:
                f.write(response.content)

    # DEPRECATED

    def __pathogenicity_feature_extractor(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and average to gene-level using population
        statistics.
        """
        self.gh_data["uniprot_id"] = self.gh_data["SWISSPROT"].fillna(self.gh_data["TREMBL"])
        self.gh_data["uniprot_id"] = self.gh_data["SWISSPROT"].fillna(self.gh_data["TREMBL"])

        am = pd.read_csv("../data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')

        am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']
        self.gh_data['ALT'] = self.gh_data['ALT'].str.split(',')
        self.gh_data = self.gh_data.explode('ALT')
        self.gh_data = self.gh_data[(self.gh_data['ALT'].str.len() == 1) & (self.gh_data['REF'].str.len() == 1)]
        self.gh_data = self.gh_data[self.gh_data['Consequence'] == 'missense_variant']
        pos_list = self.gh_data['POS'].tolist()
        chrom_list = self.gh_data['#CHROM'].tolist()
        ref_list = self.gh_data['REF'].tolist()
        alt_list = self.gh_data['ALT'].tolist()

        pos_list = [str(pos) for pos in pos_list]
        variant_id_list = [chrom + '_' + pos + '_' + ref + '_' + alt for chrom, pos, ref, alt in
                           zip(chrom_list, pos_list, ref_list, alt_list)]
        self.gh_data['variant_id'] = variant_id_list
        am = am[['am_pathogenicity', 'variant_id']]

        self.gh_data = self.gh_data.merge(am, on='variant_id', how='left')

        if not os.path.exists("../data/alphamissense/gh_am_data.pkl"):
            self.gh_data.to_pickle('../data/alphamissense/gh_am_data.pkl')

        variant_am_features = {}
        for index, row in tqdm(self.gh_data.iterrows()):
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
            vp_pretrained_data = pd.read_csv("../data/VariPred/varipred_output_data_pretrained.csv", sep="\t")
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
        elif len(os.listdir('../data/VariPred/input')) == self.config['varipred']['num_batches']:
            print("Variants already processed.")
        else:
            print('Not all variants are preprocessed yet. Put the preprocess flag to True in the MissenseVariantLoader '
                  'and run again.')
        if train:
            # NOTE: In order to prepare the data for training, first all the datasets generated in evaluation need to be
            # loaded.

            # raw_data here is dummy file, normally should be done on cluster for all batch files

            raw_data = pd.read_csv("../data/elgh/train_batch_mivas/variant_data_396.csv")
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

        if os.path.exists("../data/elgh/varipred_elgh_data.csv"):
            self.variant_data = pd.read_csv("../data/elgh/varipred_elgh_data.csv", sep="\t")
        else:
            self.variant_data = utils.combine_varipred_elgh(varipred_output, self.variant_data)

        if evaluation:
            clinvar_data = pd.read_csv("../data/clinvar/variant_summary.txt", sep="\t")
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
        am = pd.read_csv("../data/alphamissense/AlphaMissense_hg38.tsv", sep='\t')
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
        sequence_table.to_csv(f"../data/VariPred/input/variants_{variants_id}.csv", index=False)

    def train_test_val_loader(self, data, downsampling=True):
        train_files = os.listdir("../data/VariPred/train/")
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
            sequence_table.to_csv(f"../data/VariPred/train/variants_{variants_id}.csv", index=False)
        else:
            if not os.path.exists("../data/VariPred/all_train.csv"):
                utils.combine_train_files()
            elif not os.path.exists("../data/VariPred/train.csv"):
                raw_train = pd.read_csv("../data/VariPred/all_train.csv")
                df = raw_train.copy()
                df = df[df.target_id != 'target_id']
                df = df.rename(columns={'target_id': 'seq_id'})
                train, test = train_test_split(df, test_size=0.1, random_state=555, stratify=df['label'])
                train.to_csv("../data/VariPred/train.csv", index=False)
                # val.to_csv("../data/VariPred/val.csv", index=False)
                test.to_csv("../data/VariPred/test.csv", index=False)
                print(f"Train and test data loaded with size: {len(train)} and {len(test)}")
                return train, test
            elif downsampling and not os.path.exists("../data/VariPred/train_downsample.csv"):
                train = pd.read_csv("../data/VariPred/train.csv")
                test = pd.read_csv("../data/VariPred/test.csv")
                # val = pd.read_csv("../data/VariPred/test.csv")
                train = utils.downsampler(train)
                test = utils.downsampler(test)
                # val = utils.downsampler(val)
                train.to_csv("../data/VariPred/train_downsample.csv", index=False)
                test.to_csv("../data/VariPred/test_downsample.csv", index=False)
                # val.to_csv("../data/VariPred/val_downsample.csv", index=False)
                print(f"Train and test data downsampled with sizes: {len(train)} {len(test)}")
                return train, test
            elif downsampling:
                train = pd.read_csv("../data/VariPred/train_downsample.csv")
                # val = pd.read_csv("../data/VariPred/val_downsample.csv")
                test = pd.read_csv("../data/VariPred/test_downsample.csv")
                print(f"Downsampled train and test data loaded with sizes: {len(train)}, {len(test)}")
                return train, test
            else:
                train = pd.read_csv("../data/VariPred/train.csv")
                val = pd.read_csv("../data/VariPred/val.csv")
                test = pd.read_csv("../data/VariPred/test.csv")
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
