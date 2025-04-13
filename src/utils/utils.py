import os
import subprocess
import re
import csv

import requests
import random
import torch
import warnings

import numpy as np
import pandas as pd
import pickle as pkl
import biorosetta as br

from sklearn.metrics import matthews_corrcoef, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from scipy.sparse import csr_matrix, issparse
from Bio import Seq
from torch import nn
from typing import Tuple, Optional
from tqdm import tqdm


class random_seed_context:
    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = random.getstate()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.state)


def correct_aa_position(target_id):
    if target_id == 'target_id':
        return 'target_id'
    else:
        parts = target_id.split('_')
        aa_position = int(parts[1])
        adjusted_aa_position = aa_position + 1
        parts[1] = str(adjusted_aa_position)
        return '_'.join(parts)


def map_gene_names(list_of_genes: list, source_type: str, target_type: str) -> dict:
    idmap = br.IDMapper('all')
    list_of_targets = idmap.convert(list_of_genes, source_type, target_type)
    if 'N/A' in list_of_targets:
        warnings.warn("Some genes were not found in the mapping. Check the input list of genes.")
        missing = [list_of_genes[i] for i, x in enumerate(list_of_targets) if x == 'N/A']
        warnings.warn(f"Number of missing genes: {len(missing)}")
    return dict(zip(list_of_genes, list_of_targets))


def get_protein_length(ensp, ensg):
    ensp_api_url = f"https://rest.ensembl.org/sequence/id/{ensp}"
    ensg_api_url = f"https://rest.ensembl.org/sequence/id/{ensg}?type=protein;multiple_sequences=1"

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(ensp_api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            protein = data.get('seq', None)
            protein_length = len(protein)

            if protein_length is not None:
                return protein_length
            else:
                raise KeyError(f"Protein length for {ensp} not found in the response.")
        else:
            raise KeyError(f"Failed to retrieve protein information. Status code: {response.status_code}")
    except KeyError:
        response = requests.get(ensg_api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            protein = data[0].get('seq', None)
            protein_length = len(protein)

            if protein_length is not None:
                return protein_length
            else:
                raise KeyError(f"Protein length for {ensg} not found in the response.")
        else:
            raise KeyError(f"Failed to retrieve protein information. Status code: {response.status_code}")


def count_zeros(df: pd.DataFrame) -> None:
    """
    For each column in a given dataframe, count how many zeros occur
    :return:
    """
    print("Feature sparsity:")

    for col in df.columns:
        num_zeros = len(df[df[col] == 0])
        print(f"{col}: {round(num_zeros / len(df) * 100, 2)}%")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def df_col_to_dense(x):
    # inspect all the types in x and collect in a list
    print(x)
    print("\n\n\n")
    print("type of x: \n")
    print(type(x))
    print("\n\n\n")
    types_unwrapped = [type(i) for i in x]
    unique_types = set(types_unwrapped)
    print("Types unwrapped: \n")
    print(unique_types)
    if issparse(x):
        _x = np.array(x.todense()).flatten()
        return _x
    elif isinstance(x, np.matrix):
        _x = np.array(x).flatten()
        return _x
    return x


def _convert_to_dense(indices, shape):
    dense_matrix = np.zeros(shape)

    for index in indices:
        dense_matrix[index] = 1

    return dense_matrix


def load_fda_labels() -> pd.DataFrame:
    return pd.read_excel("../data/FDA_approved_drug_targets_2023_Q3.xlsx")


def load_combined_labels(ot_targets, config) -> pd.DataFrame:
    """
    Load the HPA/manual FDA and citeline labels pkl file and combine them with Platform Known Drugs from ChEMBL
    """
    with open(config['paths']['CITELINE_LABELS'], "rb") as f:
        labels = pkl.load(f)
    labels = labels.drop(columns=["Gene"])
    labels_ensembl = labels["Ensembl"].tolist()
    new_labels = pd.DataFrame(
        [{"Ensembl": target, "Status": "Launched"} for target in ot_targets[ot_targets['target'] == 1]['targetId'] if
         target not in labels_ensembl]
    )
    labels = pd.concat([labels, new_labels], ignore_index=True)
    return labels


def get_labels(gene_names: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    target_genes = list(target["Ensembl"])
    labels = {}
    for gene in gene_names:
        if gene in target_genes:
            labels[gene] = 1
        else:
            labels[gene] = 0
    # features["target"] = 0
    # features.loc[gene_names.isin(target_genes), "target"] = 1
    return labels


def combine_features_and_labels(gene_names: pd.DataFrame, features: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    target_genes = list(target["Ensembl"])
    features["target"] = 0
    features.loc[gene_names.isin(target_genes), "target"] = 1
    return features


def padding(batch):
    """
    Pads the input data to the same length.
    :param batch:
    :return:
    """
    X = []
    for x, reduct_dim in batch:
        current_dimension = x.size(-1)
        if current_dimension == reduct_dim:
            X.append(x)
        elif current_dimension < reduct_dim:
            padding_left = (reduct_dim - current_dimension) // 2
            padding_right = reduct_dim - current_dimension - padding_left
            padded_x = F.pad(x, (padding_left, padding_right), value=0)
            X.append(padded_x)
        else:
            pooled_x = pooling(x, reduct_dim)
            pooled_x = pooled_x.squeeze(0)
            X.append(pooled_x)
    X = torch.stack(X)
    return X


def pooling(x, reduct_dim):
    x = x.unsqueeze(0)
    pool = nn.AdaptiveAvgPool1d(reduct_dim)
    pooled_x = pool(x)
    return pooled_x


def aa_to_idx(aa: str, dna_encoded=False) -> int:
    """
    Convert single-letter amino acid codes to index. NOTE: This contains the non-standard amino acid U!
    """
    if not dna_encoded:
        aa_to_idx_map = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
            'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'U': 17, 'V': 18, 'W': 19, 'Y': 20
        }
    else:
        aa_to_idx_map = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
            'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
        }
    return aa_to_idx_map[aa]


def three_letter_aa_to_idx(aa: str) -> int:
    """
    Convert three-letter amino acid code to index.
    """
    three_letter_aa_to_idx_map = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
    }
    return three_letter_aa_to_idx_map[aa]


def aa1_to_aa3(single_code):
    amino_acids = {
        'A': 'ALA',
        'R': 'ARG',
        'N': 'ASN',
        'D': 'ASP',
        'C': 'CYS',
        'E': 'GLU',
        'Q': 'GLN',
        'G': 'GLY',
        'H': 'HIS',
        'I': 'ILE',
        'L': 'LEU',
        'K': 'LYS',
        'M': 'MET',
        'F': 'PHE',
        'P': 'PRO',
        'S': 'SER',
        'T': 'THR',
        'W': 'TRP',
        'Y': 'TYR',
        'V': 'VAL'
    }

    # Convert input to uppercase
    single_code = single_code.upper()
    return amino_acids.get(single_code, 'Unknown')
