import pandas as pd
import numpy as np
import requests
import pickle as pkl
import os
import gc
import warnings
import argparse
import shutil
import torch

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from sklearn.model_selection import train_test_split
from Bio import SeqIO
from torch.nn.functional import one_hot

import utils


class GeneCharacterisationPreprocessor:
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self, config=None):

        self.config = config
        self.files_and_dirs = os.listdir("../data")
        self.data_name_mapping = {
            "CTD_chem_gene_ixns.csv": "CTD Chemical-Gene Interactions",
            "gnomad.exomes.v2.1.1.lof_metrics.by_gene.csv": "gnomAD Exomes Loss-of-Function Metrics",
            "9606.protein.links.full.v12.0.txt": "STRING Protein-Protein Interactions",
            "part-00000-31eba8be-aff8-492e-9edb-4b5e8c821237-c000.snappy.parquet": "Mouse Knockout Phenotypes",

            "FDA_approved_drug_targets_2023_Q3.xlsx": "FDA Approved Drug Targets"
        }
        self.files = self._get_files()
        self.datasets = self.load_data()

        self.chem_features = None
        self.gnomad_features = None
        self.mouse_ko_features = None
        self.gene_essentiality_features = None
        self.ppi_features = None
        self.pathogenicity_features = None
        self.alphafold_features = None
        self.protein_atlas_features = None
        self.protein_atlas_feature_names = None

        # Population genomics data
        self.gh_data = self.load_gh_data()

        if os.path.exists('../data/features/protein_atlas_feature_names.pkl'):
            with open('../data/features/protein_atlas_feature_names.pkl', 'rb') as fp:
                self.protein_atlas_feature_names = pkl.load(fp)

        feature_extractors = {
            'chem_features.pkl': self.chem_feature_extractor,
            'gnomad_features.pkl': self.gnomad_feature_extractor,
            'mouse_ko_features.pkl': self.mouse_knockout_feature_extractor,
            'gene_essentiality_features.pkl': self.gene_essentiality_feature_extractor,
            'ppi_features.pkl': self.ppi_feature_extractor,
            'pathogenicity_features.pkl': self.pathogenicity_feature_extractor,
            'alphafold_features.pkl': self.alphafold_feature_extractor,
            'protein_atlas_features.pkl': self.protein_atlas_feature_extractor
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

        self.features = self.combine_features()

        # Ground truth
        self.target = self.load_ground_truth()

        # Combine features and target
        self.data = self.make_data()
        print("Data loaded!")

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
        exclude = ['.DS_Store', 'elgh']
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
        variant_data = pd.read_csv(self.config['paths']['MIVA_PATH'], sep="\t")
        variant_data = variant_data.loc[:, ~variant_data.columns.str.contains('^Unnamed')]
        return variant_data

    def load_ground_truth(self):
        """
        Load the ground truth data.
        """
        return self.datasets["FDA Approved Drug Targets"]

    def make_data(self):
        """
        Combine the features and the target.
        """
        target_genes = list(self.target["Ensembl"])
        feature_genes = list(self.features["ENSG"])
        target_genes_in_features = [gene for gene in target_genes if gene in feature_genes]
        print(f"Found {len(target_genes_in_features)} FDA approved GH genes out of a total of {len(target_genes)} FDA "
              f"approved genes.")
        self.features["target"] = 0
        self.features.loc[self.features["ENSG"].isin(target_genes_in_features), "target"] = 1
        return self.features

    def download_af_cifs(self):
        uniprot_data = self.gh_data[["SWISSPROT", "TREMBL", "varipred_id"]]
        uniprot_data["uniprot_id"] = uniprot_data["SWISSPROT"].fillna(uniprot_data["TREMBL"])
        uniprot_data = uniprot_data.drop(["SWISSPROT", "TREMBL"], axis=1).rename(columns={"uniprot_id": "UNIPROT"})
        uniprot_ids = uniprot_data["UNIPROT"].unique().tolist()

        url = "https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.cif"

        folder = "../data/alphafold/alphafold_cifs"
        if not os.path.exists(folder):
            os.makedirs(folder)

        for uni_id in tqdm(uniprot_ids):
            # check if uni_id occurs in swissprot_cif_v4 folder, if so copy it to alphafold_cifs
            if os.path.exists(f"../data/alphafold/swissprot_cif_v4/AF-{uni_id}-F1-model_v4.cif"):
                shutil.copy(f"../data/alphafold/swissprot_cif_v4/AF-{uni_id}-F1-model_v4.cif",
                            f"../data/alphafold/alphafold_cifs/AF-{uni_id}-F1-model_v4.cif")
                continue
            file_url = url.format(id=uni_id)
            file_name = os.path.basename(file_url)

            response = requests.get(file_url)
            with open(os.path.join(folder, file_name), "wb") as f:
                f.write(response.content)

    def alphafold_feature_extractor(self):
        """
        Extract AlphaFold features from the AlphaFold API. Specifically, we get the average pLDDT score for the proteins
        in our dataset
        """
        uniprot_data = self.gh_data[["SWISSPROT", "TREMBL", "varipred_id"]]
        uniprot_data["uniprot_id"] = uniprot_data["SWISSPROT"].fillna(uniprot_data["TREMBL"])
        uniprot_data = uniprot_data.drop(["SWISSPROT", "TREMBL"], axis=1).rename(columns={"uniprot_id": "UNIPROT"})
        uniprot_ids = uniprot_data["UNIPROT"].unique().tolist()
        if not os.path.exists('../data/alphafold/alphafold_cifs'):
            self.download_af_cifs()
        extracted_values = {}
        if os.path.exists('../data/alphafold/af_plddt_features.pkl'):
            with open('../data/alphafold/af_plddt_features.pkl', 'rb') as fp:
                features = pkl.load(fp)
            return features
        else:
            if os.path.exists('../data/alphafold/af_plddt_features_non_normalized.pkl'):
                with open('../data/alphafold/af_plddt_features_non_normalized.pkl', 'rb') as fp:
                    extracted_values = pkl.load(fp)
            else:
                for qualifier in tqdm(uniprot_ids):
                    extracted_values[qualifier] = {}
                    cif_file_path = f"{self.config['paths']['AF_PATH']}AF-{qualifier}-F1-model_v4.cif"
                    target_format_mean = "_ma_qa_metric_global.metric_value"
                    target_format_max = "_ma_qa_metric_local.ordinal_id"
                    extract = False
                    values_list = []
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
                                if len(parts) >= 5:
                                    plddt = float(parts[4])
                                    values_list.append(plddt)
                    if len(values_list) != 0:
                        protein_len = len(values_list)
                        max_value = max(values_list)
                        extracted_values[qualifier]['max'] = max_value
                        # we extract the len for later experiments
                        extracted_values[qualifier]['protein_len'] = protein_len
                    else:
                        print(f"\nError: Unable to fetch data for {qualifier}. Inserting 0.0.")
                        extracted_values[qualifier]['mean'] = 0.0
                        extracted_values[qualifier]['max'] = 0.0
                        extracted_values[qualifier]['protein_len'] = np.nan
            # NOTE: normalize AFTER train test split
            # scaler = MinMaxScaler()
            # extracted_values = pd.DataFrame.from_dict(extracted_values, orient='index')
            # extracted_values[['mean', 'max']] = scaler.fit_transform(extracted_values[['mean', 'max']])
            # features = extracted_values.to_dict()
            self.alphafold_features = extracted_values

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
        loss-of-function intolerant. There are more potential features we can extract from the gnomAD data.
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

    def pathogenicity_feature_extractor(self):
        """
        Extract variant-level AlphaMissense pathogenicity score and average to gene-level using population statistics.
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

    def gene_essentiality_feature_extractor(self):
        raw_data = pd.read_csv(self.config['paths']['COMMON_ESSENTIALS_PATH'])
        raw_data = raw_data.rename(columns={raw_data.columns[0]: 'gene_name'})
        raw_data['gene_name'] = raw_data['gene_name'].str.split(' ').str[0]
        gene_names = list(raw_data['gene_name'])
        mapped_names = utils.map_gene_names(gene_names, 'symb', 'ensg')
        gene_list = list(mapped_names.values())
        gene_essentiality = {gene: 1 for gene in gene_list}

        self.gene_essentiality_features = gene_essentiality

    def combine_features(self):
        """
        Combine all the features into a single feature matrix. Use the ELGH variant data as a framework such
        that we can easily map between ensg, uniprot, and symbols as contained in the ELGH variant data.
        """
        self.gh_data["UNIPROT"] = self.gh_data["SWISSPROT"].fillna(self.gh_data["TREMBL"])
        self.gh_data = self.gh_data.drop(["SWISSPROT", "TREMBL"], axis=1)

        feature_matrix = self.gh_data[["Gene", "UNIPROT"]]
        feature_matrix = feature_matrix.rename(columns={"Gene": "ENSG"})
        feature_matrix = feature_matrix.drop_duplicates(subset=['ENSG'])

        ensg_features = {
            "chemical_interaction_count": self.chem_features,
            "pli_lof_constraint": self.gnomad_features,
            "mouse_ko_effect": self.mouse_ko_features,
            "ppi_count": self.ppi_features,
            "common_essentials": self.gene_essentiality_features,
            "am_missense_pathogenicity_score": self.pathogenicity_features,
            "biological_processes": self.protein_atlas_features['biological_processes'],
            "molecular_functions": self.protein_atlas_features['molecular_processes'],
            "subcellular_locations": self.protein_atlas_features['subcellular_locations']
        }
        uniprot_features = {
            "af_protein_structure_score": self.alphafold_features["mean"]
        }

        for feature, values in ensg_features.items():
            feature_matrix[feature] = feature_matrix["ENSG"].map(values)
        for feature, values in uniprot_features.items():
            feature_matrix[feature] = feature_matrix["UNIPROT"].map(values)

        feature_matrix = feature_matrix.drop(["UNIPROT"], axis=1)

        # cols = list(feature_matrix.columns)
        # cols_to_move = ['biological_processes', 'molecular_functions', 'subcellular_locations']
        #
        # cols = [col for col in cols if col not in cols_to_move]
        #
        # cols += cols_to_move
        #
        # feature_matrix = feature_matrix[cols]

        sub_df = feature_matrix.copy()
        sub_df = sub_df.iloc[:, -4:-1]

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

        # utils.count_zeros(feature_matrix)

        # plot.correlation_heatmap(feature_matrix)

        return feature_matrix

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

        with open('../data/features/protein_atlas_feature_names.pkl', 'wb') as fp:
            pkl.dump(self.protein_atlas_feature_names, fp)

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

    def _protein_atlas_feature_extractor(self):
        protein_atlas_features = pd.read_csv(self.config['paths']['PROTEIN_ATLAS_FEATURES'], sep='\t')
        protein_atlas_features = protein_atlas_features[['Ensembl', 'Biological process', 'Molecular function',
                                                         'Subcellular location']]

        bio_proc = []
        mol_func = []
        sub_loc = []

        bio_proc_lens = []
        mol_func_lens = []
        sub_loc_lens = []
        for index, row in protein_atlas_features.iterrows():
            if pd.isna(row['Biological process']):
                continue
            bio_proc += [x.strip() for x in row['Biological process'].split(',')]
            bio_proc_len = len(row['Biological process'].split(','))
            bio_proc_lens.append(bio_proc_len)

        for index, row in protein_atlas_features.iterrows():
            if pd.isna(row['Molecular function']):
                continue
            mol_func += [x.strip() for x in row['Molecular function'].split(',')]
            mol_func_len = len(row['Molecular function'].split(','))
            mol_func_lens.append(mol_func_len)

        for index, row in protein_atlas_features.iterrows():
            if pd.isna(row['Subcellular location']):
                continue
            sub_loc += [x.strip() for x in row['Subcellular location'].split(',')]
            sub_loc_len = len(row['Subcellular location'].split(','))
            sub_loc_lens.append(sub_loc_len)

        max_bio_proc_len = max(bio_proc_lens)
        max_mol_func_len = max(mol_func_lens)
        max_sub_loc_len = max(sub_loc_lens)

        # make the entries unique
        bio_proc = list(set(bio_proc))
        mol_func = list(set(mol_func))
        sub_loc = list(set(sub_loc))

        one_hot_bio_proc = utils.one_hot_encode(bio_proc)
        one_hot_mol_func = utils.one_hot_encode(mol_func)
        one_hot_sub_loc = utils.one_hot_encode(sub_loc)

        bio_proc_dict = {}
        mol_func_dict = {}
        sub_loc_dict = {}

        for i, feature in enumerate(bio_proc):
            bio_proc_dict[feature] = one_hot_bio_proc[i]
        for i, feature in enumerate(mol_func):
            mol_func_dict[feature] = one_hot_mol_func[i]
        for i, feature in enumerate(sub_loc):
            sub_loc_dict[feature] = one_hot_sub_loc[i]

        protein_atlas_features['Biological process'] = protein_atlas_features['Biological process'].str.split(
            ',').apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else x)
        protein_atlas_features['Molecular function'] = protein_atlas_features['Molecular function'].str.split(
            ',').apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else x)
        protein_atlas_features['Subcellular location'] = protein_atlas_features['Subcellular location'].str.split(
            ',').apply(lambda x: [i.strip() for i in x] if isinstance(x, list) else x)

        bio_proc_features = {}
        mol_func_features = {}
        sub_loc_features = {}

        for index, row in protein_atlas_features.iterrows():
            ensg = row['Ensembl']
            bio_proc = row['Biological process']
            mol_func = row['Molecular function']
            sub_loc = row['Subcellular location']

            bio_proc_matrix = np.zeros((max_bio_proc_len, len(one_hot_bio_proc[0])))
            mol_func_matrix = np.zeros((max_mol_func_len, len(one_hot_mol_func[0])))
            sub_loc_matrix = np.zeros((max_sub_loc_len, len(one_hot_sub_loc[0])))

            if isinstance(bio_proc, list):
                for i, feature in enumerate(bio_proc):
                    bio_proc_matrix[i] = bio_proc_dict[feature]
            if isinstance(mol_func, list):
                for i, feature in enumerate(mol_func):
                    mol_func_matrix[i] = mol_func_dict[feature]
            if isinstance(sub_loc, list):
                for i, feature in enumerate(sub_loc):
                    sub_loc_matrix[i] = sub_loc_dict[feature]

            bio_proc_features[ensg] = bio_proc_matrix
            mol_func_features[ensg] = mol_func_matrix
            sub_loc_features[ensg] = sub_loc_matrix

        self.protein_atlas_features = {
            'biological_processes': bio_proc_features,
            'molecular_processes': mol_func_features,
            'subcellular_locations': sub_loc_features
        }

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
        parser.add_argument('--varipred_input', type=str)
        self.args = parser.parse_args()
        if self.args.data is not None:
            self.elgh_path = self.args.data
        else:
            self.elgh_path = self.config['paths']['MIVA_PATH'].strip("\n")
        self.genome_path = self.config['paths']['GENOME_PATH']
        self.variant_cols = ["#CHROM", "POS", "REF", "Allele", "SYMBOL", "Gene", "HGVSp", "AF_ELGH", "UNIPARC",
                             "SWISSPROT", "TREMBL", "Protein_position", "Amino_acids", "SIFT", "PolyPhen",
                             "varipred_id"]
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
