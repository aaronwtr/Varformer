"""GeneOntologyPreprocessor — moved from src/preprocessing.py (Phase 4A)."""
import os
import pickle as pkl
import gzip

import pandas as pd
import numpy as np

from tqdm import tqdm


class GeneOntologyPreprocessor:
    """
    This class processes gene ontology data, specifically it extracts and processes data from the Human Protein Atlas:
    biological processes, molecular functions, subcellular locations and tissue specificity.
    """
    gene_ontology_features: pd.DataFrame
    data: pd.DataFrame

    def __init__(self, config, gcp):
        print("Gene Ontology Preprocessor booting up...")
        self.config = config
        self.gcp = gcp
        self.pop_data = gcp.pop_data
        self.target = gcp.target
        self.gcp_data = gcp.data
        self.full_gcp_data = gcp.full_data
        self.population = gcp.population

        self.hpa_tissue_specificity_features = None
        self.gtex_tissue_specificity_features = None
        self.protein_atlas_feature_names = None

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
