"""Population variant preprocessing and pathogenicity feature extraction."""
import os
import pickle as pkl

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

from varformer.data.parsers.alphamissense import merge_am_data


class PopulationVariantPreprocessor:
    """
    This class processes protein variant information, specifically it obtains and processes amino acid sequence embeddings
    and missense variant pathogenicity embeddings, and it processes protein structure confidence scores, in particular
    it generates and processes embeddings of AlphaFold's residue-wise pLDDT score.
    """

    def __init__(self, config, gcp):
        self.gcp = gcp
        self.pop_data = gcp.pop_data
        self.target = gcp.target
        self.gcp_data = gcp.data
        self.full_gcp_data = gcp.full_data
        self.population = gcp.population
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
        self.pop_data['AA_ref'] = components['ref_aa']
        self.pop_data['AA_alt'] = components['alt_aa']

        mutation_map = self.missense_mutation_map()
        data_dir = self.config['paths']['DATA_DIR']

        gene_map = {gene: i for i, gene in enumerate(self.pop_data['Gene'].unique())}
        with open(self.config['paths']['VAR_MAP'], 'rb') as file:
            variant_map = pkl.load(file)
        var_pat_features = {}
        gene_var_map = {}
        for index, row in tqdm(self.pop_data.iterrows(), total=self.pop_data.shape[0]):
            gene = row['Gene']
            variant_id = row['variant_id']
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
