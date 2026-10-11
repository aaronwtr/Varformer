"""Population variant preprocessing and pathogenicity feature extraction."""
import os
import pickle as pkl

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm

from varformer.data.parsers.alphamissense import merge_am_data


def resolve_variant_features_dir(config, population):
    """Resolve an optional experiment-isolated PVC cache directory."""
    override = os.environ.get("VARFORMER_VARIANT_FEATURE_DIR")
    if override:
        return override
    default = f"{config['paths']['FEATURES_DIR']}/{population}"
    active_link = os.path.join(default, "active_variant_features")
    if os.path.islink(active_link):
        resolved = os.path.realpath(active_link)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(
                f"Active PVC feature link is broken: {active_link} -> {resolved}"
            )
        return resolved
    return default


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

        features_dir = resolve_variant_features_dir(config, self.population)
        print("Obtaining AlphaMissense pathogenicity embeddings...")
        token_schema = config['hyperparameters'].get('variant_token_schema', 'original')
        frequency_qc = bool(config['hyperparameters'].get('variant_frequency_qc', False))
        enhanced_cache = f'{features_dir}/variant_token_features_v1.pkl'
        use_enhanced_cache = token_schema != 'original' or frequency_qc
        cache_path = enhanced_cache if use_enhanced_cache else f'{features_dir}/var_pat_features.pkl'
        if use_enhanced_cache and not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"{cache_path} is required for schema={token_schema!r} with "
                f"variant_frequency_qc={frequency_qc}. Build it with "
                "validation/build_enhanced_variant_cache.py."
            )
        if not os.path.exists(cache_path):
            print("Preparing variant features...")
            self.variant_gh_data(config)
            self.var_pat_features, self.var_gene_map = self.varformer_pathogenicity_input()
            with open(f'{features_dir}/var_pat_features.pkl', 'wb') as file:
                pkl.dump(self.var_pat_features, file)
        else:
            with open(cache_path, 'rb') as f:
                self.var_pat_features = pkl.load(f)
            with open(f'{features_dir}/gene_loc_var_map.pkl', 'rb') as f:
                self.var_gene_map = pkl.load(f)

        if use_enhanced_cache:
            for gene, tensor in self.var_pat_features.items():
                if tensor.ndim != 2 or tensor.shape[1] != 7:
                    raise ValueError(f"Enhanced PVC tensor for {gene} must have 7 columns")

        # Callability is retained as an optional sensitivity analysis. It must
        # not be applied merely because an enhanced token schema is selected:
        # population-relative AN thresholds remove every variant from hundreds
        # of genes in NFE/AFR/AMR and would therefore change the frozen gene
        # universe and splits.
        if frequency_qc:
            threshold = float(config['hyperparameters'].get('min_callability_fraction', 0.8))
            if not 0 <= threshold <= 1:
                raise ValueError(f"min_callability_fraction must be in [0,1], got {threshold}")
            filtered = {}
            filtered_gene_map = {}
            emptied = []
            for gene, tensor in self.var_pat_features.items():
                keep = tensor[:, 6] >= threshold
                if not torch.any(keep):
                    emptied.append(gene)
                else:
                    filtered[gene] = tensor[keep]
                    variant_ids = self.var_gene_map.get(gene, [])
                    if len(variant_ids) != tensor.shape[0]:
                        raise ValueError(f"{gene}: enhanced tensor/gene-map row mismatch")
                    filtered_gene_map[gene] = [
                        variant for variant, retained in zip(variant_ids, keep.tolist()) if retained
                    ]
            if emptied:
                raise ValueError(
                    f"Callability threshold {threshold} removes every variant from "
                    f"{len(emptied)} genes (examples: {emptied[:3]}), which would change splits."
                )
            self.var_pat_features = filtered
            self.var_gene_map = filtered_gene_map

        position_encoding = config['hyperparameters'].get('position_encoding', 'modulo')
        if position_encoding == 'absolute':
            self._restore_absolute_positions(features_dir)
        elif position_encoding != 'modulo':
            raise ValueError(f"Unknown position_encoding: {position_encoding}")

        self.norm = False

        # Ground truth
        self.target = self.full_gcp_data['target']
        self.labels = {key: 1 if key in self.target.tolist() else 0 for key in self.var_pat_features.keys()}

        self.data = {key: value for key, value in self.var_pat_features.items()}
        self.data = {key: value for key, value in self.var_pat_features.items()}
        self.data['labels'] = self.labels

    def _restore_absolute_positions(self, features_dir):
        """Replace cached modulo positions with collision-free protein positions."""
        raw_path = f'{features_dir}/am_pop_merge.pkl'
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Absolute-coordinate sensitivity requires {raw_path}."
            )
        raw = pd.read_pickle(raw_path)
        required = {'variant_id', 'Protein_position'}
        missing_columns = required.difference(raw.columns)
        if missing_columns:
            raise ValueError(f"Missing absolute-position columns: {sorted(missing_columns)}")
        positions = (
            raw.dropna(subset=['variant_id', 'Protein_position'])
            .drop_duplicates('variant_id')
            .set_index('variant_id')['Protein_position']
        )
        restored = {}
        missing_variants = []
        for gene, tensor in self.var_pat_features.items():
            variant_ids = self.var_gene_map.get(gene, [])
            if len(variant_ids) != tensor.shape[0]:
                raise ValueError(
                    f"{gene}: {len(variant_ids)} variant IDs for {tensor.shape[0]} cached rows."
                )
            gene_missing = [variant for variant in variant_ids if variant not in positions.index]
            missing_variants.extend(gene_missing)
            if gene_missing:
                continue
            updated = tensor.clone()
            updated[:, 1] = torch.as_tensor(
                [positions.loc[variant] for variant in variant_ids], dtype=updated.dtype
            )
            restored[gene] = updated
        if missing_variants:
            raise ValueError(
                f"Could not restore absolute positions for {len(missing_variants)} variants "
                f"(examples: {missing_variants[:3]})."
            )
        self.var_pat_features = restored

    def variant_gh_data(self, config):
        print("Preparing GH data for variant-level embeddings...")
        data_dir = config['paths']['FEATURES_DIR']
        if not os.path.exists(f'{data_dir}/{self.population}/am_pop_merge.pkl'):
            am = merge_am_data(self.pop_data, self.population, config=config)
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
        self.pop_data = self.pop_data[
            self.pop_data['Consequence'].str.contains(
                r'(?:^|&)missense_variant(?:&|$)', regex=True, na=False
            )
        ]

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

        all_aas = sorted(set(
            self.pop_data['AA_ref'].dropna().unique().tolist()
            + self.pop_data['AA_alt'].dropna().unique().tolist()
        ))

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
        with open(
            f'{data_dir}/features/{self.population}/missense_mutation_map.pkl', 'wb'
        ) as f:
            pkl.dump(mutation_map, f)

        gene_map = {gene: i for i, gene in enumerate(self.pop_data['Gene'].unique())}
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
