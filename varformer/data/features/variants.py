"""PopulationVariantPreprocessor and extract_pvc_features — moved from src/preprocessing.py (Phase 4A)."""
import os
import gzip
import pickle as pkl

import torch
import requests
import time

import pytorch_lightning as pl
import scipy.sparse as sparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import aa_to_idx, three_letter_aa_to_idx
from utils.merge_am_data import merge_am_data


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

            transcript_ids = var_seq_data['Feature'].tolist()
            ensg_ids = var_seq_data['Gene'].tolist()
            prot_pos = var_seq_data['Protein_position'].tolist()
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
