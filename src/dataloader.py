import pandas as pd
import numpy as np
import requests
import pickle as pkl
import os
import gc
import warnings
import argparse
import ensembl_rest
import matplotlib.pyplot as plt

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from pymatgen.io.cif import CifParser

import plot
import utils
import config


class MissenseVariantLoader:
    def __init__(self, preprocess=False, train=False, predict=False, evaluation=False):
        parser = argparse.ArgumentParser(description='Script to process variants')
        parser.add_argument('--data', type=str)
        parser.add_argument('--varipred_input', type=str)
        self.args = parser.parse_args()
        if self.args.data is not None:
            self.elgh_path = self.args.data
        else:
            self.elgh_path = config.MIVA_PATH
        self.genome_path = config.GENOME_PATH
        self.variant_cols = ["#CHROM", "POS", "REF", "Allele", "SYMBOL", "UNIPARC", "SWISSPROT", "TREMBL",
                             "Protein_position", "Amino_acids", "SIFT", "PolyPhen", "varipred_id"]
        self.variant_data = self.load_gh_data()
        self.variant_data = self.variant_data.rename(columns={'Allele': 'ALT'})
        self.variant_data["uniprot_id"] = self.variant_data["SWISSPROT"].fillna(self.variant_data["TREMBL"])

        am = self.load_am_data()
        am['variant_id'] = am['#CHROM'] + '_' + am['POS'].astype(str) + '_' + am['REF'] + '_' + am['ALT']
        self.variant_data['variant_id'] = self.variant_data['#CHROM'] + '_' + self.variant_data['POS'].astype(str) + \
                                            '_' + self.variant_data['REF'] + '_' + self.variant_data['ALT']
        am = am[['am_pathogenicity', 'variant_id']]

        self.variant_data = self.variant_data.merge(am, on='variant_id')

        if preprocess:
            self.process_variants_proteomic()
        elif len(os.listdir('../data/VariPred/input')) == config.NUM_VP_BATCHES:
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
            utils.run_shell_script(config.VP_TRAINING_PATH)

        if predict:
            if self.args.varipred_input is not None:
                variant_files = os.listdir(self.args.varipred_input)
                variant_files = sorted(variant_files, key=utils.extract_number)
                for file in variant_files:
                    if file.endswith('.csv'):
                        self.predict_pathogenicity(file[:-4])
            else:
                self.predict_pathogenicity()

        varipred_output = utils.preprocess_varipred_output(config.VP_OUTPUT_PATH)
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
                train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])
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
            data_folder = self.args.varipred_input.split('/')[3]
            file = f"{data_folder}/{variant_file}"
            utils.run_shell_script(config.VP_INFERENCE_PATH, file)
        else:
            utils.run_shell_script(config.VP_INFERENCE_PATH)

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


class GeneCharacterisation:
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self):
        self.files_and_dirs = os.listdir("../data")
        self.data_name_mapping = {
            "CTD_chem_gene_ixns.csv": "CTD Chemical-Gene Interactions",
            "gnomad.exomes.v2.1.1.lof_metrics.by_gene.csv": "gnomAD Exomes Loss-of-Function Metrics",
            "STRING_PPIs.txt": "STRING Protein-Protein Interactions",
            "tractability.xlsb": "Tractability Scores",
            "FDA_approved_drug_targets_2022.xlsb": "FDA Approved Drug Targets",
            "part-00000-31eba8be-aff8-492e-9edb-4b5e8c821237-c000.snappy.parquet": "Mouse Knockout Phenotypes"
        }
        self.files = self._get_files()
        self.datasets = self.load_data()

        # Population genomics data
        self.gh_data = self.load_gh_data()

        # OpenTargets proof-of-concept model
        # self.bin_tract_features = self._tractability_feature_extractor()
        # self.tract_features = self._tractability_feature_calculator()

        # Our model
        self.alphafold_features = self._alphafold_feature_extractor()
        print(self.alphafold_features)
        # self.ppi_features = self._ppi_feature_extractor()
        # self.mouse_ko_features = self._mouse_knockout_feature_extractor()
        # self.chem_features = self._chem_feature_extractor()
        # self.gnomad_features = self._gnomad_feature_extractor()
        #
        # # Ground truth
        # self.ot_fda_approvals = self._ground_truth_extractor()

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
        if "datasets.pkl" in os.listdir('data/'):
            with open('data/datasets.pkl', 'rb') as fp:
                datasets = pkl.load(fp)
            return datasets
        else:
            for file in self.files:
                file_name = file.split("/")[-1]
                if file_name in self.data_name_mapping.keys():
                    file_id = self.data_name_mapping[file_name]
                    if any(word in file for word in ["csv", "txt"]):
                        datasets[file_id] = pd.read_csv(file)
                    elif "xlsb" in file:
                        datasets[file_id] = pd.read_excel(file)
                    elif "parquet" in file:
                        datasets[file_id] = pd.read_parquet(file)
                    else:
                        raise ValueError(
                            "The file format is not supported. Make sure data is .csv, .txt, Excel, or parquet.")
            with open('data/datasets.pkl', 'wb') as fp:
                pkl.dump(datasets, fp)
            return datasets

    @staticmethod
    def load_gh_data():
        """
        Load the Genes & Health variant data and the reference genome.
        """
        variant_data = pd.read_csv(config.MIVA_PATH, sep="\t")
        variant_data = variant_data.loc[:, ~variant_data.columns.str.contains('^Unnamed')]
        return variant_data

    def _alphafold_feature_extractor(self):
        """
        Extract AlphaFold features from the AlphaFold API. Specifically, we get the average pLDDT score for the proteins
        in our dataset
        """
        uniprot_data = self.gh_data[["SWISSPROT", "TREMBL", "varipred_id"]]
        uniprot_data["uniprot_id"] = uniprot_data["SWISSPROT"].fillna(uniprot_data["TREMBL"])
        uniprot_data = uniprot_data.drop(["SWISSPROT", "TREMBL"], axis=1).rename(columns={"uniprot_id": "UNIPROT"})
        uniprot_ids = uniprot_data["UNIPROT"].unique().tolist()
        extracted_values = {}
        if os.path.exists('../data/alphafold/af_plddt_features.pkl'):
            with open('../data/alphafold/af_plddt_features.pkl', 'rb') as fp:
                extracted_values = pkl.load(fp)
            return extracted_values
        else:
            for qualifier in tqdm(uniprot_ids):
                try:
                    cif_file_path = f"{config.AF_PATH}AF-{qualifier}-F1-model_v4.cif"
                    target_format = "_ma_qa_metric_global.metric_value\t"

                    with open(cif_file_path, "r") as cif_file:
                        for line in cif_file:
                            if line.startswith(target_format):
                                value = line[len(target_format):].strip()
                                extracted_values[qualifier] = float(value)
                except FileNotFoundError:
                    base_url = "https://alphafold.ebi.ac.uk/api/uniprot/summary"
                    api_key = "AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94"

                    url = f"{base_url}/{qualifier}.json?key={api_key}"

                    response = requests.get(url)

                    if response.status_code == 200:
                        data = response.json()
                        value = float(data["structures"][0]["summary"]["confidence_avg_local_score"])
                        extracted_values[qualifier] = value
                    else:
                        print(f"\nError: Unable to fetch data for {qualifier}. Status code: {response.status_code}. "
                              f"Inserting 0.0.")
                        extracted_values[qualifier] = 0.0
            with open('../data/alphafold/af_plddt_features.pkl', 'wb') as fp:
                pkl.dump(extracted_values, fp)
            return extracted_values

    def _chem_feature_extractor(self):
        """
        Extract chemical features from the CTD dataset. Note: we can further disentangle this data based on interaction
        type, e.g. increasing or decreasing action of target. This is not yet implemented.
        """
        keys = list(self.datasets.keys())
        chem_data = self.datasets[keys[0]]
        chem_features = chem_data[["GeneSymbol", "# ChemicalName", "Organism", "InteractionActions"]]
        chem_features = chem_features[chem_features["Organism"] == "Homo sapiens"]
        gene_counts = chem_features["GeneSymbol"].value_counts(normalize=True)
        chem_features = pd.DataFrame({
            "GeneSymbol": gene_counts.index,
            "Count": gene_counts.values,
        })

        return chem_features

    def _gnomad_feature_extractor(self):
        """
        Extract target conservation scores from gnomAD data. Note that pLI measures the probability of a gene being
        loss-of-function intolerant.
        """
        keys = list(self.datasets.keys())
        gnom_data = self.datasets[keys[2]]
        gnom_data_raw = gnom_data[["gene", "pLI"]]
        gnom_data = gnom_data_raw["pLI"].fillna(0.0)

        return gnom_data

    def _tractability_feature_extractor(self):
        """
        Extract tractability scores from the tractability dataset.
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

    def _ground_truth_extractor(self):
        keys = list(self.datasets.keys())
        tract_data_raw = self.datasets[keys[3]]
        sm_cols = tract_data_raw.filter(regex='(SM_B)').columns.tolist()[0]
        ab_cols = tract_data_raw.filter(regex='(AB_B)').columns.tolist()[0]
        pr_cols = tract_data_raw.filter(regex='(PR_B)').columns.tolist()[0]

        ground_truth_sm = tract_data_raw.loc[:, sm_cols]
        ground_truth_ab = tract_data_raw.loc[:, ab_cols]
        ground_truth_pr = tract_data_raw.loc[:, pr_cols]

        return ground_truth_sm, ground_truth_ab, ground_truth_pr

    def _tractability_feature_calculator(self):
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

    def _ppi_feature_extractor(self, threshold=500):
        """
        Featurise PPI data, i.e. count and normalize the PPIs for each protein with a confidence score above a threshold.
        TODO: Use another version of STRING dataset where we can select the experimentally validated PPIs.
        """
        with open('../data/9606.protein.info.v11.5.txt', 'r') as f:
            protein_info = f.readlines()
        protein_info = [x.strip().split('\t') for x in protein_info]
        protein_info = pd.DataFrame(protein_info)
        protein_info.columns = protein_info.iloc[0]

        keys = list(self.datasets.keys())
        string_data_raw = self.datasets[keys[4]]
        string_data_raw[['protein1', 'protein2', 'combined_score']] = string_data_raw[
            'protein1 protein2 combined_score'].str.split(expand=True)
        string_data_raw.drop('protein1 protein2 combined_score', axis=1, inplace=True)
        string_data_raw['combined_score'] = string_data_raw['combined_score'].astype(int)

        count = 0

        if os.path.exists('data/string_data_counts.pkl'):
            with open('data/string_data_counts.pkl', 'rb') as f:
                string_data_counts = pkl.load(f)
                count = len(string_data_counts)

        batch_len = 1000
        loop_len = np.ceil(len(list(string_data_raw['protein1'].unique())) / batch_len)
        for i in range((int(loop_len))):
            if not os.path.exists('data/string_data_counts.pkl'):
                string_data_counts = pd.DataFrame({'Protein': string_data_raw['protein1'].unique()[:batch_len]})
            else:
                try:
                    string_data_counts = pd.DataFrame(
                        {'Protein': string_data_raw['protein1'].unique()[count:count + batch_len]})
                except IndexError:
                    string_data_counts = pd.DataFrame({'Protein': string_data_raw['protein1'].unique()[count:]})

            string_data_counts['Num_PPIs'] = [sum(string_data_raw[(string_data_raw['protein1'] == gene) |
                                                                  (string_data_raw['protein2'] == gene)]
                                                  ['combined_score'] > threshold)
                                              for gene in string_data_counts['Protein']]
            count += 1000
            if os.path.exists('data/string_data_counts.pkl'):
                with open('data/string_data_counts.pkl', 'rb') as f:
                    string_data_counts_prev = pkl.load(f)
                string_data_counts = pd.concat([string_data_counts_prev, string_data_counts], ignore_index=True)

            with open('data/string_data_counts.pkl', 'wb') as f:
                pkl.dump(string_data_counts, f)

        return string_data_counts

    def _mouse_knockout_feature_extractor(self):
        keys = list(self.datasets.keys())
        df = self.datasets[keys[5]]
        target_counts = {}
        for target in df['targetInModel']:
            if target in target_counts:
                target_counts[target] += 1
            else:
                target_counts[target] = 1
        target_freqs = {k: v / sum(target_counts.values()) for k, v in target_counts.items()}
        return target_freqs


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
