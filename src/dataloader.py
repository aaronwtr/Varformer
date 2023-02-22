import pandas as pd
import requests
from tqdm import tqdm
import pickle as pkl
import os
from Bio import SeqIO


class VariantLoader:
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
        cont_idx, num_variants = self._file_tracker(uniprot_ids_dict)
        if cont_idx == 0:
            with open(self.msa_output, "w") as f:
                f.write("")
            f.close()
        elif cont_idx == num_variants:
            print("Data preprocessing completed!")
            return list(SeqIO.parse(self.msa_output, "fasta"))
        else:
            print("Data preprocessing incomplete. Continuing from where it left off.")
            with open("preprocessing_log.txt", "r") as f:
                last_processed_gene, last_processed_variant = f.read().split("\t")
            f.close()
            new_uniprot_ids_dict = {}
            found_gene = False
            for gene_name in uniprot_ids_dict:
                if found_gene:
                    new_uniprot_ids_dict[gene_name] = uniprot_ids_dict[gene_name]
                if gene_name == last_processed_gene:
                    new_uniprot_ids_dict[gene_name] = uniprot_ids_dict[gene_name][uniprot_ids_dict[gene_name].index(
                        last_processed_variant) + 1:]
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
                        raise ValueError(f"{response.status_code} Could not retrieve the MSA for variant {uniprot_id}"
                                         f" in gene {gene_name}.")
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
        num_variants = 0
        if self.msa_file_name in os.listdir():
            with open(self.msa_file_name, "r") as f:
                count = f.read().count(">")
            f.close()
            for gene_name in uniprot_ids_dict:
                num_variants += len(uniprot_ids_dict[gene_name])
            if count != num_variants:
                cont_idx = count - 1
            else:
                cont_idx = num_variants
        else:
            cont_idx = 0
        return cont_idx, num_variants


class GeneCharacterisation:
    """
    This class loads and combines the different data sources into a single feature matrix to be fed into our model.
    """

    def __init__(self):
        self.files_and_dirs = os.listdir("data")
        self.data_name_mapping = {
            "CTD_chem_gene_ixns.csv": "CTD Chemical-Gene Interactions",
            "gnomad.exomes.v2.1.1.lof_metrics.by_gene.csv": "gnomAD Exomes Loss-of-Function Metrics",
            "STRING_PPIs.txt": "STRING Protein-Protein Interactions",
            "tractability.xlsb": "Tractability Scores",
            "FDA_approved_drug_targets_2022.xlsb": "FDA Approved Drug Targets",
            "part-00000-31eba8be-aff8-492e-9edb-4b5e8c821237-c000.snappy.parquet": "Mouse Knockout Phenotypes"
        }
        self.files = self._get_files()
        self.datasets = self._load_data()
        self.chem_features = self._chem_feature_extractor()
        self.gnomad_features = self._gnomad_feature_extractor()
        self.tract_features = self._tractability_feature_extractor()
        self.tract_truth_features = self._ground_truth_extractor()
        self.ground_truth = self._ground_truth_calculator()

    def _get_files(self):
        """
        Get the files from the data directory.
        """
        files = []
        exclude = ['.DS_Store', 'elgh']

        for file in self.files_and_dirs:
            if "." in file and file not in exclude:
                files.append(f"data/{file}")
            elif file not in exclude:
                file_path = f"data/{file}"
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

    def _load_data(self):
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

    def _ground_truth_calculator(self):
        tract_sm = self.tract_features[0]
        tract_ab = self.tract_features[1]
        tract_pr = self.tract_features[2]
        # TODO Import shap values to use as weights for the tractibility buckets



