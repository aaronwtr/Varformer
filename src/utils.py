import numpy as np
from Bio import Seq
import pandas as pd
import os
import subprocess
import config
import re


def count_scaling(counts):
    """
    Implements $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$.
    :param counts: array of count
    :return: list of features scaled between 0 and 1.
    """
    return [(count - min(counts)) / (max(counts) - min(counts)) for count in counts]


def lognorm(vector, eps=1e-8):
    """
    Takes in a vector and return a log-normalized spectrum
    """
    vector_base = np.array(vector)
    vector = vector_base + eps
    sqrt_sum = np.sqrt(np.sum(vector ** 2))
    vector = -np.log(vector / sqrt_sum)
    vector = vector - np.max(vector)
    vector = [-element for element in vector if element != 0]
    return vector


def translate_sequence(sequence):
    """
    Translate a DNA sequence to an amino acid sequence.
    """
    seq = Seq.Seq(sequence)
    return seq.translate()


def split_data(data_path, num_batches):
    """
    Splits data into batches.
    """
    variant_cols = ["#CHROM", "SYMBOL", "UNIPARC", "Protein_position", "Amino_acids"]
    variant_data = pd.read_csv(data_path, sep="\t")
    variant_data = variant_data[variant_cols]
    batches = np.array_split(variant_data, num_batches)
    for i, batch in enumerate(batches):
        batch.to_csv(f"data/elgh/batch_mivas/variant_data_{i + 1}.csv")


def find_error_files(path):
    output_files = os.listdir(path)
    output_files = [file for file in output_files if file.endswith(".txt")]
    output_files = [int(file.split("_")[1].split(".")[0]) for file in output_files]
    output_files = sorted(output_files)
    input_files = np.arange(1, 1001)
    missing_files = np.setdiff1d(input_files, output_files)
    # save the missing files as a row separated .txt file
    np.savetxt("data/VariPred/missing_vp_files.txt", missing_files, fmt="%d")
    print(len(missing_files))


def run_shell_script(file_path):
    script_path = config.VP_SCRIPT_PATH
    try:
        subprocess.run(["bash", script_path, file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the shell script: {e}")
    else:
        print("Shell script executed successfully!")


def extract_number(filename):
    # Use regular expression to find the digits in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


# TODO implement VariPred evaluation
# Step 1: Map VariPred output to the original data
# Step 2: Find common identifier between the two datasets
# Step 3: Calculate the performance metrics


def correct_aa_position(target_id):
    if target_id == 'target_id':
        return 'target_id'
    else:
        parts = target_id.split('_')
        aa_position = int(parts[1])
        adjusted_aa_position = aa_position + 1
        parts[1] = str(adjusted_aa_position)
        return '_'.join(parts)


def add_varipred_id():
    variant_data = pd.read_csv(config.MIVA_PATH, sep="\t")
    columns = ["SYMBOL", "Protein_position", "Amino_acids"]
    variant_data_of_interest = variant_data[columns]

    variant_data = variant_data[~variant_data_of_interest['Protein_position'].astype(str).str.contains('-')]
    variant_data_of_interest = variant_data_of_interest[~variant_data_of_interest['Protein_position'].astype(str).str.contains('-')]
    aa_ref = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[0]
    aa_alt = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[1]
    aa_index = variant_data_of_interest["Protein_position"].astype(str)
    variant_data["varipred_id"] = variant_data_of_interest["SYMBOL"] + "_" + aa_index + "_" + aa_ref + "_" \
                                  + aa_alt
    variant_data.to_csv(config.MIVA_PATH, sep="\t", index=False)


def preprocess_varipred_output(varipred_output_path):
    """
    Preprocess the VariPred output to match the original data.
    """
    if os.path.exists("data/VariPred/varipred_output_data.csv"):
        return pd.read_csv("data/VariPred/varipred_output_data.csv", sep="\t")
    else:
        files = os.listdir(varipred_output_path)
        files.sort(key=extract_number)
        dataframes = []
        for filename in files[1:]:
            if filename.endswith(".txt"):
                file_path = os.path.join(config.VP_OUTPUT_PATH, filename)
                df = pd.read_csv(file_path, sep='\t')
                df['target_id'] = df['target_id'].apply(correct_aa_position)
                dataframes.append(df)
        varipred_data = pd.concat(dataframes, ignore_index=True)
        varipred_data.to_csv("data/VariPred/varipred_output_data.csv", sep="\t", index=False)
        return varipred_data


def combine_varipred_elgh(varipred_data, elgh_data):
    merged_df = pd.merge(varipred_data, elgh_data, left_on='target_id', right_on='varipred_id', how='inner')
    merged_df.drop('varipred_id', axis=1, inplace=True)
    columns = elgh_data.columns.tolist()
    columns.remove('varipred_id')
    columns = columns + ['classification', 'probability']
    merged_df = merged_df[columns]
    new_column_names = {'classification': 'vp_classification', 'probability': 'vp_probability'}
    merged_df.rename(columns=new_column_names, inplace=True)
    merged_df.to_csv("data/elgh/varipred_elgh_data.csv", sep="\t", index=False)


def clinvar_filtering(clinvar_data):
    """
    Filters the ClinVar data to only contain rows assembled with GRCh38 and are missense variants.
    """
    if os.path.exists("data/clinvar/clinvar_filtered.csv"):
        return pd.read_csv("data/clinvar/clinvar_filtered.csv", sep="\t")
    else:
        columns = ["Name", "GeneSymbol", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
                   "ClinSigSimple"]
        clinvar_data = clinvar_data[clinvar_data["Assembly"] == "GRCh38"]
        clinvar_data = clinvar_data[clinvar_data["Type"] == "single nucleotide variant"]
        clinvar_data = clinvar_data[
            (clinvar_data["ReviewStatus"] == "criteria provided, single submitter") |
            (clinvar_data["ReviewStatus"] == "criteria provided, multiple submitters, no conflicts")
            ]
        clinvar_data = clinvar_data.dropna(subset=["ReferenceAlleleVCF", "AlternateAlleleVCF"])
        clinvar_data = clinvar_data[columns]
        clinvar_data.to_csv("data/clinvar/clinvar_filtered.csv", sep="\t", index=False)
        return clinvar_data


def clinvar_varipred_id(varipred_data, clinvar_data):
    """
    Makes new ID var varipred and clinvar overlap: {gene_name}_{genomic_position}_{ref_allele}_{alt_allele}
    """
    if os.path.exists("data/clinvar/clinvar_varipred_id_final.csv"):
        clinvar_data = pd.read_csv("data/clinvar/clinvar_varipred_id_final.csv", sep="\t")
    else:
        clinvar_data["vp_cv_id"] = clinvar_data["GeneSymbol"] + "_" + clinvar_data["Start"].astype(str) + "_" + \
                                     clinvar_data["ReferenceAlleleVCF"] + "_" + clinvar_data["AlternateAlleleVCF"]
        clinvar_data = clinvar_data.drop_duplicates(subset=['vp_cv_id'])
        clinvar_data.to_csv("data/clinvar/clinvar_vp_id_final.csv", sep="\t", index=False)

    if os.path.exists("data/VariPred/varipred_vp_id_final.csv"):
        varipred_data = pd.read_csv("data/VariPred/varipred_vp_id_final.csv", sep="\t")
    else:
        ref_aa = varipred_data["Amino_acids"].str.split("/", expand=True)[0]
        alt_aa = varipred_data["Amino_acids"].str.split("/", expand=True)[1]
        varipred_data["vp_cv_id"] = varipred_data["SYMBOL"] + "_" + varipred_data["POS"].astype(str) + "_" + ref_aa \
                                       + "_" + alt_aa
        varipred_data = varipred_data.drop_duplicates(subset=['vp_cv_id'])
        varipred_data.to_csv("data/VariPred/varipred_vp_id_final.csv", sep="\t", index=False)

    return varipred_data, clinvar_data


def combine_varipred_clinvar(varipred_data, clinvar_data):
    """
    Combine the VariPred and ClinVar data. The columns we want to keep are: vp_cv_id, SYMBOL, POS, ReferenceAlleleVCF,
    AlternateAlleleVCF, ClinSigSimple, vp_classification, vp_probability
    """
    merged_df = pd.merge(varipred_data, clinvar_data, on='vp_cv_id', how='inner')
    columns = ["vp_cv_id", "SYMBOL", "POS", "ReferenceAlleleVCF", "AlternateAlleleVCF", "ClinSigSimple",
               "vp_classification", "vp_probability"]
    merged_df = merged_df[columns]
    print(merged_df)


def varipred_evaluation(varipred_data, clinvar_data):
    # TODO:
    # 1. X Map the varipred data to the original as extra columns
    # 2. X Filter the ClinVar data to only contain rows assembled with GRCh38 and are missense variants
    # 3. X Fix bug: varipred data is just 900 in length. Traceback where this happened and fix it.
    # 4. X Make new ID var varipred and clinvar overlap: {gene_name}_{genomic_position}_{ref_allele}_{alt_allele}
    # 5. X Map the overlap between ClinVar and ELGH via chr and position of the variant and ref and alt allele
    clinvar_data = clinvar_filtering(clinvar_data)
    varipred_data, clinvar_data = clinvar_varipred_id(varipred_data, clinvar_data)
    eval_df = combine_varipred_clinvar(varipred_data, clinvar_data)
    
    print(clinvar_data)
