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


def add_varipred_id():
    variant_data = pd.read_csv(config.MIVA_PATH, sep="\t")
    columns = ["SYMBOL", "Protein_position", "Amino_acids"]
    variant_data_of_interest = variant_data[columns]

    aa_ref = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[0]
    aa_alt = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[1]
    variant_data["varipred_id"] = variant_data_of_interest["SYMBOL"] + "_" + \
                                                    variant_data_of_interest["Protein_position"].astype(str) + "_" + \
                                                            aa_ref + "_" + aa_alt
    variant_data.to_csv(config.MIVA_PATH_2, sep="\t", index=False)


def preprocess_varipred_output(varipred_output_path):
    """
    Preprocess the VariPred output to match the original data.
    """
    files = os.listdir(varipred_output_path)
    files.sort(key=extract_number)
    # open the first file in varipred_output_path as df
    variant_file = files[1]
    df = pd.read_csv(f"{varipred_output_path}{variant_file}", sep="\t")
    print(df)


def varipred_evaluation(elgh_data, varipred_data):
    # TODO Make a column in the variant_loader in the format {gene}_{aa_pos}_{aa_ref}_{aa_alt}
    pass
