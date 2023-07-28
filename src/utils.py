import numpy as np
from Bio import Seq
import pandas as pd
import os
import subprocess
import config

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
    output_files = [file for file in output_files if file.endswith(".csv")]
    output_files = [int(file.split("_")[1].split(".")[0]) for file in output_files]
    output_files = sorted(output_files)
    input_files = np.arange(1, 1001)
    missing_files = np.setdiff1d(input_files, output_files)
    # save the missing files as a row separated .txt file
    np.savetxt("data/elgh/missing_miva_files.txt", missing_files, fmt="%d")
    print(len(missing_files))


def run_shell_script(file_path):
    script_path = config.VP_SCRIPT_PATH
    try:
        subprocess.run(["bash", script_path, file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the shell script: {e}")
    else:
        print("Shell script executed successfully!")