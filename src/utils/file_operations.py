import os
import subprocess
import re
import csv
import glob
import pandas as pd
import numpy as np


def split_data(data_path, num_batches):
    """
    Splits data into batches.
    """
    # variant_cols = ["#CHROM", "SYMBOL", "UNIPARC", "Protein_position", "Amino_acids"]
    train_cols = ["vp_cv_id", "SYMBOL", "ReferenceAlleleVCF", "AlternateAlleleVCF", "POS", "UNIPARC", "Amino_acids",
                  "Protein_position", "ClinSigSimple", "vp_classification", "vp_probability"]
    variant_data = pd.read_csv(data_path, sep="\t")
    variant_data = variant_data[train_cols]
    batches = np.array_split(variant_data, num_batches)
    for i, batch in enumerate(batches):
        batch.to_csv(f"../data/elgh/train_batch_mivas/variant_data_{i + 1}.csv")


def find_error_files(path):
    output_files = os.listdir(path)
    output_files = [file for file in output_files if file.endswith(".csv")]
    output_files = [int(file.split("_")[1].split(".")[0]) for file in output_files]
    output_files = sorted(output_files)
    input_files = np.arange(1, 1001)
    missing_files = np.setdiff1d(input_files, output_files)
    np.savetxt("../data/VariPred/missing_vp_train_files.txt", missing_files, fmt="%d")
    print(len(missing_files))


def run_shell_script(script_path, *file_paths):
    try:
        subprocess.run(["bash", script_path] + list(file_paths), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running the shell script: {e}")
    else:
        print("Shell script executed successfully!")


def extract_number(filename):
    # Use regular expression to find the digits in the filename
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def combine_train_files():
    input_files = glob.glob('../data/VariPred/train/*.csv')
    with open('../data/VariPred/all_train.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        count = 0
        for filename in input_files:
            with open(filename, 'r', newline='') as readfile:
                reader = csv.reader(readfile)
                for row in reader:
                    if count == 0:
                        count += 1
                        writer.writerow(row)
                    elif row[0] == 'target_id':
                        continue
                    # elif len(row[5]) > 1022:
                    #     continue
                    else:
                        writer.writerow(row)
