import numpy as np
import pandas as pd
import os
import subprocess
import config
import re
import csv
import glob

from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc
from Bio import Seq

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
    # variant_cols = ["#CHROM", "SYMBOL", "UNIPARC", "Protein_position", "Amino_acids"]
    train_cols = ["vp_cv_id", "SYMBOL", "ReferenceAlleleVCF", "AlternateAlleleVCF", "POS", "UNIPARC", "Amino_acids",
                    "Protein_position", "ClinSigSimple", "vp_classification", "vp_probability"]
    variant_data = pd.read_csv(data_path, sep="\t")
    variant_data = variant_data[train_cols]
    batches = np.array_split(variant_data, num_batches)
    for i, batch in enumerate(batches):
        batch.to_csv(f"data/elgh/train_batch_mivas/variant_data_{i + 1}.csv")


def find_error_files(path):
    output_files = os.listdir(path)
    output_files = [file for file in output_files if file.endswith(".csv")]
    output_files = [int(file.split("_")[1].split(".")[0]) for file in output_files]
    output_files = sorted(output_files)
    input_files = np.arange(1, 1001)
    missing_files = np.setdiff1d(input_files, output_files)
    np.savetxt("data/VariPred/missing_vp_train_files.txt", missing_files, fmt="%d")
    print(len(missing_files))


# def run_shell_script(script_path, file_path):
#     try:
#         subprocess.run(["bash", script_path, file_path], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error while running the shell script: {e}")
#     else:
#         print("Shell script executed successfully!")


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
    variant_data_of_interest = variant_data_of_interest[
        ~variant_data_of_interest['Protein_position'].astype(str).str.contains('-')]
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
    if os.path.exists("../data/VariPred/varipred_output_data.csv"):
        return pd.read_csv("../data/VariPred/varipred_output_data.csv", sep="\t")
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
        varipred_data.to_csv("../data/VariPred/varipred_output_data.csv", sep="\t", index=False)
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
    merged_df.to_csv("../data/elgh/varipred_elgh_data.csv", sep="\t", index=False)


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
        clinvar_data.to_csv("../data/clinvar/clinvar_filtered.csv", sep="\t", index=False)
        return clinvar_data


def clinvar_varipred_id(varipred_data, clinvar_data):
    """
    Makes new ID for varipred and clinvar overlap: {gene_name}_{genomic_position}_{ref_allele}_{alt_allele}
    """
    if os.path.exists("../data/clinvar/clinvar_varipred_id_final.csv"):
        clinvar_data = pd.read_csv("../data/clinvar/clinvar_varipred_id_final.csv", sep="\t")
    else:
        clinvar_data["vp_cv_id"] = clinvar_data["GeneSymbol"] + "_" + clinvar_data["Start"].astype(str) + "_" + \
                                   clinvar_data["ReferenceAlleleVCF"] + "_" + clinvar_data["AlternateAlleleVCF"]
        clinvar_data = clinvar_data.drop_duplicates(subset=['vp_cv_id'])
        clinvar_data.to_csv("../data/clinvar/clinvar_vp_id_final.csv", sep="\t", index=False)

    if os.path.exists("../data/VariPred/varipred_vp_id_final.csv"):
        varipred_data = pd.read_csv("../data/VariPred/varipred_vp_id_final.csv", sep="\t")
    else:
        ref_aa = varipred_data["REF"]
        alt_aa = varipred_data["Allele"]
        varipred_data["vp_cv_id"] = varipred_data["SYMBOL"] + "_" + varipred_data["POS"].astype(str) + "_" + \
                                    ref_aa + "_" + alt_aa

        varipred_data = varipred_data.drop_duplicates(subset=['vp_cv_id'])
        varipred_data.to_csv("../data/VariPred/varipred_vp_id_final.csv", sep="\t", index=False)

    return varipred_data, clinvar_data


def combine_varipred_clinvar(varipred_data, clinvar_data):
    """
    Combine the VariPred and ClinVar data. The columns we want to keep are: vp_cv_id, SYMBOL, POS, ReferenceAlleleVCF,
    AlternateAlleleVCF, ClinSigSimple, vp_classification, vp_probability
    """
    if not os.path.exists("../data/merged_varipred_clinvar.csv"):
        merged_df = pd.merge(varipred_data, clinvar_data, on='vp_cv_id', how='inner')
        columns = ["vp_cv_id", "SYMBOL", "ReferenceAlleleVCF", "AlternateAlleleVCF", "POS", "UNIPARC", "Amino_acids",
                   "Protein_position", "ClinSigSimple", "vp_classification", "vp_probability"]
        merged_df = merged_df[columns]
    else:
        merged_df = pd.read_csv("../data/merged_varipred_clinvar.csv", sep="\t")
    merged_df.to_csv("../data/merged_varipred_clinvar.csv", sep="\t", index=False)
    return merged_df


def varipred_eval(eval_df):
    """
    Calculates the performance metrics for VariPred.
    """
    vp_data = eval_df["vp_classification"]
    clinvar_data = eval_df["ClinSigSimple"]

    print("Sample size: " + str(len(vp_data)))

    mcc = round(matthews_corrcoef(clinvar_data, vp_data), 3)
    conf_matrix = confusion_matrix(clinvar_data, vp_data)

    true_negatives = conf_matrix[0, 0]
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    true_positives = conf_matrix[1, 1]

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)
    false_negative_rate = false_negatives / (false_negatives + true_positives)
    recall = true_positives / (true_positives + false_negatives)

    fpr, tpr, thresholds = roc_curve(clinvar_data, eval_df["vp_probability"])
    roc_auc = auc(fpr, tpr)

    print("False positive rate: " + str(round(false_positive_rate, 3)))
    print("False negative rate: " + str(round(false_negative_rate, 3)))
    print("Recall: " + str(round(recall, 3)))
    print("Accuracy:", str(round(accuracy, 3)))
    print("Matthews Correlation Coefficient:", mcc)
    print("ROC AUC:", str(round(roc_auc, 3)))


def varipred_evaluation(varipred_data, clinvar_data, posthoc=False):
    clinvar_data = clinvar_filtering(clinvar_data)
    varipred_data, clinvar_data = clinvar_varipred_id(varipred_data, clinvar_data)
    eval_df = combine_varipred_clinvar(varipred_data, clinvar_data)
    if posthoc:
        eval_df["vp_classification"] = np.where(eval_df["vp_probability"] > 0.018, 1, 0)
    varipred_eval(eval_df)


def combine_train_files():
    input_files = glob.glob('../data/VariPred/train/*.csv')
    with open('../data/VariPred/all_train.csv', 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        for filename in input_files:
            with open(filename, 'r', newline='') as readfile:
                reader = csv.reader(readfile)
                for row in reader:
                    writer.writerow(row)


def downsampler(data):
    pos_data = data[data.label == 1]
    num_samples = len(pos_data) * 6
    neg_samples = data[data.label == 0].sample(n=num_samples, random_state=42)
    downsampled_data = pd.concat([pos_data, neg_samples])
    return downsampled_data
