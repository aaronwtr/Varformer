import numpy as np
import pandas as pd
import pickle as pkl
import os
import subprocess
import re
import csv
import glob
import biorosetta as br
import warnings
import requests
import yaml

from sklearn.metrics import matthews_corrcoef, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from Bio import Seq, SeqIO, Entrez


def count_scaling(counts):
    """
    Implements \(x' = \frac{x - x_{min}}{x_{max} - x_{min}}\).
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


def add_varipred_id(config):
    variant_data = pd.read_csv(config['paths']['MIVA_PATH'], sep="\t")
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
    variant_data.to_csv(config['paths']['MIVA_PATH'], sep="\t", index=False)


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
                file_path = os.path.join(varipred_output_path, filename)
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
    new_column_names = {'classification': 'vpgh_classification', 'probability': 'vpgh_probability'}
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
        alt_aa = varipred_data["ALT"]
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


def downsampler(data):
    pos_data = data[data.label == 1]
    neg_data = data[data.label == 0]
    num_samples = len(pos_data) * 4
    neg_data = neg_data.sample(n=num_samples, random_state=42)
    downsampled_data = pd.concat([pos_data, neg_data])
    return downsampled_data


def analyze_legacy_pathogenicity(variant_data):
    """
    Analyze the pathogenicity determined by SIFT and PolyPhen.
    """
    sift = variant_data["SIFT"].values
    polyphen = variant_data["PolyPhen"].values

    pathogenicity = pd.DataFrame({"sift": sift, "polyphen": polyphen})

    pathogenicity['sift'] = pathogenicity['sift'].str.extract(r'\((.*?)\)').astype(float)
    pathogenicity['polyphen'] = pathogenicity['polyphen'].str.extract(r'\((.*?)\)').astype(float)
    plot.variant_sparsity_barplot(pathogenicity, save=True)

    num_vars = len(list(pathogenicity['polyphen'].values))

    sift_nan_count = pathogenicity['sift'].isna().sum()
    polyphen_nan_count = pathogenicity['polyphen'].isna().sum()

    pp_sparsity = round(polyphen_nan_count / num_vars, 3)
    sift_sparsity = round(sift_nan_count / num_vars, 3)

    print(f"PolyPhen sparsity: {pp_sparsity}")
    print(f"SIFT sparsity: {sift_sparsity}")

    plot.pathogenicity_correlation_plot(pathogenicity, save=True)

    print("Done analyzing pathogenicity.")


def evaluate_am(am_data, fold):
    """
    Evaluate the performance of the AM model. Calculate the labels for AM given the same threshold as VariPred.
    Map the labels from the test data to the AM data and calculate the performance metrics.
    """
    test_data = pd.read_csv(f"../data/VariPred/test_downsample_fold_{fold}.csv")
    test_data = test_data[["seq_id", "label"]]

    am_data["seq_id"] = am_data["SYMBOL"] + "_" + am_data["POS"].astype(str) + "_" + am_data["REF"] + "_" + \
                        am_data["ALT"]

    vars_not_in_am = test_data[~test_data.seq_id.isin(am_data.seq_id)]

    merged_df = pd.merge(am_data, test_data, on='seq_id', how='inner')
    # drop vp_probability

    vp_test_output = pd.read_csv(f"../data/VariPred/output/varipred_output_finetuned_fold_{fold}.txt", sep="\t")
    vp_test_output = vp_test_output[["target_id", "probability"]]
    vp_test_output.rename(columns={"target_id": "seq_id"}, inplace=True)

    merged_df = pd.merge(merged_df, vp_test_output, on='seq_id', how='inner')  # add new model outputs
    merged_df.rename(columns={"probability": "vpgh_pathogenicity"}, inplace=True)

    # add a majority class baseline
    merged_df["majority_baseline"] = 0

    threshold = 0.45

    merged_df["am_classification"] = np.where(merged_df["am_pathogenicity"] > threshold, 1, 0)
    merged_df["vpgh_classification"] = np.where(merged_df["vpgh_pathogenicity"] > threshold, 1, 0)
    merged_df["vp_classification"] = np.where(merged_df["vp_pathogenicity"] > threshold, 1, 0)

    all_columns = list(merged_df.columns)

    columns_to_reorder = ['am_classification', 'am_pathogenicity', 'vp_classification', 'vp_pathogenicity',
                          'vpgh_classification', 'vpgh_pathogenicity']

    for col in columns_to_reorder:
        all_columns.remove(col)

    all_columns = all_columns + columns_to_reorder

    merged_df = merged_df[all_columns]

    print("Majority class baseline performance metrics:")
    eval_metrics(merged_df["label"], merged_df["majority_baseline"], 0.5, 'majority_baseline', fold=fold)


    print("VariPred performance metrics:")
    eval_metrics(merged_df["label"], merged_df["vp_pathogenicity"], threshold, 'vp', fold=fold)

    print("VariPred-GH performance metrics:")
    eval_metrics(merged_df["label"], merged_df["vpgh_pathogenicity"], threshold, 'vpgh', fold=fold)

    print("AlphaMissense performance metrics:")
    eval_metrics(merged_df["label"], merged_df["am_pathogenicity"], threshold, 'am', fold=fold)


# noinspection PyTypeChecker
def eval_metrics(y_true, preds, threshold, model, fold):
    if os.path.exists(f"../data/VariPred/output/{model}_crossval_results.pkl"):
        with open(f"../data/VariPred/output/{model}_crossval_results.pkl", "rb") as f:
            results = pkl.load(f)
    else:
        results = {}
    label_names = {'0': 0, '1': 1}

    y_true_np = np.array(y_true)

    spearman_corr, _ = spearmanr(y_true_np, preds)
    print('Spearman correlation: ', spearman_corr)

    auc_value = roc_auc_score(y_true, preds)
    print('AUC score: ', auc_value)

    y_true_np = np.array(y_true)
    preds_bin = np.array(preds >= threshold, dtype=int)

    mcc = matthews_corrcoef(y_true_np, preds_bin)
    print('MCC: ', mcc)

    report = classification_report(
        y_true_np, preds_bin, target_names=label_names, output_dict=True)
    print(report)
    results[f'fold_{fold}'] = {'auroc': auc_value,
                               'mcc': mcc,
                               'spearman_corr': spearman_corr,
                               'classification_report': report
                               }

    with open(f"../data/VariPred/output/{model}_crossval_results.pkl", "wb") as f:
        pkl.dump(results, f)


def map_gene_names(list_of_genes, source_type, target_type):
    idmap = br.IDMapper('all')
    list_of_targets = idmap.convert(list_of_genes, source_type, target_type)
    if 'N/A' in list_of_targets:
        warnings.warn("Some genes were not found in the mapping. Check the input list of genes.")
        missing = [list_of_genes[i] for i, x in enumerate(list_of_targets) if x == 'N/A']
        warnings.warn(f"Number of missing genes: {len(missing)}")
    return dict(zip(list_of_genes, list_of_targets))


def get_protein_length(ensp, ensg):
    ensp_api_url = f"https://rest.ensembl.org/sequence/id/{ensp}"
    ensg_api_url = f"https://rest.ensembl.org/sequence/id/{ensg}?type=protein;multiple_sequences=1"

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        response = requests.get(ensp_api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()

            protein = data.get('seq', None)
            protein_length = len(protein)

            if protein_length is not None:
                return protein_length
            else:
                raise KeyError(f"Protein length for {ensp} not found in the response.")
        else:
            raise KeyError(f"Failed to retrieve protein information. Status code: {response.status_code}")
    except KeyError:
        response = requests.get(ensg_api_url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            protein = data[0].get('seq', None)
            protein_length = len(protein)

            if protein_length is not None:
                return protein_length
            else:
                raise KeyError(f"Protein length for {ensg} not found in the response.")
        else:
            raise KeyError(f"Failed to retrieve protein information. Status code: {response.status_code}")


def get_protein_length_up(up):
    uniprot_api_url = f"https://www.uniprot.org/uniprot/{up}.fasta"

    response = requests.get(uniprot_api_url)
    try:
        if response.status_code == 200:
            output = response.text
            protein_sequence = output.split("\n")[1:]
            protein_sequence = "".join(protein_sequence)
            protein_length = len(protein_sequence)
            if protein_length == 0:
                raise ValueError(f"Protein length for {up} not found in the response.")

            return protein_length
        else:
            raise KeyError(f"Failed to retrieve protein sequence from UniProt. Status code: {response.status_code}")

    except ValueError:
        print(f"Protein length for {up} not found in the response.")
        return 0
    except KeyError:
        print(f"Failed to retrieve protein sequence from UniProt. Status code: {response.status_code}")
        return 0


def count_zeros(df):
    """
    For each column in a given dataframe, count how many zeros occur
    :return:
    """
    print("Feature sparsity:")

    for col in df.columns:
        num_zeros = len(df[df[col] == 0])
        print(f"{col}: {round(num_zeros / len(df) * 100, 2)}%")
