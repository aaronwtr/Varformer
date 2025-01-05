import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import matthews_corrcoef, classification_report, roc_auc_score, confusion_matrix, roc_curve, auc


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
