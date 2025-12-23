from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    average_precision_score,
    f1_score
)

from scipy.stats import spearmanr

import pandas as pd
import biorosetta as br

import os
import pickle
import warnings
import fnmatch
import wandb

def map_gene_names(list_of_genes: list, source_type: str, target_type: str) -> dict:
    idmap = br.IDMapper('all')
    list_of_targets = idmap.convert(list_of_genes, source_type, target_type)
    if 'N/A' in list_of_targets:
        warnings.warn("Some genes were not found in the mapping. Check the input list of genes.")
        missing = [list_of_genes[i] for i, x in enumerate(list_of_targets) if x == 'N/A']
        warnings.warn(f"Number of missing genes: {len(missing)}")
    return dict(zip(list_of_genes, list_of_targets))


# open the data
population = "nfe" # elgh, nfe, amr
data_path = "output/processed-feature-tables/"
test_labels = f"data/drugnomeai/test_genes_{population}.txt" # \n separated
model_path = "output/supervised-learning/models/"
testing_data_per_source = "../data/test_data/full_test_labels_per_source.pkl"
testing_data_labels = "../data/test_data/full_test_labels.pkl"

data = pd.read_csv(data_path + "processed_feature_table.tsv", sep='\t')
test_labels = pd.read_csv(test_labels, sep='\t', header=None)
testing_data_per_source = pd.read_pickle(testing_data_per_source)
testing_data_labels = pd.read_pickle(testing_data_labels)

test_labels.columns = ['Gene_Name']
test_labels_list = test_labels['Gene_Name'].tolist()

data = data[data['Gene_Name'].isin(test_labels_list)]

# load the models
models = os.listdir(model_path)

# getting predictions
feature_data = data.drop(columns=['Gene_Name', 'known_gene'])
gene_names_drgnmai = data['Gene_Name'].tolist()
all_probas = {}
for model_it in models:
    if fnmatch.fnmatch(model_it, "iteration_*"):
        continue
    if "GradientBoostingClassifier" not in model_it:
        continue
    model_file_path = os.path.join(model_path, model_it)
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
        print(f"Loaded model: {model}")

    probas = model.predict_proba(feature_data)
    pos_probas = probas[:, 1]
    model_name = model.__class__.__name__
    if model_name not in all_probas:
        all_probas[model_name] = pos_probas
    else:
        all_probas[model_name] = (all_probas[model_name] + pos_probas) / 2


ensemble_scores = pd.DataFrame(all_probas)
ensemble_scores['probas_ensemble'] = ensemble_scores.mean(axis=1)
ensemble_scores['preds_ensemble'] = ensemble_scores['probas_ensemble'].apply(lambda x: 1 if x > 0.5 else 0)
ensemble_scores['Gene_Name'] = gene_names_drgnmai

# map the gene_names to HGNC
gene_map = {}
for source, genes in testing_data_per_source.items():
    mapped_genes = map_gene_names(genes, 'ensg', 'symb')
    gene_map.update(mapped_genes)
ensg_genes = list(gene_map.keys())
inv_gene_map = {v: k for k, v in gene_map.items()}

wandb.init(
        project="varformer-benchmark-v1-04-2025",
        group="drugnome_ai",
    )

# evaluation
for source, source_genes in testing_data_per_source.items():
    print(f"Source: {source}\n")
    source_genes_symb = [gene_map[gene] for gene in source_genes if gene in gene_map]
    source_genes_symb = [gene for gene in source_genes_symb if gene in ensemble_scores['Gene_Name'].tolist()]
    # precomputed_probas_tclin = precomputed_scores[precomputed_scores['Gene Name'].isin(source_genes_symb)]['Tclin'].tolist()
    # precomputed_preds_tclin = precomputed_probas_tclin.apply(lambda x: 1 if x > 0.5 else 0).tolist()

    ensemble_subset = ensemble_scores[ensemble_scores['Gene_Name'].isin(source_genes_symb)]
    drgai_preds = ensemble_subset['preds_ensemble'].tolist()
    drgai_probas = ensemble_subset['probas_ensemble'].tolist()

    # Step 1: Get the list of gene symbols from the ensemble_subset
    ensemble_gene_order = ensemble_subset['Gene_Name'].tolist()

    # Use a set to track seen gene symbols (to avoid duplicates)
    seen = set()
    source_genes_symb_ordered = []
    for gene in ensemble_gene_order:
        if gene in source_genes_symb and gene not in seen:
            source_genes_symb_ordered.append(gene)
            seen.add(gene)

    source_genes_ensg = [inv_gene_map[gene] for gene in source_genes_symb_ordered if gene in inv_gene_map]
    labels = [testing_data_labels[gene] for gene in source_genes_ensg]

    acc = accuracy_score(labels, drgai_preds)
    auroc = roc_auc_score(labels, drgai_probas)
    precision = precision_score(labels, drgai_preds)
    recall = recall_score(labels, drgai_preds)
    f1 = f1_score(labels, drgai_preds)
    avg_precision = average_precision_score(labels, drgai_probas)
    spearman = spearmanr(labels, drgai_probas)[0]

    wandb.log({
        f"test_acc_{source}": acc,
        f"test_auroc_{source}": auroc,
        f"test_precision_{source}": precision,
        f"test_recall_{source}": recall,
        f"test_f1_{source}": f1,
        f"test_auprc_{source}": avg_precision,
        f"test_spearman_{source}": spearman
    })

print("Done")
