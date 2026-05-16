"""DrugnomeAI baseline for Varformer paper.

Merged from:
  - src/training.py (drugnome_ai function) in Phase 5
  - benchmark/drugnome_ai_testing.py in Phase 5

Note: The subprocess call contains hardcoded local paths for a paper-reproduction
run. These are left as-is intentionally — this script is meant to be run locally.
"""
import os
import subprocess
import warnings
import fnmatch
import pickle

import pandas as pd
import wandb

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    average_precision_score,
    f1_score,
)
from scipy.stats import spearmanr

from varformer.data.pipeline import ModuleDataProcessor


# ---------------------------------------------------------------------------
# Helper: gene name mapping (replicated from benchmark/drugnome_ai_testing.py)
# ---------------------------------------------------------------------------

def map_gene_names(list_of_genes: list, source_type: str, target_type: str) -> dict:
    """Map gene identifiers between namespaces using biorosetta."""
    import biorosetta as br
    idmap = br.IDMapper('all')
    list_of_targets = idmap.convert(list_of_genes, source_type, target_type)
    if 'N/A' in list_of_targets:
        warnings.warn("Some genes were not found in the mapping. Check the input list of genes.")
        missing = [list_of_genes[i] for i, x in enumerate(list_of_targets) if x == 'N/A']
        warnings.warn(f"Number of missing genes: {len(missing)}")
    return dict(zip(list_of_genes, list_of_targets))


# ---------------------------------------------------------------------------
# Prepare step: write test gene list + (optionally) run DrugnomeAI training
# ---------------------------------------------------------------------------

def drugnome_ai(**modules):
    """
    Prepare DrugnomeAI labels and, if already prepared, run the DrugnomeAI pipeline.

    Note: the subprocess call uses hardcoded local paths (paper-reproduction script).
    """
    import utils

    # check if drugnome_ai_labels.txt already exists
    if not os.path.exists("../benchmark/data/drugnomeai/drugnome_ai_labels.txt"):
        drugnome_ai_labels = pd.read_csv("../benchmark/data/drugnomeai/gene_druggable_labels.csv")

        gc = modules.get('gc', False)
        go = modules.get('go', False)
        pvc = modules.get('pvc', False)
        psc = modules.get('psc', False)
        config = modules.get('config', None)

        data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()
        test_data = list(data['test_labels'].keys())

        gene_to_hgnc = utils.utils.map_gene_names(test_data, 'ensg', 'symb')
        test_data_hgnc = [gene_to_hgnc[gene] for gene in test_data if gene in gene_to_hgnc]
        with open("../benchmark/data/drugnomeai/test_genes_elgh.txt", 'w') as f:
            for gene in test_data_hgnc:
                f.write(str(gene) + '\n')

        print("break")
        # drugnome_ai_labels = drugnome_ai_labels[~drugnome_ai_labels['ensembl_gene_id'].isin(test_data)]
        # # write a .txt file separated by new lines with the HGNC gene names of the genes in the drugnome_ai_labels dataframe
        # with open("../benchmark/data/drugnomeai/drugnome_ai_labels.txt", 'w') as f:
        #     for gene in drugnome_ai_labels['Gene_Name']:
        #         f.write(str(gene) + '\n')
        # print("DrugnomeAI labels written to file.")
    else:
        print("DrugnomeAI labels already exist. Skipping...")

        subprocess.run(
            [
                "python",
                "/Users/aaronw/Desktop/PhD/Research/QMUL/Research/genetic-drug-targeting-and-classification/benchmark"
                "/DrugnomeAI-release/drugnome_ai/modules/main/__main__.py",
                "-o", "/Users/aaronw/Desktop/PhD/Research/QMUL/Research/genetic-drug-targeting-and-classification"
                      "/benchmark/DrugnomeAI-release/drugnome_ai/output/processed-feature-tables",
                "-k", "/Users/aaronw/Desktop/PhD/Research/QMUL/Research/genetic-drug-targeting-and-classification"
                      "/benchmark/data/drugnomeai/drugnome_ai_labels.txt"
            ]
        )


# ---------------------------------------------------------------------------
# Evaluation step: score DrugnomeAI predictions against Varformer test set
# (merged from benchmark/drugnome_ai_testing.py)
# ---------------------------------------------------------------------------

def evaluate_drugnome_ai(
    population: str = "nfe",
    data_path: str = "output/processed-feature-tables/",
    test_labels_path: str = None,
    model_path: str = "output/supervised-learning/models/",
    testing_data_per_source_path: str = "../data/test_data/full_test_labels_per_source.pkl",
    testing_data_labels_path: str = "../data/test_data/full_test_labels.pkl",
    wandb_project: str = "varformer-benchmark-v1-04-2025",
    wandb_group: str = "drugnome_ai",
):
    """
    Evaluate DrugnomeAI predictions on the Varformer test set and log to wandb.

    Args:
        population: Population identifier (elgh, nfe, amr, afr).
        data_path: Path to the DrugnomeAI processed feature table directory.
        test_labels_path: Path to test gene list (HGNC symbols, one per line).
        model_path: Path to the directory containing serialised DrugnomeAI models.
        testing_data_per_source_path: Pickle of {source: [ensg_gene_ids]}.
        testing_data_labels_path: Pickle of {ensg_gene_id: 0/1 label}.
        wandb_project: W&B project name.
        wandb_group: W&B run group.
    """
    if test_labels_path is None:
        test_labels_path = f"data/drugnomeai/test_genes_{population}.txt"

    data = pd.read_csv(os.path.join(data_path, "processed_feature_table.tsv"), sep='\t')
    test_labels_df = pd.read_csv(test_labels_path, sep='\t', header=None)
    testing_data_per_source = pd.read_pickle(testing_data_per_source_path)
    testing_data_labels = pd.read_pickle(testing_data_labels_path)

    test_labels_df.columns = ['Gene_Name']
    test_labels_list = test_labels_df['Gene_Name'].tolist()

    data = data[data['Gene_Name'].isin(test_labels_list)]

    # Load models and get predictions
    models = os.listdir(model_path)
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

    # Build gene name mapping
    gene_map = {}
    for source, genes in testing_data_per_source.items():
        mapped_genes = map_gene_names(genes, 'ensg', 'symb')
        gene_map.update(mapped_genes)
    ensg_genes = list(gene_map.keys())
    inv_gene_map = {v: k for k, v in gene_map.items()}

    wandb.init(project=wandb_project, group=wandb_group)

    # Evaluate per source
    for source, source_genes in testing_data_per_source.items():
        print(f"Source: {source}\n")
        source_genes_symb = [gene_map[gene] for gene in source_genes if gene in gene_map]
        source_genes_symb = [gene for gene in source_genes_symb if gene in ensemble_scores['Gene_Name'].tolist()]

        ensemble_subset = ensemble_scores[ensemble_scores['Gene_Name'].isin(source_genes_symb)]
        drgai_preds = ensemble_subset['preds_ensemble'].tolist()
        drgai_probas = ensemble_subset['probas_ensemble'].tolist()

        # Align gene order
        seen = set()
        source_genes_symb_ordered = []
        for gene in ensemble_subset['Gene_Name'].tolist():
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DrugnomeAI baseline: prepare labels or evaluate predictions.")
    subparsers = parser.add_subparsers(dest="command")

    # Prepare subcommand
    prep_parser = subparsers.add_parser("prepare", help="Write test gene list and run DrugnomeAI training.")
    prep_parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Score DrugnomeAI predictions against Varformer test set.")
    eval_parser.add_argument("--population", type=str, default="nfe")
    eval_parser.add_argument("--data-path", type=str, default="output/processed-feature-tables/")
    eval_parser.add_argument("--test-labels", type=str, default=None)
    eval_parser.add_argument("--model-path", type=str, default="output/supervised-learning/models/")
    eval_parser.add_argument("--per-source", type=str, default="../data/test_data/full_test_labels_per_source.pkl")
    eval_parser.add_argument("--labels", type=str, default="../data/test_data/full_test_labels.pkl")
    eval_parser.add_argument("--wandb-project", type=str, default="varformer-benchmark-v1-04-2025")
    eval_parser.add_argument("--wandb-group", type=str, default="drugnome_ai")

    args = parser.parse_args()

    if args.command == "prepare":
        from utils import utils as _utils
        config = _utils.load_config(args.config)
        drugnome_ai(pvc=True, go=True, gc=True, config=config)
    elif args.command == "evaluate":
        evaluate_drugnome_ai(
            population=args.population,
            data_path=args.data_path,
            test_labels_path=args.test_labels,
            model_path=args.model_path,
            testing_data_per_source_path=args.per_source,
            testing_data_labels_path=args.labels,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
        )
    else:
        parser.print_help()
