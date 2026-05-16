"""Random baseline for Varformer paper.

Moved from src/training.py (random) in Phase 5.
"""
import wandb
import numpy as np

from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve, auc
)
from scipy.stats import spearmanr

from varformer.data.pipeline import ModuleDataProcessor


def random(**modules):
    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    config = modules.get('config', None)

    data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()
    np.random.seed(data['config']['hyperparameters']['seed'])

    # Initialize wandb run
    run = wandb.init(
        project="drug-target-prediction",
        config=data['config']["hyperparameters"],
        group="random-baseline"
    )

    # Extract test datasets and class prior
    test_datasets = data['test_labels_per_source']
    all_test_labels = data['test_labels']
    threshold = data['config']['hyperparameters']['threshold']
    class_prior = data['class_prior']  # Probability of positive class

    # For each test dataset
    for dataset_name, gene_ids in test_datasets.items():
        print(f"Testing on {dataset_name} dataset...")

        np.random.shuffle(gene_ids)
        y_test = np.array([all_test_labels[gene_id] for gene_id in gene_ids])

        test_size = len(gene_ids)

        random_probs = np.random.random(test_size)

        threshold = np.percentile(random_probs, (1 - class_prior) * 100)

        # Create binary predictions based on this threshold
        random_preds = (random_probs >= 0.5).astype(int)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, random_preds)
        test_auroc = roc_auc_score(y_test, random_probs)
        test_recall = recall_score(y_test, random_preds)
        test_precision = precision_score(y_test, random_preds)
        precision_arr, recall_arr, _ = precision_recall_curve(y_test, random_probs)
        test_auprc = auc(recall_arr, precision_arr)
        test_f1 = f1_score(y_test, random_preds)
        test_spearman = spearmanr(y_test, random_probs)

        # Log metrics for this test dataset
        wandb.log({
            f"test_acc_{dataset_name}": test_accuracy,
            f"test_auroc_{dataset_name}": test_auroc,
            f"test_recall_{dataset_name}": test_recall,
            f"test_precision_{dataset_name}": test_precision,
            f"test_f1_{dataset_name}": test_f1,
            f"test_auprc_{dataset_name}": test_auprc,
            f"test_spearman_{dataset_name}": test_spearman.correlation
        })

    run.finish()


if __name__ == "__main__":
    import argparse
    from utils import utils as _utils

    parser = argparse.ArgumentParser(description="Run random baseline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()

    config = _utils.load_config(args.config)
    random(pvc=True, go=True, gc=True, config=config)
