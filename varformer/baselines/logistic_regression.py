"""Logistic-regression baseline: a linear model over the same GC/GO/PVC features."""
import os
import datetime
import pickle as pkl

import wandb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve, auc
)
from scipy.stats import spearmanr

from varformer.data.pipeline import ModuleDataProcessor
from varformer.baselines.preprocessor import LogisticRegressionPreprocessor


def logistic_regression(**modules):
    """
    Train a logistic regression model on the same data that would be used for the neural network model.
    Record results in wandb under the logistic-regression-1 group.

    Args:
        modules: Keyword arguments containing gc, go, pvc, psc flags and config
    """

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    config = modules.get('config', None)

    data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()

    config = data['config']
    hyperparameters = config['hyperparameters']
    population = hyperparameters['population']

    run = wandb.init(
        project="drug-target-prediction",
        config=config["hyperparameters"],
        group=f"logistic-regression-{population}"
    )

    # Process features using the custom preprocessor
    print("Preparing features for logistic regression...")
    preprocessor = LogisticRegressionPreprocessor(config, data)
    processed_data = preprocessor.prepare_features()

    # Extract train, validation and test sets
    X_train = processed_data['train']['X']
    y_train = processed_data['train']['y']
    train_genes = processed_data['train']['genes']

    X_val = processed_data['val']['X']
    y_val = processed_data['val']['y']

    test_datasets = processed_data['test']

    # Log feature dimensions
    feature_dim = X_train.shape[1]
    print(f"Training with {feature_dim} features on {len(train_genes)} genes")
    wandb.log({"num_features": feature_dim, "num_train_genes": len(train_genes)})

    # Initialize and train logistic regression model
    print("Training logistic regression model...")
    model = LogisticRegression(
        C=hyperparameters['C'],
        penalty=hyperparameters['penalty'],
        solver=hyperparameters['solver'],
        max_iter=hyperparameters['max_iter'],
        class_weight=hyperparameters['class_weight'],
        random_state=hyperparameters['seed'],
        verbose=1
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= hyperparameters['threshold']).astype(int)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_auroc = roc_auc_score(y_val, val_probs)
    val_recall = recall_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds)
    precision_arr, recall_arr, _ = precision_recall_curve(y_val, val_probs)
    val_auprc = auc(recall_arr, precision_arr)
    val_f1 = f1_score(y_val, val_preds)
    val_spearman = spearmanr(y_val, val_probs)

    # Log validation metrics
    wandb.log({
        "val_accuracy": val_accuracy,
        "val_auroc": val_auroc,
        "val_recall": val_recall,
        "val_precision": val_precision,
        "val_auprc": val_auprc,
        "val_f1": val_f1,
        "val_spearman": val_spearman.correlation
    })

    # Save the model
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    current_time = datetime.datetime.now().strftime("%H-%M-%S")
    checkpoint_dir = f'checkpoints/{current_date}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_path = f"{checkpoint_dir}/logistic_regression_model_{current_time}.pkl"
    with open(model_path, 'wb') as f:
        pkl.dump(model, f)

    # Test on the same test datasets
    for dataset_name, test_data in test_datasets.items():
        print(f"Testing on {dataset_name} dataset...")
        X_test = test_data['X']
        y_test = test_data['y']
        test_genes = test_data['genes']

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= hyperparameters['threshold']).astype(int)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, test_preds)
        test_auroc = roc_auc_score(y_test, test_probs)
        test_recall = recall_score(y_test, test_preds)
        test_precision = precision_score(y_test, test_preds)
        prc = precision_recall_curve(y_test, test_probs)
        test_auprc = auc(prc[1], prc[0])
        test_f1 = f1_score(y_test, test_preds)
        test_spearman = spearmanr(y_test, test_probs)

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

        # Create table of high-confidence predictions
        predictions_table = wandb.Table(columns=["Gene", "True Label", "Predicted Probability"])
        high_conf_indices = np.argsort(np.abs(test_probs - 0.5))[-20:]  # 20 most confident predictions
        for idx in high_conf_indices:
            predictions_table.add_data(test_genes[idx], int(y_test[idx]), float(test_probs[idx]))

        wandb.log({f"test_{dataset_name}_predictions": predictions_table})

    run.finish()


if __name__ == "__main__":
    from varformer.config import Config

    config = Config.load()
    logistic_regression(pvc=True, go=True, gc=True, config=config)
