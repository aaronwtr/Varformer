import gc

import yaml
import os
import torch
import optuna
import wandb
import datetime
import utils
import subprocess

import pytorch_lightning as pl
import pandas as pd
import numpy as np
import pickle as pkl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc
from scipy.stats import spearmanr
from typing import Dict, Union, Optional
from tqdm import tqdm
from optuna.samplers import GridSampler, TPESampler
from optuna.integration import PyTorchLightningPruningCallback

from dataloader import DrugTargetData, ModuleDataProcessor, VarformerDataset, MultiModalData, MultiModalDataLoader
from models.target_identifier import MultiModalTargetIdentifier
from models.lightning import (MLPLightningTargetIdentifier, ShardedVarformerLightningTargetIdentifier,
                              MultiModalLightningTargetIdentifier)
from preprocessing import ModelPreprocessorEval, ModelPreprocessorInference, LogisticRegressionPreprocessor
from utils.custom_callbacks import BestThresholdCallback


def tune():
    sampler = TPESampler()
    study = optuna.create_study(
        study_name="gdtp_varformer_tpe",
        direction="maximize",
        sampler=sampler
    )
    n_trials = 100  # Initial trial budget for TPE (adjust as needed)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial, optimized for auROC:")
    best_trial = study.best_trial

    print("  Metric: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


def objective(trial: optuna.trial.Trial) -> float:
    with open("cluster_config.yml", 'r') as stream:
        config = yaml.safe_load(stream)
    pl.seed_everything(config['hyperparameters']['seed'])
    # Explicit categorical values for hyperparameters
    config['hyperparameters']['lr_start'] = trial.suggest_categorical(
        'lr_start', [1e-5, 3e-5, 1e-4])  # Retain all – 3e-5 performed best overall
    config['hyperparameters']['lr_fraction'] = trial.suggest_categorical(
        'lr_fraction', [1 / 10, 1 / 100])
    config['hyperparameters']['lr_end'] = float(config['hyperparameters']['lr_start']) * float(
        config['hyperparameters']['lr_fraction'])
    config['hyperparameters']['weight_decay'] = trial.suggest_categorical(
        'weight_decay',
        [5e-4, 3e-4, 5e-3, 3e-3])
    config['hyperparameters']['batch_size'] = trial.suggest_categorical(
        'batch_size', [64, 128, 256, 512])
    config['hyperparameters']['dropout'] = trial.suggest_categorical('dropout', [0.1, 0.2, 0.3])
    config['hyperparameters']['depth_cls_head'] = trial.suggest_categorical('depth_cls_head', [2, 4, 6])
    config['hyperparameters']['num_encoder_layers'] = trial.suggest_categorical('num_encoder_layers', [4, 6, 8])
    config['hyperparameters']['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
    config['hyperparameters']['gc_width'] = trial.suggest_categorical('gc_width', [16, 32, 64])
    config['hyperparameters']['go_width'] = trial.suggest_categorical('go_width', [256, 512])
    config['hyperparameters']['d_model'] = trial.suggest_categorical('d_model', [256, 512, 768, 1024])
    config['hyperparameters']['gv_attn_dim'] = trial.suggest_categorical('gv_attn_dim', [512, 1024, 2048])
    config['hyperparameters']['dim_feedforward'] = trial.suggest_categorical('dim_feedforward', [1024, 2048, 4096])

    scheduler_name = trial.suggest_categorical('scheduler', ['CosineAnnealingLR'])
    config['hyperparameters']['scheduler'] = scheduler_name

    if scheduler_name == 'CosineAnnealingLR':
        config['hyperparameters']['T0'] = trial.suggest_categorical('T0_cosine', [100, 200])
    elif scheduler_name == 'ExponentialLR':
        config['hyperparameters']['gamma'] = trial.suggest_categorical('gamma_exp', [0.85, 0.9, 0.95, 0.99])
    # No scheduler case can be handled by default in the training loop if scheduler is None

    hyperparameters = dict(
        lr_start=config['hyperparameters']['lr_start'],
        lr_end=config['hyperparameters']['lr_end'],
        T0=config['hyperparameters']['T0'],
        cls_depth=config['hyperparameters']['depth_cls_head'],
        num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
        nhead=config['hyperparameters']['nhead'],
        gc_width=config['hyperparameters']['gc_width'],
        go_width=config['hyperparameters']['go_width'],
        d_model=config['hyperparameters']['d_model'],
        dim_feedforward=config['hyperparameters']['dim_feedforward'],
        gv_attn_dim=config['hyperparameters']['gv_attn_dim'],
        batch_size=config['hyperparameters']['batch_size'],
        optimizer=config['hyperparameters']['optimizer'],
        epochs=config['hyperparameters']['epochs'],
        dropout=config['hyperparameters']['dropout'],
        weight_decay=config['hyperparameters']['weight_decay']
    )

    # Add only the scheduler-specific parameters that are relevant
    if scheduler_name == 'CosineAnnealingLR':
        hyperparameters['T0'] = config['hyperparameters']['T0']
    elif scheduler_name == 'StepLR':
        hyperparameters['step_size'] = config['hyperparameters']['step_size']
        hyperparameters['gamma'] = config['hyperparameters']['gamma']
    elif scheduler_name == 'ExponentialLR':
        hyperparameters['gamma'] = config['hyperparameters']['gamma']
    elif scheduler_name == 'ReduceLROnPlateau':
        hyperparameters['factor'] = config['hyperparameters']['factor']
        hyperparameters['patience'] = config['hyperparameters']['patience']

    # Initialize a wandb run
    run = wandb.init(
        project="varformer-hyperparameter-tuning",
        dir="/data/scratch/bty174/genomic-drug-targeting/src/",
        config=hyperparameters,
        group="varformer-tuning-run-6"
    )

    data = ModuleDataProcessor(True, True, True, False).process()

    preprocessor = ModelPreprocessor(config, data)
    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params}, allow_val_change=True)

    utils.utils.set_seed(config['hyperparameters']['seed'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = PyTorchLightningPruningCallback(trial, monitor='val_f1')

    if torch.cuda.device_count() > 1:
        trainer_kwargs = {
            'max_epochs': int(config['hyperparameters']['epochs']),
            'accelerator': accelerator,
            'enable_progress_bar': True,
            'log_every_n_steps': 1,
            'precision': config['hyperparameters']['precision'],
            'logger': WandbLogger(wandb.run),
            'callbacks': [lr_monitor, early_stop_callback],
            'strategy': "ddp_find_unused_parameters_true",
            'devices': -1,
            'deterministic': True,
            'num_sanity_val_steps': 0
        }
    else:
        # eff_batch_size = int(config['hyperparameters']['grad_accum']) * int(config['hyperparameters']['batch_size'])
        # print(f"\nTraining with effective batch size {eff_batch_size}\n")
        if config['hyperparameters']['grad_accum'] is not None:
            trainer_kwargs = {
                'max_epochs': int(config['hyperparameters']['epochs']),
                'accelerator': accelerator,
                'enable_progress_bar': True,
                'log_every_n_steps': 1,
                'precision': config['hyperparameters']['precision'],
                'logger': WandbLogger(wandb.run),
                'callbacks': [lr_monitor, early_stop_callback],
                'deterministic': True,
                'num_sanity_val_steps': 0
            }
        else:
            trainer_kwargs = {
                'max_epochs': int(config['hyperparameters']['epochs']),
                'accelerator': accelerator,
                'enable_progress_bar': True,
                'log_every_n_steps': 1,
                'precision': config['hyperparameters']['precision'],
                'logger': WandbLogger(wandb.run),
                'callbacks': [lr_monitor, early_stop_callback],
                'deterministic': True,
                'accumulate_grad_batches': config['hyperparameters']['grad_accum'],
                'num_sanity_val_steps': 0
            }

    # Only add gradient_clip_val if it exists and is not None
    if config['hyperparameters'].get('gradient_clip_val') is not None:
        trainer_kwargs['gradient_clip_val'] = config['hyperparameters']['gradient_clip_val']

    # Create the trainer with the kwargs
    trainer = pl.Trainer(**trainer_kwargs)
    print("Trainer initialized successfully")

    print("Starting fit . . . ")
    try:
        trainer.fit(model, train_combined, val_combined)
        trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')
        run.finish()
        # remove the lightning logs directory if it exists
        if os.path.exists("lightning_logs"):
            os.system("rm -r lightning_logs")
        return trainer.callback_metrics["val_f1"].item()
    except Exception as e:
        print(f"{e}\n")
        print("Out-of-memory error. Architecture is too big.")
        run.finish()
        return 0.0


def train_model(data, split_idx=None):
    torch.set_float32_matmul_precision('medium')
    config = data['config']
    # Initialize wandb run
    hyperparameters = dict(
        lr_start=config['hyperparameters']['lr_start'],
        lr_end=config['hyperparameters']['lr_end'],
        T0=config['hyperparameters']['T0'],
        cls_depth=config['hyperparameters']['depth_cls_head'],
        num_encoder_layers=config['hyperparameters']['num_encoder_layers'],
        nhead=config['hyperparameters']['nhead'],
        gc_width=config['hyperparameters']['gc_width'],
        go_width=config['hyperparameters']['go_width'],
        d_model=config['hyperparameters']['d_model'],
        dim_feedforward=config['hyperparameters']['dim_feedforward'],
        gv_attn_dim=config['hyperparameters']['gv_attn_dim'],
        batch_size=config['hyperparameters']['batch_size'],
        optimizer=config['hyperparameters']['optimizer'],
        epochs=config['hyperparameters']['epochs'],
        dropout=config['hyperparameters']['dropout'],
        weight_decay=config['hyperparameters']['weight_decay'],
        threshold=config['hyperparameters']['threshold']
    )

    if config['hyperparameters']['mode'] == 'eval':
        run = wandb.init(
            project="drug-target-prediction",
            config=hyperparameters,
            group="multimodal-training-run-2"
        )
        preprocessor = ModelPreprocessorEval(config, data)
    elif config['hyperparameters']['mode'] == 'inference':
        run = wandb.init(
            project="drug-target-prediction",
            config=hyperparameters,
            group="inference-run-1"
        )
        preprocessor = ModelPreprocessorInference(config, data, split_idx)
    else:
        raise ValueError("Invalid mode in config. Please set 'mode' to either 'eval' or 'inference'.")

    model, train_combined, val_combined, test_combined, hyperparameters, accelerator = preprocessor.model_init()

    # Log model parameters
    model_summary = ModelSummary(model)
    print(model_summary)
    total_params = model_summary.total_parameters
    wandb.config.update({"total_params": total_params})

    # Setup training callbacks
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")
    checkpoint_dir = f'checkpoints/{current_date}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_spearman',
        dirpath=checkpoint_dir,
        filename=f"seed{config['hyperparameters']['seed']}" + '-{epoch:02d}-{val_spearman:.2f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    best_threshold_callback = BestThresholdCallback(
        monitor='val_spearman',
        mode='max'
    )

    utils.utils.set_seed(config['hyperparameters']['seed'])
    # Configure trainer based on available GPUs
    if torch.cuda.device_count() > 1:
        trainer = pl.Trainer(
            max_epochs=int(config['hyperparameters']['epochs']),
            accelerator=accelerator,
            enable_progress_bar=True,
            log_every_n_steps=1,
            logger=WandbLogger(wandb.run),
            callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
            precision=config['hyperparameters']['precision'],
            strategy="ddp_find_unused_parameters_true",
            devices=-1,
            gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
            deterministic=True
        )
    else:
        # eff_batch_size = int(config['hyperparameters']['grad_accum']) * int(config['hyperparameters']['batch_size'])
        # print(f"\nTraining with effective batch size {eff_batch_size}\n")
        if config['hyperparameters']['grad_accum'] is not None:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=1,
                precision=config['hyperparameters']['precision'],
                logger=WandbLogger(wandb.run),
                callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
                gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                accumulate_grad_batches=config['hyperparameters']['grad_accum'],
                limit_train_batches=0.1,  # Limit to 10% of training data for debugging purposes
                deterministic=True
            )
        else:
            trainer = pl.Trainer(
                max_epochs=int(config['hyperparameters']['epochs']),
                accelerator=accelerator,
                enable_progress_bar=True,
                log_every_n_steps=10,
                precision=config['hyperparameters']['precision'],
                logger=WandbLogger(wandb.run),
                callbacks=[lr_monitor, checkpoint_callback, best_threshold_callback],
                gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
                deterministic=True
            )

    trainer.fit(model, train_combined, val_combined)

    print("Starting testing...")
    if config['hyperparameters']['mode'] == 'eval':
        trainer.test(dataloaders=test_combined["pfam"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["rcnt"], ckpt_path='best')
        trainer.test(dataloaders=test_combined["pharos"], ckpt_path='best')
    elif config['hyperparameters']['mode'] == 'inference':
        prediction_results = trainer.predict(model=model, dataloaders=test_combined, ckpt_path='best')
        output_path = f"{config['paths']['VARFORMER_PREDICT_OUTPUT']}predictions_split_{split_idx}.pkl"
        torch.save(prediction_results, output_path)
        print(f"Predictions for split {split_idx} saved to {output_path}")
    run.finish()


def setup_training(**modules):
    torch.set_float32_matmul_precision('medium')

    print("Training teacher model...\n")

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    config = modules.get('config', None)

    data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()

    if config is not None:
        pl.seed_everything(config['hyperparameters']['seed'])
        if config['hyperparameters']['mode'] == 'eval':
            train_model(data)
        elif config['hyperparameters']['mode'] == 'inference':
            for i, data_split in enumerate(data):
                if not os.path.exists(f"{config['paths']['VARFORMER_PREDICT_OUTPUT']}predictions_split_{i}.pkl"):
                    train_model(data_split, i)
                else:
                    print(f"Predictions for split {data['split_idx']} already exist. Skipping inference...")
        else:
            raise ValueError("Invalid mode in config. Please set 'mode' to either 'eval' or 'inference'.")
    else:
        raise ValueError("Config is None. Please provide a valid configuration dictionary by setting the --config "
                         "parameter.")


def logistic_regression(**modules):
    """
    Train a logistic regression model on the same data that would be used for the neural network model.
    Record results in wandb under the logistic-regression-1 group.

    Args:
        data: Dictionary containing configuration and dataset information
    """

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)
    config = modules.get('config', None)

    data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()

    config = data['config']
    hyperparameters = config['hyperparameters']

    run = wandb.init(
        project="drug-target-prediction",
        config=config["hyperparameters"],
        group="logistic-regression-1"
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


def drugnome_ai(**modules):
    # check if drugnome_ai_labels.txt already exists
    if not os.path.exists("../benchmark/data/drugnomeai/drugnome_ai_labels.txt"):
        drugnome_ai_labels = pd.read_csv("../benchmark/data/drugnomeai/gene_druggable_labels.csv")
        # drugnome_ai_labels = drugnome_ai_labels[drugnome_ai_labels['druggability_tier'] == 'Tier 1']

        gc = modules.get('gc', False)
        go = modules.get('go', False)
        pvc = modules.get('pvc', False)
        psc = modules.get('psc', False)
        config = modules.get('config', None)

        data = ModuleDataProcessor(gc, go, pvc, psc, config=config).process()
        test_data = list(data['test_labels'].keys())

        gene_to_hgnc = utils.utils.map_gene_names(test_data, 'ensg', 'symb')
        test_data_hgnc = [gene_to_hgnc[gene] for gene in test_data if gene in gene_to_hgnc]
        with open("../benchmark/data/drugnomeai/test_genes.txt", 'w') as f:
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


# legacy code
def kfold_student(ensemble=False, **modules):
    """
    Training a student model by distilling from a teacher model where the teacher model is used to provide pseudo-labels
    for the unlabeled data used by the student model.

    Firstly, we will run inference over the unlabelled data using the teacher model to define the pseudolabels. Then,
    we add the pseudolabels to the training data and train the student model on the combined dataset.
    """
    # TODO: Connect this to the teacher to learn end-to-end
    print("Training student model by distillation from teacher model...\n")

    gc = modules.get('gc', False)
    go = modules.get('go', False)
    pvc = modules.get('pvc', False)
    psc = modules.get('psc', False)

    modules = {
        "gc": gc,
        "go": go,
        "pvc": pvc,
        "psc": psc
    }

    print(f"Training teacher model with {' '.join([k for k, v in modules.items() if v])} modules...\n")

    data = ModuleDataProcessor(gc, go, pvc, psc).process()

    if not ensemble:
        for module, preprocessor in data.items():
            train_df = preprocessor.data
            num_features = preprocessor.num_features
            config = preprocessor.config

            labels = train_df.iloc[:, -1].values

            unlabelled_data = data[data.iloc[:, -1] == 0]
            U = np.where(labels == 0)[0]

            hyperparams = config['hyperparameters']

            val = None
            unlabelled_dl = normalise_data(unlabelled_data, val, hyperparams)
            unlabelled_tensor = unlabelled_dl.dataset.features

            teacher_model = BaseTargetIdentifier(config=hyperparams, num_features=num_features)
            raw_state_dict = torch.load(
                "checkpoints/distillation-3-teacher/swift-jazz-179-epoch=99-val_auroc=0.84-fold0.ckpt"
            )['state_dict']

            state_dict = {k.replace("model.", ""): v for k, v in raw_state_dict.items()}
            teacher_model.load_state_dict(state_dict)
            teacher_model.eval()

            logits, probas, bin_preds = teacher_model(unlabelled_tensor)

            delta = 0.25  # threshold for pseudo-labeling

            pseudo_labels = torch.full_like(probas, -1)

            pos_count = 0
            neg_count = 0
            for i, proba in enumerate(probas):
                if proba >= hyperparams['mlp']['threshold']:
                    pseudo_labels[i] = proba
                    pos_count += 1
                elif proba <= hyperparams['mlp']['threshold'] - delta:
                    pseudo_labels[i] = proba
                    neg_count += 1
            unlabelled_count = len(probas) - pos_count - neg_count

            # plot.plot_kde(pseudo_labels)

            data.iloc[U, -1] = pseudo_labels.detach().numpy()

            data = data[data.iloc[:, -1] != -1]

            kfold_train(data, num_features, hyperparams, model_type="student", modules=module)
    else:
        # TODO: Implement ensemble training
        pass
