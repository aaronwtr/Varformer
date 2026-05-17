"""Inference pipeline for the Varformer model.

Moved from src/training.py (run_inference) in Phase 5.
Phase 7: Added predict_subset() for SDK predict() method.
"""
import os
import torch
import pickle as pkl

import pandas as pd

from pytorch_lightning import Trainer

from varformer.training.lightning_module import VarformerLightningModule
from varformer.data.loaders import ModelPreprocessorInference


def run_inference(data):
    """Run inference using pre-trained checkpoint on all data splits"""
    torch.set_float32_matmul_precision('medium')

    config = data[0]['config']
    print(f"Consolidating data from {len(data)} splits for unified inference...")

    # --- Consolidate all splits into one dataset ---
    consolidated_data = {modality: [] for modality in ["gc", "go"]}
    consolidated_pvc = {}
    consolidated_labels = {}
    all_genes = []

    for i, split_data in enumerate(data):
        print(f"Consolidating split {i + 1}/{len(data)}...")

        # Add gc/go DataFrames from both train + test
        for modality in ["gc", "go"]:
            consolidated_data[modality].append(split_data["test_data"][modality])

        # Merge pvc dicts
        consolidated_pvc.update(split_data["test_data"]["pvc"])

        # Merge labels
        consolidated_labels.update(split_data["test_labels"])

        # Collect all genes
        all_genes.extend(split_data["test_genes"])

    # Concatenate DataFrames
    for modality in ["gc", "go"]:
        consolidated_data[modality] = pd.concat(
            consolidated_data[modality], ignore_index=False
        )

    print(f"Total samples before filtering: {len(all_genes)}")

    # --- Create one loader with all unlabeled samples ---
    unlabeled_loader, num_samples = ModelPreprocessorInference.create_unlabeled_loader(
        config=config,
        consolidated_data=consolidated_data,
        pvc_data=consolidated_pvc,
        gene_names=list(consolidated_labels.keys()),
        torch_dtype=config['hyperparameters']['precision'],
    )

    # --- Create test loaders for approved genes ---
    test_loaders = ModelPreprocessorInference.create_test_loaders(
        config=config,
        consolidated_data=consolidated_data,
        pvc_data=consolidated_pvc,
        torch_dtype=config['hyperparameters']['precision']
    )

    if unlabeled_loader is None:
        print("No unlabeled data found. Exiting inference.")
        return

    # Get gene names and count
    gene_names = next(iter(unlabeled_loader.datasets.values())).gene_names
    print(f"Created unified dataloader with {len(gene_names)} unlabeled genes")

    # Load missense map and calculate dimensions using first split
    with open(config['paths']['MISSENSE_MAP'], "rb") as f:
        missense_map = pkl.load(f)

    num_mutations = len(missense_map)

    # Calculate total genes across all splits
    total_train_genes = sum(len(split_data['train_genes']) for split_data in data)
    total_test_genes = sum(len(split_data['test_genes']) for split_data in data)
    num_genes = total_train_genes + total_test_genes

    # Use first split to get feature dimensions
    first_split = data[0]
    num_features_gc = first_split['train']['gc'].shape[1] - 1 if 'target' in first_split['train']['gc'].columns else \
        first_split['train']['gc'].shape[1]
    num_features_go = first_split['train']['go'].shape[1]

    # Load checkpoint
    ckpt_folder = f"{config['paths']['CKPT_PATH']}{config['hyperparameters']['population']}"
    ckpt_names = list(os.listdir(ckpt_folder))
    for ckpt_name in ckpt_names:
        seed_raw = ckpt_name.split("-")[0]
        if seed_raw[4:].isdigit():
            seed = int(seed_raw[4:])
        else:
            continue
        ckpt_path = f"{ckpt_folder}/{ckpt_name}"

        print(f"Loading model from checkpoint: {ckpt_path}")

        # Load pre-trained model
        model = VarformerLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            config=config,
            num_features_gc=num_features_gc,
            num_features_go=num_features_go,
            num_mutations=num_mutations,
            max_seq_len=config['hyperparameters']['max_seq_len'],
            num_genes=num_genes,
            num_samples_per_class=None,  # Not needed for inference
            class_prior=None  # Not needed for inference
        )

        # Run inference once on the unified loader
        trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
        print("Running unified inference...")
        prediction_results = trainer.predict(model=model, dataloaders=unlabeled_loader)

        # Save results
        output_path_folder = (f"{config['paths']['VARFORMER_PREDICT_OUTPUT']}{config['hyperparameters']['population']}"
                              f"/unlabeled_predictions")
        output_path = f"{output_path_folder}/unlabeled_predictions_seed_{seed}.pkl"
        os.makedirs(output_path_folder, exist_ok=True)
        torch.save(prediction_results, output_path)
        print(f"Unified predictions saved to {output_path}")
        print(f"Total predictions: {len(prediction_results)}")

        # Run inference on test genes (approved targets)
        if len(test_loaders) > 0:
            print("Running inference on test genes (approved targets)...")
            all_test_results = []

            for test_name, test_loader in test_loaders.items():
                print(f"  Processing {test_name}...")
                test_results = trainer.predict(model=model, dataloaders=test_loader)
                all_test_results.extend(test_results)

            # Save approved predictions
            approved_output_folder = f"{config['paths']['VARFORMER_PREDICT_OUTPUT']}{config['hyperparameters']['population']}/approved_predictions/"
            os.makedirs(approved_output_folder, exist_ok=True)
            approved_output_path = f"{approved_output_folder}approved_predictions_seed_{seed}.pkl"
            torch.save(all_test_results, approved_output_path)
            print(f"Approved predictions saved to {approved_output_path}")


def predict_subset(model, genes, return_attention=False):
    """Run inference for the SDK's predict() method.

    Uses the LightningModule and config cached on ``model`` by
    ``Varformer._build_and_load`` and mirrors the data pipeline used in
    ``benchmark/generate_reference.py:generate_for_population`` so that outputs
    are bit-exact with the benchmark reference.

    Args:
        model:            A ``Varformer`` nn.Module instance with ``_lightning_module``,
                          ``_config``, and ``_population`` attributes set by
                          ``Varformer._build_and_load``.
        genes:            List of Ensembl gene IDs to return predictions for.
        return_attention: If False, strip ``attn_weights`` from returned payloads.

    Returns:
        dict mapping gene_id -> {"prediction", "classification", "z_var"[, "attn_weights"]}
    """
    # Intentionally do NOT call torch.set_float32_matmul_precision('medium') here:
    # that's a GLOBAL setting that downgrades matmul to TF32 on Ampere, causing ~1e-3
    # drift vs the reference predictions (which were captured with default 'highest').
    # Training paths may set 'medium' for throughput, but inference must preserve precision.

    lm = model._lightning_module
    # Reuse test_loaders cached at model-load time; they were built from the same data
    # pipeline + config that produced the benchmark reference predictions.
    test_loaders = model._test_loaders

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

    gene_set = set(genes)
    results: dict = {}
    for loader_name, loader in test_loaders.items():
        batch_results = trainer.predict(model=lm, dataloaders=loader)
        for batch in batch_results:
            for gid, payload in batch.items():
                if gid in gene_set:
                    results[gid] = payload

    # Strip attn_weights if not requested.
    if not return_attention:
        for payload in results.values():
            payload.pop("attn_weights", None)

    return results
