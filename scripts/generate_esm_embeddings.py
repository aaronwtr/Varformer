"""Offline data-prep script: generate raw and Varformer gene embeddings.

Moved from src/generate_embeddings.py (Phase 6 refactor).
Run from the repo root with the package installed (or PYTHONPATH=src:.).
"""
import argparse
import os
import pickle as pkl
import re

import numpy as np  # noqa: F401 — used transitively
import pandas as pd  # noqa: F401 — used transitively
import torch
import yaml

from varformer.data.datasets import MultiModalData
from varformer.data.samplers import MultiModalDataLoader
from varformer.data.pipeline import ModuleDataProcessor
from varformer.data.loaders import ModelPreprocessorEval
from varformer.data.features.variants import extract_pvc_features

# MultiModalLightningTargetIdentifier still lives in src/ until Phase 8
from models.lightning import MultiModalLightningTargetIdentifier


def get_varformer_embeddings(
    model: MultiModalLightningTargetIdentifier,
    dataloader: MultiModalDataLoader,
):
    """Extract gene-level embeddings from the Varformer part of the model."""
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    embeddings = {}
    gene_labels = {}
    dtype = torch.float32  # Always float32 due to PyTorch mixed precision issues in eval mode

    with torch.no_grad():
        for batch in dataloader:
            processed_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    processed_batch[key] = value.to(model.device, dtype=dtype)

                elif key in ['gc', 'go'] and isinstance(value, (list, tuple)):
                    processed_data = []
                    for item in value:
                        if isinstance(item, torch.Tensor):
                            processed_data.append(item.to(model.device, dtype=dtype))
                        else:
                            processed_data.append(item)
                    processed_batch[key] = tuple(processed_data)

                elif isinstance(value, dict) and key == 'pvc':
                    processed_pvc = {}
                    for pvc_key, pvc_value in value.items():
                        if isinstance(pvc_value, torch.Tensor):
                            processed_pvc[pvc_key] = pvc_value.to(model.device, dtype=dtype)
                        else:
                            processed_pvc[pvc_key] = pvc_value
                    processed_batch[key] = processed_pvc
                else:
                    processed_batch[key] = value

            gene_names = (
                processed_batch['pvc']['gene_name']
                if 'gene_name' in processed_batch['pvc']
                else None
            )
            mask = processed_batch['pvc']['mask']

            if model.config['return_attn']:
                _, _, _, z_var, _ = model.model(processed_batch, mask)
            else:
                _, _, _, z_var = model.model(processed_batch, mask)

            batch_labels = processed_batch['pvc']['labels']

            if gene_names is not None:
                for i, gene_name in enumerate(gene_names):
                    embeddings[gene_name] = z_var[i].cpu().numpy()
                    if isinstance(batch_labels, torch.Tensor):
                        gene_labels[gene_name] = batch_labels[i].cpu().item()
                    else:
                        gene_labels[gene_name] = batch_labels[i]
            else:
                for i in range(z_var.shape[0]):
                    gene_key = f"gene_{i}"
                    embeddings[gene_key] = z_var[i].cpu().numpy()
                    if isinstance(batch_labels, torch.Tensor):
                        gene_labels[gene_key] = batch_labels[i].cpu().item()
                    else:
                        gene_labels[gene_key] = batch_labels[i]

    return embeddings, gene_labels


def get_raw_variant_embeddings(pvc_data, genes, max_variants):
    """Generate embeddings from raw variant data using statistical feature extraction."""
    embeddings = {}
    for gene in genes:
        embeddings[gene] = extract_pvc_features(gene, pvc_data, max_variants)
    return embeddings


def combine_and_filter_loaders(train_loader, val_loader, test_genes_to_exclude, all_labels):
    """Combine train and val loaders, filtering out test genes and FDA-approved genes."""
    train_datasets = train_loader.datasets
    val_datasets = val_loader.datasets

    train_genes = set(train_datasets['gc'].gene_names)
    val_genes = set(val_datasets['gc'].gene_names)
    all_combined_genes = train_genes.union(val_genes)

    test_genes_set = set(test_genes_to_exclude)
    fda_approved_genes = {gene for gene, label in all_labels.items() if label == 1}

    genes_to_exclude = test_genes_set.union(fda_approved_genes)
    unlabeled_genes = [gene for gene in all_combined_genes if gene not in genes_to_exclude]

    print(f"Total genes in train+val: {len(all_combined_genes)}")
    print(f"Genes excluded (test): {len(test_genes_set)}")
    print(f"Genes excluded (FDA approved): {len(fda_approved_genes)}")
    print(f"Final unlabeled genes: {len(unlabeled_genes)}")

    if not unlabeled_genes:
        print("No unlabeled genes found after filtering")
        return None

    filtered_datasets = {}

    train_gc_data = {
        gene: train_datasets['gc'].data[gene]
        for gene in train_datasets['gc'].gene_names
        if gene in unlabeled_genes
    }
    val_gc_data = {
        gene: val_datasets['gc'].data[gene]
        for gene in val_datasets['gc'].gene_names
        if gene in unlabeled_genes
    }
    combined_gc_data = {**train_gc_data, **val_gc_data}

    train_go_data = {
        gene: train_datasets['go'].data[gene]
        for gene in train_datasets['go'].gene_names
        if gene in unlabeled_genes
    }
    val_go_data = {
        gene: val_datasets['go'].data[gene]
        for gene in val_datasets['go'].gene_names
        if gene in unlabeled_genes
    }
    combined_go_data = {**train_go_data, **val_go_data}

    train_pvc_data = {
        gene: train_datasets['pvc'].variant_data['data'][gene]
        for gene in train_datasets['pvc'].gene_names
        if gene in unlabeled_genes
    }
    val_pvc_data = {
        gene: val_datasets['pvc'].variant_data['data'][gene]
        for gene in val_datasets['pvc'].gene_names
        if gene in unlabeled_genes
    }
    combined_pvc_data = {**train_pvc_data, **val_pvc_data}

    combined_labels = {gene: 0 for gene in unlabeled_genes}
    torch_dtype = train_datasets['gc'].torch_dtype

    filtered_datasets['gc'] = MultiModalData(
        data=combined_gc_data,
        labels=combined_labels,
        gene_names=unlabeled_genes,
        dtype=torch_dtype,
    )
    filtered_datasets['go'] = MultiModalData(
        data=combined_go_data,
        labels=combined_labels,
        gene_names=unlabeled_genes,
        dtype=torch_dtype,
    )
    filtered_datasets['pvc'] = MultiModalData(
        data=None,
        labels=None,
        gene_names=unlabeled_genes,
        dtype=torch_dtype,
        variant_data={'data': combined_pvc_data, 'labels': combined_labels},
        max_variants=train_datasets['pvc'].max_variants,
    )

    batch_size = min(32, len(unlabeled_genes))
    filtered_loader = MultiModalDataLoader(
        datasets=filtered_datasets,
        batch_size=batch_size,
        shuffle=False,
    )
    return filtered_loader


def generate_embeddings_for_population(population: str, config: dict, checkpoint_path: str):
    """Generate and save raw and Varformer embeddings for all genes (test + unlabeled)."""
    print(f"Processing population: {population}")
    config['hyperparameters']['population'] = population
    config['hyperparameters']['mode'] = 'eval'

    data_processor = ModuleDataProcessor(gc=True, go=True, pvc=True, psc=False, config=config)
    data = data_processor.process()

    test_data = data['test_data']
    all_test_genes_list = data['test_genes']
    all_labels = data['labels']

    print(f"Test genes: {len(all_test_genes_list)}")

    output_dir = f"{config['paths']['DATA_DIR']}/output/{population}"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating raw variant embeddings for test sets...")
    max_variants = config['hyperparameters'].get('max_seq_len', 100)
    for test_set_name, test_set_data in test_data.items():
        raw_pvc_data = {
            gene: variants
            for gene, variants in test_set_data['pvc'].items()
            if gene != 'labels'
        }
        genes_in_set = list(raw_pvc_data.keys())
        raw_embeddings = get_raw_variant_embeddings(raw_pvc_data, genes_in_set, max_variants)
        save_path = f"{output_dir}/raw_variant_embeddings_{test_set_name}.pkl"
        with open(save_path, 'wb') as f:
            pkl.dump(raw_embeddings, f)
        print(f"Saved raw variant embeddings for {population} - {test_set_name} to {save_path}")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found. Skipping Varformer embeddings for {population}.")
        return

    print("Loading model and generating Varformer embeddings...")
    model = MultiModalLightningTargetIdentifier.load_from_checkpoint(
        checkpoint_path, config=config
    )

    preprocessor = ModelPreprocessorEval(config, data)
    _, train_loader, val_loader, test_loaders, _, _ = preprocessor.model_init()

    all_test_labels_dict = {}
    for test_set_name, dataloader in test_loaders.items():
        varformer_embeddings, labels = get_varformer_embeddings(model, dataloader)
        all_test_labels_dict[test_set_name] = labels
        save_path = f"{output_dir}/varformer_embeddings_{test_set_name}.pkl"
        with open(save_path, 'wb') as f:
            pkl.dump(varformer_embeddings, f)
        print(f"Saved Varformer embeddings for {population} - {test_set_name} to {save_path}")

    print("Generating embeddings for unlabeled genes from train+val data...")
    combined_unlabeled_loader = combine_and_filter_loaders(
        train_loader, val_loader, all_test_genes_list, all_labels
    )

    if combined_unlabeled_loader is not None:
        unlabeled_genes = combined_unlabeled_loader.datasets['gc'].gene_names
        unlabeled_pvc_data = combined_unlabeled_loader.datasets['pvc'].variant_data['data']

        unlabeled_raw_embeddings = get_raw_variant_embeddings(
            unlabeled_pvc_data, unlabeled_genes, max_variants
        )
        save_path = f"{output_dir}/raw_variant_embeddings_unlabeled.pkl"
        with open(save_path, 'wb') as f:
            pkl.dump(unlabeled_raw_embeddings, f)
        print(
            f"Saved raw variant embeddings for {population} - unlabeled "
            f"({len(unlabeled_raw_embeddings)} genes) to {save_path}"
        )

        unlabeled_varformer_embeddings, _ = get_varformer_embeddings(
            model, combined_unlabeled_loader
        )
        save_path = f"{output_dir}/varformer_embeddings_unlabeled.pkl"
        with open(save_path, 'wb') as f:
            pkl.dump(unlabeled_varformer_embeddings, f)
        print(
            f"Saved Varformer embeddings for {population} - unlabeled "
            f"({len(unlabeled_varformer_embeddings)} genes) to {save_path}"
        )

    if not os.path.exists(f"{output_dir}/test_labels.pkl"):
        with open(f"{output_dir}/test_labels.pkl", 'wb') as f:
            pkl.dump(all_test_labels_dict, f)


def main():
    """Generate embeddings for all populations."""
    parser = argparse.ArgumentParser(
        description="Generate variant embeddings from raw and Varformer-processed variant data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cluster_config.yml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    populations = ['amr', 'nfe', 'afr']
    pattern = re.compile(r"val_spearman=([0-9]*\.?[0-9]+)")

    for pop in populations:
        ckpt_folder = f"/data/scratch/bty174/genomic-drug-targeting/src/checkpoints/{pop}"
        ckpt_files = [f for f in os.listdir(ckpt_folder) if f.endswith(".ckpt")]

        def get_spearman(fname: str) -> float:
            m = pattern.search(fname)
            return float(m.group(1)) if m else float("-inf")

        ckpt_file = max(ckpt_files, key=get_spearman) if ckpt_files else None
        checkpoint_path = os.path.join(ckpt_folder, ckpt_file)
        generate_embeddings_for_population(pop, config, checkpoint_path)


if __name__ == "__main__":
    main()
