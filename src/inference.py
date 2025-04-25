import torch
import yaml

import pickle as pkl
import pandas as pd

from collections import defaultdict
from datetime import datetime
from models.lightning import MultiModalLightningTargetIdentifier
from preprocessing import ModelPreprocessor
from dataloader import ModuleDataProcessor
from pytorch_lightning import Trainer


def load_model(checkpoint_path, data, config):
    with open(config['paths']['MISSENSE_MAP'], "rb") as f:
        missense_map = pkl.load(f)

    num_mutations = len(missense_map)

    num_genes = len(data['genes']) + len(data['test_genes'])

    test_data = {k: v for d in data["test_data"].values() for k, v in d.items()}

    combined_data = {
        modality: pd.concat([data['train'][modality], test_data[modality]], ignore_index=True)
        if isinstance(data['train'][modality], pd.DataFrame)
        else {**data['train'][modality], **test_data[modality]}
        for modality in data['train']
    }

    num_features_gc = combined_data['gc'].shape[1]
    num_features_go = combined_data['go'].shape[1]

    model = MultiModalLightningTargetIdentifier.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        num_features_gc=num_features_gc,
        num_features_go=num_features_go,
        num_mutations=num_mutations,
        max_seq_len=config['hyperparameters']['max_seq_len'],
        num_genes=num_genes,
        num_samples_per_class=None,  # Not needed for inference
        class_prior=None  # Not needed for inference
    )
    return model, combined_data


def prepare_data(config):
    data_processor = ModuleDataProcessor(
        gc=True, go=True, pvc=True, psc=False, config=config
    )
    data = data_processor.process()

    return data


def run_inference(model, test_data, batch_size=32):
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    predictions = trainer.predict(model, dataloaders=test_data)
    return predictions


def run_inference_pipeline(checkpoint, output):
    config_path = 'cluster_config.yml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    data = prepare_data(config)

    preprocessor = ModelPreprocessor(config, data)
    _, _, _, test_combined, _, _ = preprocessor.model_init()

    model, data = load_model(checkpoint, data, config)

    with open(config['paths']['GENE_VAR_MAP'], "rb") as f:
        gene_var_map = pkl.load(f)

    # TODO: All of the below needs to change. We need to train N models in order to be able to predict all unlabeled
    #  genes
    check = True
    if check:
        # load the predictions from the ../data/output folder
        with open('../data/output/predictions_20250415_213835.pkl', 'rb') as f:
            batches = torch.load(f)
    else:
        batches = run_inference(model, test_combined)
        # Save predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"{output}/predictions_{timestamp}.pkl"
        torch.save(batches, output)
        print(f"Predictions saved to {output}")

    predictions = {}
    for i, batch in enumerate(batches):
        predictions[f"testset_{i}"] = (batch[0][1], batch[0][3])

    gene_names_per_modality = {}
    for key, test_loader in test_combined.items():
        gene_names = list(test_loader.datasets['gc'].data.keys())
        gene_names_per_modality[key] = gene_names

    sums = defaultdict(float)
    counts = defaultdict(int)

    for gene_name_list, preds in zip(gene_names_per_modality.values(), predictions.values()):
        pred_tensor = preds[0]
        for i, gene_name in enumerate(gene_name_list):
            sums[gene_name] += pred_tensor[i].item()
            counts[gene_name] += 1

    preds_by_gene = {gene: sums[gene] / counts[gene] for gene in sums}
    pred_df = pd.DataFrame(preds_by_gene.items(), columns=['Gene', 'Prediction'])

    sums = defaultdict(lambda: torch.zeros(1024))
    counts = defaultdict(int)

    for gene_name_list, preds in zip(gene_names_per_modality.values(), predictions.values()):
        attn = preds[1]
        for gene, vec in zip(gene_name_list, attn):
            sums[gene] += vec
            counts[gene] += 1

    attn_by_gene = {g: sums[g] / counts[g] for g in sums}
    attn_df = pd.DataFrame(attn_by_gene.items(), columns=['Gene', 'Attention'])
    attn_df['Attention'] = attn_df['Attention'].apply(lambda x: x.tolist() if torch.is_tensor(x) else x)
    _attn_df = pd.DataFrame(attn_df['Attention'].to_list(), columns=[f'Variant_{i}' for i in range(1024)])
    attn_df = pd.concat([attn_df['Gene'], _attn_df], axis=1)

    pred_df_top = pred_df.sort_values(by='Prediction', ascending=False).head(5)
    attn_df_top = attn_df[attn_df['Gene'].isin(pred_df_top['Gene'].values)]

    attn_df_top = attn_df_top.set_index('Gene')
    per_gene_dfs = {}

    for gene in attn_df_top.index:
        variant_ids = gene_var_map[gene]
        variant_colnames = [f'Variant_{i}' for i in range(len(variant_ids))]

        # Get row, select relevant columns
        raw_attention_values = attn_df_top.loc[gene, variant_colnames]

        # Rename the index to actual variant IDs
        renamed_series = raw_attention_values.rename(dict(zip(variant_colnames, variant_ids)))

        # Convert to single-column DataFrame
        gene_df = renamed_series.to_frame(name='Attention')

        # Extract the pathogenicity × allele frequency from PVC data (first column of the tensor)
        pathogenicity_scores = None
        for test_set_data in preprocessor.test_data.values():
            pvc_dict = test_set_data['pvc']
            if gene in pvc_dict:
                pvc_tensor = pvc_dict[gene]  # Shape: (N_g, 4)
                pathogenicity_scores = pvc_tensor[:, 0].tolist()
                break  # Stop once we find the gene

        if pathogenicity_scores is None:
            raise ValueError(f"Gene {gene} not found in any test set PVC data.")

        # Add to the DataFrame assuming order matches variant_ids
        gene_df['AF Weighted Pathogenicity'] = pd.Series(pathogenicity_scores, index=variant_ids)

        # Store without gene column — key is the gene name
        per_gene_dfs[gene] = gene_df


# TODO: add pathogenicity score and allele frequency seperately
#  - load AM data and get the pathogenicity scores and AFs for the rsids
print('break')
