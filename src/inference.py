import torch
import yaml
import os

import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from datetime import datetime
from models.lightning import MultiModalLightningTargetIdentifier
from preprocessing import ModelPreprocessorEval, ModelPreprocessorInference
from dataloader import ModuleDataProcessor
from pytorch_lightning import Trainer
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from adjustText import adjust_text
from scipy import stats
from collections import defaultdict


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

def run_inference_pipeline(config, ckpt_path, save_path):
    data = prepare_data(config)
    model = load_model(ckpt_path, data, config)
    # filter all genes to just the unlabeled genes
    filtered_data = None
    predictions = run_inference(model, filtered_data)
    # save predictions to .xlsx file
    predictions.to_excel(save_path)
    print("Predictions saved to {}".format(save_path))


def extract_base_variant_id(variant_id_with_consequence):
    # Split by underscore and take the first 4 parts (chr, pos, ref, alt)
    parts = variant_id_with_consequence.split('_')
    return '_'.join(parts[:4])


def create_pharmgkb_plots(pharmgkb_visualisation):
    """
    Create scatter plots from pharmgkb_visualisation data with non-overlapping labels.
    Only outlier points are labeled, and "No Evidence" points are shown in gray and transparent.

    Args:
        pharmgkb_visualisation: Dictionary with gene_rsid keys and metric values

    Returns:
        DataFrame of the data for further analysis
    """

    # Convert dictionary to DataFrame for easier plotting
    data = []
    for key, values in pharmgkb_visualisation.items():
        gene, rsid = key.split('_')
        row = {
            'gene': gene,
            'rsID': rsid,
            'Attention': values['Attention'],
            'am_pathogenicity': values['am_pathogenicity'],
            'AF': values['AF'],
            'Therapeutic_Evidence': values['Therapeutic Evidence']
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Separate evidence and no evidence data
    no_evidence_df = df[df['Therapeutic_Evidence'] == "No Evidence"]
    evidence_df = df[df['Therapeutic_Evidence'] != "No Evidence"]

    # Get unique evidence types for color palette (excluding "No Evidence")
    evidence_types = evidence_df['Therapeutic_Evidence'].unique()

    # Set up a categorical color palette with distinct colors
    if len(evidence_types) <= 10:
        # Use standard categorical palette for <= 10 categories
        color_palette = sns.color_palette("tab10", len(evidence_types))
    else:
        # For more categories, use a large color palette and cycle
        color_palette = sns.color_palette("husl", len(evidence_types))

    # Create color dictionary for evidence types
    color_dict = dict(zip(evidence_types, color_palette))

    # Add "No Evidence" to the color dictionary as gray
    all_evidence_types = list(evidence_types) + ["No Evidence"]
    color_dict["No Evidence"] = (0.7, 0.7, 0.7)  # Gray color

    # Function to identify outliers
    def get_outliers(df, x_col, y_col, threshold=1.5):
        """
        Identify outlier points based on Z-scores

        Args:
            df: DataFrame containing the data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            threshold: Z-score threshold to consider a point an outlier

        Returns:
            DataFrame containing only the outlier points
        """
        # Calculate Z-scores for each dimension
        x_z = np.abs(stats.zscore(df[x_col], nan_policy='omit'))
        y_z = np.abs(stats.zscore(df[y_col], nan_policy='omit'))

        # Find points that are outliers in either dimension
        outlier_indices = (x_z > threshold) | (y_z > threshold)

        # Return the outlier points
        return df[outlier_indices]

    # Set seaborn style with clean white background but keep grid lines
    sns.set_style("whitegrid")  # Clean white background with subtle grid lines

    # Set color palette to something more professional
    # Options: "deep", "muted", "bright", "pastel", "dark", "colorblind"
    sns.set_palette("colorblind")  # Good for accessibility
    # Or custom palette: sns.set_palette(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    # Configure matplotlib for publication quality
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (8, 6),  # Default figure size
        'figure.dpi': 300,  # High resolution for publications
        'savefig.dpi': 300,  # High resolution when saving
        'savefig.bbox': 'tight',  # Remove extra whitespace
        'savefig.pad_inches': 0.1,  # Small padding

        # Font settings
        'font.family': 'serif',  # Professional serif font
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,  # Base font size
        'axes.titlesize': 14,  # Title font size
        'axes.labelsize': 12,  # Axis label font size
        'xtick.labelsize': 10,  # X-tick label size
        'ytick.labelsize': 10,  # Y-tick label size
        'legend.fontsize': 10,  # Legend font size

        # Axes and layout
        'axes.linewidth': 1.2,  # Thicker axis lines
        'axes.spines.top': False,  # Remove top spine
        'axes.spines.right': False,  # Remove right spine
        'axes.grid': True,  # Enable grid lines
        'grid.color': 'lightgray',  # Light gray grid color
        'grid.linewidth': 0.8,  # Thin grid lines
        'grid.alpha': 0.7,  # Semi-transparent grid
        'axes.axisbelow': True,  # Grid below data if used
        'axes.edgecolor': 'black',  # Black axis edges
        'axes.labelcolor': 'black',  # Black labels

        # Ticks
        'xtick.direction': 'out',  # Ticks point outward
        'ytick.direction': 'out',
        'xtick.major.size': 6,  # Tick length
        'ytick.major.size': 6,
        'xtick.minor.size': 3,  # Minor tick length
        'ytick.minor.size': 3,
        'xtick.color': 'black',  # Black ticks
        'ytick.color': 'black',

        # Lines and markers
        'lines.linewidth': 2,  # Thicker lines
        'lines.markersize': 6,  # Marker size
        'patch.linewidth': 1,  # Patch edge width

        # Legend
        'legend.frameon': True,  # Legend frame
        'legend.framealpha': 0.9,  # Semi-transparent frame
        'legend.fancybox': False,  # Square legend box
        'legend.edgecolor': 'black',  # Black legend border
        'legend.facecolor': 'white',  # White legend background

        # Colors
        'text.color': 'black',  # Black text
        'axes.labelcolor': 'black',  # Black axis labels
        'xtick.color': 'black',  # Black tick labels
        'ytick.color': 'black',
    })

    def create_plot(x_data, y_data, xlabel, title, filename, logscale=False):
        plt.figure(figsize=(14, 10))  # Larger figure size for your specific plots

        # First plot "No Evidence" points in the background with lower alpha
        if not no_evidence_df.empty:
            plt.scatter(
                no_evidence_df[x_data],
                no_evidence_df['Attention'],
                label="No Evidence",
                color=color_dict["No Evidence"],
                alpha=0.1,  # More transparent
                edgecolor=None,
                s=60,
                zorder=1  # Ensure they're in the background
            )

        # Then plot points with evidence
        for evidence_type in evidence_types:
            subset = evidence_df[evidence_df['Therapeutic_Evidence'] == evidence_type]
            if not subset.empty:
                plt.scatter(
                    subset[x_data],
                    subset['Attention'],
                    label=evidence_type,
                    color=color_dict[evidence_type],
                    alpha=0.7,
                    edgecolor='black',
                    s=80,
                    zorder=2  # Ensure they're in the foreground
                )

        # Calculate and add horizontal mean line
        # Calculate mean of all Attention scores (including both evidence and no evidence)
        all_attention_scores = df['Attention'].dropna()
        mean_attention = all_attention_scores.mean()

        # Add horizontal mean line
        plt.axhline(y=mean_attention, color='red', linestyle='--', linewidth=2,
                    label=f'Mean Attention', alpha=0.8, zorder=3)

        # Find outliers only among points with evidence (not "No Evidence" points)
        outliers = get_outliers(evidence_df, x_data, 'Attention', threshold=1.8)

        # Add labels only for outlier points
        # texts = []
        # for i, row in outliers.iterrows():
        #     txt = plt.text(
        #         row[x_data] + 0.005,
        #         row['Attention'],
        #         row['rsID'],
        #         fontsize=9,
        #         alpha=0.9,
        #         fontweight='bold',
        #         zorder=4  # Ensure labels are on top
        #     )
        #     texts.append(txt)

        if logscale:
            plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Attention Score', fontsize=12)

        # Adjust legend based on number of categories
        if len(all_evidence_types) <= 6:
            plt.legend(loc='best', fontsize=10)
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)

        plt.grid(alpha=0.3)

        # Use adjustText to prevent label overlap if there are any labels
        # if texts:
        #     adjust_text(
        #         texts,
        #         arrowprops=None,
        #         expand_points=(1.8, 1.8),
        #         force_points=(0.8, 0.8)
        #     )

        # Remove top and right spines for cleaner look
        sns.despine()

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    # Create figure 1: Attention vs. Pathogenicity
    create_plot(
        x_data='am_pathogenicity',
        y_data='Attention',
        xlabel='AlphaMissense Pathogenicity Score',
        title='Attention vs. Pathogenicity by Therapeutic Evidence Type',
        filename='attention_vs_pathogenicity.png'
    )

    # Create figure 2: Attention vs. Allele Frequency
    create_plot(
        x_data='AF',
        y_data='Attention',
        xlabel='Allele Frequency (log scale)',
        title='Attention vs. Allele Frequency by Therapeutic Evidence Type',
        filename='attention_vs_allele_frequency.png',
        logscale=True
    )

    return df


def homogenise_pharmgkb_data():
    # check if merged data exists
    if not os.path.exists("../data/pharmgkb/combined_ann.pkl"):
        clin_vars = pd.read_csv("../data/pharmgkb/clinicalVariants.tsv", sep="\t")
        drug_ann = pd.read_csv("../data/pharmgkb/var_drug_ann.tsv", sep="\t")
        pheno_ann = pd.read_csv("../data/pharmgkb/var_pheno_ann.tsv", sep="\t")
        combined_ann = pd.concat([drug_ann, pheno_ann], ignore_index=True)
        non_overlap = clin_vars[~clin_vars['variant'].isin(combined_ann['Variant/Haplotypes'])]
        non_overlap = non_overlap[['variant', 'gene', 'type']]
        non_overlap = non_overlap.rename(
            columns={'variant': 'Variant/Haplotypes', 'gene': 'Gene', 'type': 'Phenotype Category'})
        combined_ann = pd.concat([combined_ann, non_overlap], ignore_index=True)
        with open("../data/pharmgkb/combined_ann.pkl", "wb") as f:
            pkl.dump(combined_ann, f)
        return combined_ann
    else:
        with open("../data/pharmgkb/combined_ann.pkl", "rb") as f:
            combined_ann = pkl.load(f)
        return combined_ann


def plot_attention_corrs(per_gene_dfs):
    attention_values = []
    pathogenicity_values = []

    for gene, data in per_gene_dfs.items():
        attention_values.extend(data['attn_weight'])
        pathogenicity_values.extend(data['am_pathogenicity'])

    # Create a DataFrame for the extracted values
    plot_data = pd.DataFrame({
        'Attention': attention_values,
        'am_pathogenicity': pathogenicity_values
    })

    # Plotting the data
    plt.figure(figsize=(8, 6))
    plt.scatter(plot_data['am_pathogenicity'], plot_data['Attention'], alpha=0.5)
    plt.title('Correlation between Attention and AM Pathogenicity')
    plt.xlabel('AM Pathogenicity')
    plt.ylabel('Attention')
    plt.grid(True)
    plt.show()
    plt.savefig(f"attention_vs_am_pathogenicity.pdf", dpi=300)


def plot_attention_vs_gwas(gwas_path, per_gene_dfs, variant_map, disease_code="E11"):
    """
    Create a Manhattan-style plot: Attention (x) vs. -log10(p-value) (y) for a specific disease.
    Annotates high-value hits with <gene>_<variant> labels.
    """
    BONFERRONI_CUTOFF = -np.log10(5e-8)
    SUGGESTIVE_CUTOFF = -np.log10(5e-5)

    ICD10_TO_NAME = {
        "E11": "Type 2 Diabetes",
        "I10": "Hypertension",
        "I20": "Angina",
        "I21": "Myocardial Infarction",
        "I24": "Acute IHD",
        "I25": "Chronic IHD",
        "I50": "Heart Failure",
        "I61": "Intracerebral Hemorrhage",
        "I63": "Cerebral Infarction",
        "I64": "Stroke (Unspecified)",
        "I65": "Arterial Occlusion",
        "I67": "Other CVD",
        "I69": "CVD Sequelae",
        "C50": "Breast Cancer",
        "E78": "Lipid Disorders",
        "E88": "Metabolic Disorders"
    }

    # Load GWAS data
    df = pd.read_csv(gwas_path, sep=' ')
    df = df[(df['ALLELE0'].str.len() == 1) & (df['ALLELE1'].str.len() == 1)]

    # Build variant ID and log_p
    df['base_variant_id'] = (
        'chr' + df['CHROM'].astype(str) + '_' + df['GENPOS'].astype(str) + '_' +
        df['ALLELE0'] + '_' + df['ALLELE1']
    )
    df = df[~df['base_variant_id'].duplicated(keep='first')]
    df['log_p'] = df['LOG10P']
    df['rsID'] = df['base_variant_id'].map(variant_map)

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['log_p'], bins=100)
    plt.yscale('log')
    plt.axvline(BONFERRONI_CUTOFF, color='red', label='Genome-wide significance (p<5e-8)')
    plt.axvline(SUGGESTIVE_CUTOFF, color='orange', label='Suggestive significance (p<5e-5)')
    plt.legend(loc='upper right')
    disease_name = ICD10_TO_NAME.get(disease_code, disease_code)
    plt.xlabel("-log10(P)")
    plt.ylabel("Count")
    plt.title(f"GWAS p-value distribution for {disease_name}")
    plt.tight_layout()
    plt.savefig(f"gwas_pval_hist_{disease_name.replace(' ', '_')}.png", dpi=300)
    plt.close()

    # Collect attention scores + gene lookup
    attention_collector = defaultdict(list)
    base_variant_to_gene = {}

    for gene, gene_df in per_gene_dfs.items():
        for full_id, row in gene_df.iterrows():
            base_id = '_'.join(full_id.split('_')[:4])
            if pd.notna(row['Attention']):
                attention_collector[base_id].append(row['Attention'])
                base_variant_to_gene[base_id] = gene  # assumes one gene per base_id

    variant_attention_map = {
        k: np.mean(v) for k, v in attention_collector.items()
    }

    df['Attention'] = df['base_variant_id'].map(variant_attention_map)
    df.dropna(subset=['Attention'], inplace=True)

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='Attention',
        y='log_p',
        alpha=0.6,
        edgecolor=None
    )
    plt.axhline(BONFERRONI_CUTOFF, color='red', linestyle='--', label='Genome-wide significance')
    plt.axhline(SUGGESTIVE_CUTOFF, color='orange', linestyle='--', label='Suggestive significance')
    plt.xscale('log')
    plt.xlabel("Attention Score")
    plt.ylabel("-log10(P-value)")
    plt.title(f"Prioritization vs GWAS Significance for {disease_name}")
    plt.legend(loc='upper right')
    plt.grid(True)

    # Annotate top hits with gene name + variant
    texts = []
    labeled_df = df[df['log_p'] > SUGGESTIVE_CUTOFF]
    for _, row in labeled_df.iterrows():
        base_id = row['base_variant_id']
        gene = base_variant_to_gene.get(base_id, "NA")
        label = f"{gene}_{base_id}"
        texts.append(plt.text(row['Attention'], row['log_p'], label, fontsize=8))

    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

    plt.tight_layout()
    plt.savefig(f"attention_vs_pval_{disease_name.replace(' ', '_')}.png", dpi=300)
    plt.close()

    return df


def varformer_prediction_analysis(output):
    config_path = 'cluster_config.yml'
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    with open(config['paths']['VAR_MAP'], "rb") as f:
        variant_map = pkl.load(f)

    combined_var_anns = homogenise_pharmgkb_data()

    am_df = pd.read_parquet(config['paths']['AM_PATH'])
    am_df.rename(columns={'variant_id': 'variant_ids'}, inplace=True)

    gh_df = pd.read_pickle(config['paths']['ALL_GH'])
    gh_df[['ref_aa', 'alt_aa']] = gh_df['Amino_acids'].str.split('/', expand=True)
    gh_df['protein_variant'] = gh_df['ref_aa'] + gh_df['Protein_position'].astype(str) + gh_df['alt_aa']
    gh_df['variant_ids'] = (gh_df['CHROM'] + '_' + gh_df['POS'].astype(str) + '_' +
                            gh_df['REF'] + '_' + gh_df['ALT'] + '_' + gh_df['protein_variant'])

    # Load pre-computed predictions from VARFORMER_PREDICT_OUTPUT directory
    batches = []
    prediction_dir = config['paths']['VARFORMER_PREDICT_OUTPUT']

    for file in os.listdir(prediction_dir):
        with open(os.path.join(prediction_dir, file), 'rb') as f:
            prediction_output = torch.load(f)
            batches.extend(prediction_output)

    varformer_output = {}
    for batch in batches:
        for key, value in batch.items():
            varformer_output[key] = value

    with open(config['paths']['GENE_VAR_LOC_MAP'], "rb") as f:
        gene_var_loc_map = pkl.load(f)

    pred_dict = {}
    for gene, variant_ids in gene_var_loc_map.items():
        if gene in varformer_output:
            gene_data = varformer_output[gene]
            attn_weights = gene_data['attn_weights']
            prediction = gene_data['prediction']
            for idx, variant_id in enumerate(variant_ids):
                pred_dict[variant_id] = {
                    'gene': gene,
                    'attn_weight': attn_weights[idx],
                    'prediction': prediction
                }

    am_df = am_df.drop_duplicates(subset='variant_ids')
    attn_weights_map = {variant_id: data['attn_weight'] for variant_id, data in pred_dict.items()}
    predictions_map = {variant_id: data['prediction'] for variant_id, data in pred_dict.items()}
    am_df['attn_weight'] = am_df['variant_ids'].map(attn_weights_map)
    am_df['prediction'] = am_df['variant_ids'].map(predictions_map)

    # drop everywhere where predictions is NaN
    am_df = am_df.dropna(subset=['prediction'])

    # get a version of am_df with only a single row per gene and the variant column is dropped
    am_df_gene = am_df.drop_duplicates(subset='Gene', keep='first')
    # drop variant_ids
    am_df_gene = am_df_gene.drop(columns=['variant_ids'])
    am_df_gene = am_df_gene.drop(columns=['attn_weight'])

    # sort the dataframe by prediction in descending order
    am_df = am_df.sort_values(by='prediction', ascending=False)
    am_df = am_df[am_df['prediction'] >= 0.9]

    # Create variant ID to rsID mapping dictionary once
    variant_to_rsid = dict(zip(variant_map.keys(), variant_map.values()))

    # Create a dictionary of variant_ids to therapeutic evidence
    rsid_to_evidence = {}
    if 'Variant/Haplotypes' in combined_var_anns.columns and 'Phenotype Category' in combined_var_anns.columns:
        for _, row in combined_var_anns.iterrows():
            rsid = row['Variant/Haplotypes']
            evidence = row['Phenotype Category']
            rsid_to_evidence[rsid] = evidence

    # Filter AM and GH dataframes once (outside the loop) to avoid repeated filtering
    am_indexed = am_df.set_index('variant_ids')
    am_indexed = am_indexed[~am_indexed.index.duplicated(keep='first')]
    gh_indexed = gh_df.set_index('variant_ids')
    gh_indexed = gh_indexed[~gh_indexed.index.duplicated(keep='first')]

    per_gene_dfs = {}
    pharmgkb_visualisation = {}
    for gene in tqdm(am_indexed['Gene'].unique()):
        try:
            gene_df = am_indexed[am_indexed['Gene'] == gene]

            gene_df['base_variant_id'] = [extract_base_variant_id(v) for v in gene_df.index]
            gene_df['rsID'] = gene_df['base_variant_id'].map(variant_to_rsid)

            gene_df['Therapeutic Evidence'] = 'No Evidence'
            rsids_in_df = set(gene_df['rsID'].dropna())
            for rsid in rsids_in_df:
                if rsid in rsid_to_evidence:
                    print(f"Found rsID {rsid} in evidence mapping!")
                    gene_df.loc[gene_df['rsID'] == rsid, 'Therapeutic Evidence'] = rsid_to_evidence[rsid]
                    pharmgkb_visualisation[f"{gene}_{rsid}"] = {
                        'Attention': gene_df.loc[gene_df['rsID'] == rsid, 'attn_weight'].values[0],
                        'am_pathogenicity': gene_df.loc[gene_df['rsID'] == rsid, 'am_pathogenicity'].values[0],
                        'AF': gene_df.loc[gene_df['rsID'] == rsid, 'AF'].values[0],
                        'Therapeutic Evidence': rsid_to_evidence[rsid]
                    }
                else:
                    pharmgkb_visualisation[f"{gene}_{rsid}"] = {
                        'Attention': gene_df.loc[gene_df['rsID'] == rsid, 'attn_weight'].values[0],
                        'am_pathogenicity': gene_df.loc[gene_df['rsID'] == rsid, 'am_pathogenicity'].values[0],
                        'AF': gene_df.loc[gene_df['rsID'] == rsid, 'AF'].values[0],
                        'Therapeutic Evidence': "No Evidence"
                    }

            per_gene_dfs[gene] = gene_df

        except Exception as e:
            print(f"Error processing gene {gene}: {e}")

    # Generate pharmgkb plots
    # pharmgkb_df = create_pharmgkb_plots(pharmgkb_visualisation)
    #
    # # Generate attention correlation plots
    # # plot_attention_corrs(per_gene_dfs)
    #
    # print('Completed generating pharmgkb plots')

    all_variants = []

    for gene, df in per_gene_dfs.items():
        variants = df.index.tolist()
        all_variants.extend(variants)

    all_variants = list(set(all_variants))

    output_path = "../data/output/all_variants_eval_set_unlabeled.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(all_variants))

    print(f"Saved {len(all_variants)} unique variants to {output_path}")
