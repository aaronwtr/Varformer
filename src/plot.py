import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import pearsonr
from umap import UMAP

def tractability_plot(tractability_scores, fda_labels, plottype=None, fda=True):
    if plottype == None:
        raise ValueError('Please specify type of plot. (SM or AB)')
    colors = ['red' if x == 1 else 'blue' for x in fda_labels]
    tractability_scores['color'] = colors
    tractability_scores['fda'] = fda_labels
    tractability_scores = tractability_scores.sort_values(by='tractability_score', ascending=False)
    tract_score = tractability_scores['tractability_score'].tolist()
    fda_labels = tractability_scores['fda'].tolist()
    colors = tractability_scores['color'].tolist()
    if fda:
        tract_scores_fda = [x for x, y in zip(tract_score, fda_labels) if y == 1]
        plt.scatter(range(len(tract_scores_fda)), tract_scores_fda, s=5, c='red')
    else:
        plt.scatter(range(len(tract_score)), tract_score, c=tractability_scores['color'], cmap='RdYlBu',
                    s=tractability_scores['color'].map({'red': 10, 'blue': 1}))

    plt.xlabel('Gene Index')
    plt.ylabel('Tractability Score')
    plt.ylim(0, max(tract_score) + 0.3)
    if plottype == 'SM':
        plt.title('Small Molecule Tractability Scores Colored by FDA Approval Status')
        plt.savefig('plots/scatter_fda_labels_sm.pdf')
    else:
        plt.title('Antibody Tractability Scores Colored by FDA Approval Status')
        plt.savefig('plots/scatter_fda_labels_ab.pdf')

    plt.xticks([])

    plt.show()


def variant_sparsity_barplot(variant_df, save=False):
    """
    Plots a bar chart of the number of variants with a given number of NaN values.
    :param variant_df: pandas dataframe of variants
    :return: bar chart of variant sparsity
    """
    sift_counts = variant_df['sift'].value_counts()
    polyphen_counts = variant_df['polyphen'].value_counts()

    # Count the number of NaN values in 'sift' and 'polyphen' columns
    sift_nan_count = variant_df['sift'].isna().sum()
    polyphen_nan_count = variant_df['polyphen'].isna().sum()

    names = ['SIFT', 'PolyPhen']
    print(sift_counts.sum(), polyphen_counts.sum())
    print(sift_nan_count, polyphen_nan_count)
    counts = {
        'Total': np.array([sift_counts.sum(), polyphen_counts.sum()]),
        'NaN': np.array([sift_nan_count, polyphen_nan_count])
    }
    width = 0.4

    fig, ax = plt.subplots(figsize=(6, 8))
    bottom = np.zeros(2)

    for category, count in counts.items():
        p = ax.bar(names, count, width, align="center", label=category)
        bottom += count

    ax.legend(loc='upper right', bbox_to_anchor=(5.5, 7.5))

    ax.set_xlabel('Pathogenicity')
    ax.set_ylabel('Variant count')
    ax.set_title('Sparsity of pathogenicity data from SIFT and PolyPhen')

    ax.legend()
    plt.show()

    if save:
        fig.savefig('plots/variant_sparsity_bar.pdf')
        fig.savefig('plots/variant_sparsity_bar.png')


def pathogenicity_correlation_plot(df, save):
    corr = df['sift'].corr(df['polyphen'])

    plt.scatter(df['sift'], df['polyphen'], alpha=0.5, s=2, c='blue')
    plt.xlabel('SIFT')
    plt.ylabel('PolyPhen')
    plt.title('Correlation between pathogenicity scores of SIFT and PolyPhen')
    plt.text(df['sift'].max() + 0.04, df['polyphen'].max() + 0.04, f'Correlation: {corr:.2f}', ha='right', va='top')

    if save:
        plt.savefig('plots/sift_pp_pathogenicity_correlation.pdf')
        plt.savefig('plots/sift_pp_pathogenicity_correlation.png')
        plt.show()


def varipred_kde_plot(df):
    pathogenic_df = df[df['ClinSigSimple'] == 1]
    benign_df = df[df['ClinSigSimple'] == 0]

    pathogenic_probs = pathogenic_df['vp_probability'].tolist()
    benign_probs = benign_df['vp_probability'].tolist()
    # pathogenic_probs = pathogenic_probs[1:]
    # probs = [x for x in probs if str(x) != 'probability']
    # probs = [float(i) for i in probs]
    # probs = [x for x in probs if str(x) != 'nan']

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.kdeplot(pathogenic_probs, label='Pathogenic', fill=True)
    sns.kdeplot(benign_probs, label='Benign', fill=True)

    intersection_point = 0.018

    # Plot vertical dotted line
    plt.axvline(x=intersection_point, color='black', linestyle='--')

    # Annotate the intersection value on the x-axis
    plt.annotate(f'{intersection_point:.3f}',
                 xy=(intersection_point + 0.005, 0.2),
                 xytext=(intersection_point, -0.02),
                 textcoords='offset points',
                 fontsize=10, color='black')

    plt.xlabel('Pathogenicity Probability')
    plt.ylabel('Density')
    plt.xlim(0, 0.5)
    plt.title('Pathogenicity Probability Distribution')
    plt.legend(loc='upper right')
    plt.savefig('../plots/varipred_kde_sep_finetuned.pdf')
    plt.savefig('../plots/varipred_kde_sep_finetuned.png')
    plt.show()


def plot_crossvalidation_results(model1, model2):
    auroc1 = [v['auroc'] for k, v in model1.items()]
    auroc2 = [v['auroc'] for k, v in model2.items()]
    auroc = pd.DataFrame({'AlphaMissense': auroc1, 'VariPred': auroc2})
    auroc = pd.melt(auroc, var_name='model', value_name='auROC')

    mcc1 = [v['mcc'] for k, v in model1.items()]
    mcc2 = [v['mcc'] for k, v in model2.items()]
    mcc = pd.DataFrame({'AlphaMissense': mcc1, 'VariPred': mcc2})
    mcc = pd.melt(mcc, var_name='model', value_name='MCC')

    acc1 = [v['classification_report']['accuracy'] for k, v in model1.items()]
    acc2 = [v['classification_report']['accuracy'] for k, v in model2.items()]
    acc = pd.DataFrame({'AlphaMissense': acc1, 'VariPred': acc2})
    acc = pd.melt(acc, var_name='model', value_name='accuracy')

    f1_1 = [v['classification_report']['weighted avg']['f1-score'] for k, v in model1.items()]
    f1_2 = [v['classification_report']['weighted avg']['f1-score'] for k, v in model2.items()]
    f1 = pd.DataFrame({'AlphaMissense': f1_1, 'VariPred': f1_2})
    f1 = pd.melt(f1, var_name='model', value_name='f1-score')

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(auroc, x="auROC", y="model", palette="Paired", errorbar="sd", capsize=.05)
    ax.set(xlabel='auROC')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/auroc.png")
    plt.savefig("../plots/varipred_alphamissense_comp/auroc.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(mcc, x="MCC", y="model", palette="Paired", errorbar="sd", capsize=.05)
    ax.set(xlabel='MCC')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/mcc.png")
    plt.savefig("../plots/varipred_alphamissense_comp/mcc.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(acc, x="accuracy", y="model", palette="Paired", errorbar="sd", capsize=.05)
    ax.set(xlabel='Accuracy')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/accuracy.png")
    plt.savefig("../plots/varipred_alphamissense_comp/accuracy.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(f1, x="f1-score", y="model", palette="Paired", errorbar="sd", capsize=.05)
    ax.set(xlabel='Class weighted F1-score')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/f1_score.png")
    plt.savefig("../plots/varipred_alphamissense_comp/f1_score.pdf")


def af_protlen_corr(data, feature_name):
    """
    This function takes in a nested dictionary of alphafold features and plots two correlation plots: one for mean af
    feature and one for the max af feature, against protein length.
    """
    if feature_name not in data:
        print(f"Feature '{feature_name}' not found in the data.")
        return

    feature_data = data[feature_name]

    x_values = []
    y_values = []

    for protein_id, value in feature_data.items():
        protein_len = data['protein_len'].get(protein_id, 0)

        if protein_len is not None and not np.isnan(protein_len) and value != 0:
            x_values.append(protein_len)
            y_values.append(value)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_values, y_values, alpha=0.2)
    plt.xlabel('Protein length')
    plt.ylabel(f'{feature_name.capitalize()} normalised pLDDT score')
    plt.ylim(-0.01, 1.03)
    plt.grid(True)

    correlation_coefficient, _ = pearsonr(x_values, y_values)
    plt.text(max(x_values) * 1.03, 0.03, f'Pearson correlation = {correlation_coefficient:.2f}', fontsize=12,
             color='black', horizontalalignment='right')

    plt.savefig(f"../plots/af_feature_protlen_correlation/af_{feature_name}_protlen_corr.png")
    plt.savefig(f"../plots/af_feature_protlen_correlation/af_{feature_name}_protlen_corr.pdf")


def feature_correlation(dataframe):
    feature_names = dataframe.columns[1:]
    fig, axes = plt.subplots(nrows=len(feature_names), ncols=len(feature_names), figsize=(40, 40))

    for i, feature_x in enumerate(feature_names):
        for j, feature_y in enumerate(feature_names):
            ax = axes[i, j]

            if i == j:
                ax.hist(dataframe[feature_x], bins=20, alpha=0.7)
                ax.set_xlabel(feature_x)
                ax.set_ylabel('Frequency')
            else:
                ax.scatter(dataframe[feature_x], dataframe[feature_y], alpha=0.5)
                ax.set_xlabel(feature_x)
                ax.set_ylabel(feature_y)
                ax.axline((0, 0), slope=1, color='black', linestyle='--')

                correlation = dataframe[[feature_x, feature_y]].corr().iloc[0, 1]
                ax.annotate(f'Corr: {correlation:.2f}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

    plt.tight_layout()
    plt.show()
    # plt.savefig('../plots/features/feature_correlations.pdf')


def correlation_heatmap(df):
    feature_df = df.iloc[:, 1:]
    correlation_matrix = feature_df.corr()

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Pearson Correlation Heatmap")
    # plt.savefig('../plots/feature_correlation_heatmaps/feature_correlations_heatmap_nov2023.pdf')
    # plt.savefig('../plots/feature_correlation_heatmaps/feature_correlations_heatmap_nov2023.png')
    plt.close()


def umap(df):
    df_target_0 = df[df['target'] == 0]
    df_target_1 = df[df['target'] == 1]

    df_target_0_downsampled = df_target_0.sample(n=len(df_target_1), random_state=42)

    df_downsampled = pd.concat([df_target_0_downsampled, df_target_1])
    features = df_downsampled.iloc[:, 1:-1]
    target = df_downsampled.iloc[:, -1]
    gene_names = df_downsampled.iloc[:, 0]

    umap_model = UMAP(n_components=3, n_neighbors=100, min_dist=0.01)
    umap_results = umap_model.fit_transform(features)
    umap_df = pd.DataFrame(umap_results, columns=['UMAP1', 'UMAP2'])
    umap_df['Target'] = target
    umap_df['Gene'] = gene_names

    sns.scatterplot(x='UMAP1', y='UMAP2',  hue='Target', data=umap_df, palette='Paired')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.show()
