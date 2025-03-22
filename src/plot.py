import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import pearsonr, spearmanr
# from umap import UMAP


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


def plot_crossvalidation_results(model1, model2, model3):
    auroc1 = [v['auroc'] for k, v in model1.items()]
    auroc2 = [v['auroc'] for k, v in model2.items()]
    auroc3 = [v['auroc'] for k, v in model3.items()]
    auroc = pd.DataFrame({'AlphaMissense': auroc1, 'VariPred-GH': auroc2, 'VariPred': auroc3})
    auroc = pd.melt(auroc, var_name='model', value_name='auROC')

    mcc1 = [v['mcc'] for k, v in model1.items()]
    mcc2 = [v['mcc'] for k, v in model2.items()]
    mcc3 = [v['mcc'] for k, v in model3.items()]
    mcc = pd.DataFrame({'AlphaMissense': mcc1, 'VariPred-GH': mcc2, 'VariPred': mcc3})
    mcc = pd.melt(mcc, var_name='model', value_name='MCC')

    acc1 = [v['classification_report']['accuracy'] for k, v in model1.items()]
    acc2 = [v['classification_report']['accuracy'] for k, v in model2.items()]
    acc3 = [v['classification_report']['accuracy'] for k, v in model3.items()]
    acc = pd.DataFrame({'AlphaMissense': acc1, 'VariPred-GH': acc2, 'VariPred': acc3})
    acc = pd.melt(acc, var_name='model', value_name='accuracy')

    f1_1 = [v['classification_report']['weighted avg']['f1-score'] for k, v in model1.items()]
    f1_2 = [v['classification_report']['weighted avg']['f1-score'] for k, v in model2.items()]
    f1_3 = [v['classification_report']['weighted avg']['f1-score'] for k, v in model3.items()]
    f1 = pd.DataFrame({'AlphaMissense': f1_1, 'VariPred-GH': f1_2, 'VariPred': f1_3})
    f1 = pd.melt(f1, var_name='model', value_name='f1-score')

    spearman1 = [v['spearman_corr'] for k, v in model1.items()]
    spearman2 = [v['spearman_corr'] for k, v in model2.items()]
    spearman3 = [v['spearman_corr'] for k, v in model3.items()]
    spearman = pd.DataFrame({'AlphaMissense': spearman1, 'VariPred-GH': spearman2, 'VariPred': spearman3})
    spearman = pd.melt(spearman, var_name='model', value_name='spearman_corr')

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(spearman, x="spearman_corr", y="model", palette="Blues", errorbar="sd", capsize=.05)
    ax.set(xlabel='Spearman Correlation')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/spearman_corr.png")
    plt.savefig("../plots/varipred_alphamissense_comp/spearman_corr.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(auroc, x="auROC", y="model", palette="Blues", errorbar="sd", capsize=.05)
    ax.set(xlabel='auROC')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/auroc.png")
    plt.savefig("../plots/varipred_alphamissense_comp/auroc.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(mcc, x="MCC", y="model", palette="Blues", errorbar="sd", capsize=.05)
    ax.set(xlabel='MCC')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/mcc.png")
    plt.savefig("../plots/varipred_alphamissense_comp/mcc.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(acc, x="accuracy", y="model", palette="Blues", errorbar="sd", capsize=.05)
    ax.set(xlabel='Accuracy')
    ax.set(xlim=(0.0, 1.005))
    sns.despine()
    plt.savefig("../plots/varipred_alphamissense_comp/accuracy.png")
    plt.savefig("../plots/varipred_alphamissense_comp/accuracy.pdf")

    plt.figure(figsize=(9, 5))
    sns.set_style("whitegrid")

    ax = sns.barplot(f1, x="f1-score", y="model", palette="Blues", errorbar="sd", capsize=.05)
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


def correlation_heatmap(df, top_n=None, filename=None):
    feature_df = df.iloc[:, :-1]  # Select all columns except the last one (target)
    correlation_matrix = feature_df.corr()

    # If top_n is provided, select only the top N features by absolute correlation
    if top_n is not None and top_n < len(correlation_matrix.columns):
        # Calculate the mean absolute correlation for each feature (excluding self-correlation)
        abs_corr = correlation_matrix.abs()
        # Set diagonal to 0 to exclude self-correlation
        np.fill_diagonal(abs_corr.values, 0)
        mean_abs_corr = abs_corr.mean()

        # Get the top N features
        top_features = mean_abs_corr.nlargest(top_n).index.tolist()
        # Filter correlation matrix to include only top features
        correlation_matrix = correlation_matrix.loc[top_features, top_features]

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create a heatmap
    figsize = (10, 8)
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(correlation_matrix,
                          cmap="Blues",
                          annot=False,
                          mask=mask,  # Apply the mask
                          square=True,
                          xticklabels=True,
                          yticklabels=True)  # Make the cells square

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=4)  # Set smaller font size for x-axis
    plt.yticks(fontsize=4)

    # Get current axes
    ax = plt.gca()

    # Get current tick positions and labels
    y_ticks = ax.get_yticks()
    y_ticklabels = [item.get_text() for item in ax.get_yticklabels()]
    x_ticks = ax.get_xticks()
    x_ticklabels = [item.get_text() for item in ax.get_xticklabels()]

    # Remove the top tick on y-axis and last tick on x-axis
    ax.set_yticks(y_ticks[1:])  # Remove first y tick
    ax.set_yticklabels(y_ticklabels[1:])  # Remove first y label

    ax.set_xticks(x_ticks[:-1])  # Remove last x tick
    ax.set_xticklabels(x_ticklabels[:-1])  # Remove last x label

    base_path = '../plots/feature_correlations/'
    if not filename:
        filename = f'{base_path}feature_correlation_heatmap'
    else:
        filename = f'{base_path}feature_correlation_heatmap_{filename}'
    if top_n is not None:
        filename += f'_top{top_n}'
    filename += '.pdf'

    plt.tight_layout()  # Adjust layout to make room for rotated labels
    plt.savefig(filename,
                dpi=300,
                bbox_inches='tight')
    plt.close()


def correlation_screeplot(df, top_n=None, filename=None):
    # Extract features (all columns except the last one which is assumed to be the target)
    feature_df = df.iloc[:, :-1]

    # Calculate correlation matrix
    correlation_matrix = feature_df.corr()

    # Calculate mean absolute correlation for each feature (excluding self-correlation)
    abs_corr = correlation_matrix.abs()
    np.fill_diagonal(abs_corr.values, 0)  # Exclude self-correlation
    mean_abs_corr = abs_corr.mean()

    # Sort features by descending correlation
    sorted_corr = mean_abs_corr.sort_values(ascending=False)

    # Create the screeplot
    plt.figure(figsize=(12, 6))

    # Plot the correlations
    plt.plot(range(1, len(sorted_corr) + 1), sorted_corr.values, 'o-', color='tab:blue', markersize=4)

    # Add vertical line at top_n if specified
    if top_n is not None and top_n < len(sorted_corr):
        plt.axvline(x=top_n, color='red', linestyle='--',
                    label=f'Top {top_n} features cutoff')

    # Add labels and title
    plt.xlabel('Feature Rank')
    plt.ylabel('Mean Absolute Correlation')
    plt.grid(True, linestyle='--', alpha=0.7)

    if top_n is not None:
        plt.legend()

    plt.tight_layout()

    # Save the plot
    base_path = '../plots/feature_correlations/'
    if not filename:
        filename = f'{base_path}feature_correlation_screeplot'
    else:
        filename = f'{base_path}feature_correlation_screeplot_{filename}'
    if top_n is not None:
        filename += f'_top{top_n}'

    plt.savefig(f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.close()


def umap(df):
    features = df.iloc[:, :-1]  # Select all columns except the last one (target)
    target = df['target']

    # Create the UMAP reducer object
    reducer = UMAP()

    # Fit and transform the data into a lower-dimensional UMAP embedding
    embedding = reducer.fit_transform(features)

    # Create the scatter plot
    plt.figure(figsize=(10, 8))

    # Separate the embeddings based on the target
    embedding_0 = embedding[target == 0]
    embedding_1 = embedding[target == 1]

    # Plot the points labeled as 0
    plt.scatter(embedding_0[:, 0], embedding_0[:, 1], c='tab:gray', s=5, label='Unknown target status')

    # Plot the points labeled as 1
    plt.scatter(embedding_1[:, 0], embedding_1[:, 1], c='tab:blue', s=5, alpha=0.7, label='Approved drug targets')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.xticks([])
    plt.yticks([])

    # TODO: Use config to determine plot name
    # plt.savefig('../plots/umap_transformer_autoencoder_h256_io1024.pdf', dpi=300)
    # plt.show()


def plot_kde(pseudo_labels):
    plt.figure(dpi=300)
    sns.kdeplot(pseudo_labels.detach().numpy()[pseudo_labels.detach().numpy() != -1], fill=True)
    plt.xlabel("Pseudo-label value")
    plt.ylabel("Density")
    # plt.savefig("../plots/pseudolabel_distribution.pdf")


def plot_embedding_distribution(embeddings: pd.DataFrame) -> None:
    """
    Plots a boxplot of the embeddings
    """
    random_columns = np.random.choice(embeddings.columns, 3, replace=False)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

    # For each subplot, plot a boxplot of one of the random columns
    for ax, column in zip(axes, random_columns):
        ax.violinplot(embeddings[column])
        ax.set_title(column)
        ax.set_xticks([])

    # Set common labels
    fig.text(0.5, 0.04, 'Latent dimension', ha='center', va='center')
    fig.text(0.06, 0.5, 'Embedding value', ha='center', va='center', rotation='vertical')

    # plt.show()
    plt.savefig("../plots/transformer_autoencoder_embedding_distribution_3.pdf", dpi=300)
