import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
