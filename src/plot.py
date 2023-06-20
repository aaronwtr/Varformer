import matplotlib.pyplot as plt
import numpy as np


def tractibility_plot(tractability_scores, fda_labels, plottype=None, fda=True):
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

