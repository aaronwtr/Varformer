import matplotlib.pyplot as plt
import numpy as np


def tractibility_plot(tractability_scores, fda_labels, plottype=None, fda=False):
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
        plt.scatter(range(len(tract_scores_fda)), tract_scores_fda, s=5)
    else:
        plt.scatter(range(len(tract_score)), tract_score, s=5, c=colors)

    plt.xlabel('Gene Index')
    plt.ylabel('Tractability Score')
    plt.ylim(0, max(tract_score) + 0.3)
    if plottype == 'SM':
        plt.title('Small Molecule Tractability Scores Colored by FDA Approval Status')
    else:
        plt.title('Antibody Tractability Scores Colored by FDA Approval Status')

    plt.xticks([])
    plt.show()

