import argparse
import pandas as pd

from dataloader import MissenseVariantLoader, GeneCharacterisation
import utils, config, plot


def load_missense_variants():
    MVL = MissenseVariantLoader(evaluation=True)
    print("Missense variants loaded!\n")
    return 0


def gene_characterisation():
    features = GeneCharacterisation()
    print("Gene characterisation features loaded!\n")
    return features


if __name__ == "__main__":
    load_missense_variants()
    # data = pd.read_csv("data/merged_varipred_clinvar.csv", sep="\t")
    # # utils.post_hoc_classification(data)
    # plot.varipred_kde_plot(data)
