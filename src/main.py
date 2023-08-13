import argparse

from dataloader import MissenseVariantLoader, GeneCharacterisation
import utils
import config
import plot


def load_missense_variants():
    MVL = MissenseVariantLoader(train=True)
    print("Missense variants loaded!\n")
    return 0


def gene_characterisation():
    features = GeneCharacterisation()
    print("Gene characterisation features loaded!\n")
    return features


if __name__ == "__main__":
    #load_missense_variants()
    vp_data = utils.preprocess_varipred_output("data/VariPred/output/varipred_output_data.csv")
    plot.varipred_kde_plot(vp_data)
