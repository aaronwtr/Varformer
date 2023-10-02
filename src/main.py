import pandas as pd
import pickle as pkl

from dataloader import MissenseVariantLoader, GeneCharacterisation
import plot
import utils


def load_missense_variants():
    MVL = MissenseVariantLoader()
    print("Missense variants loaded!\n")
    return MVL


def gene_characterisation():
    features = GeneCharacterisation()
    print("Gene characterisation features loaded!\n")
    return features


if __name__ == "__main__":
    # mvl = load_missense_variants()

    # gene_characterisation()

    # mvl = load_missense_variants()  # Load missense variants

    # open am_results pickle file
    with open('../data/VariPred/output/am_crossval_results.pkl', 'rb') as f:
        am_results = pkl.load(f)
    with open('../data/VariPred/output/vp_crossval_results.pkl', 'rb') as f:
        vp_results = pkl.load(f)

    plot.plot_crossvalidation_results(am_results, vp_results)
