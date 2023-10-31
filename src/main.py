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
    gc = gene_characterisation()

    # TODO: Properly load pathogenicity features from mvl
