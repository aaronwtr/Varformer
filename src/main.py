import argparse

from dataloader import MissenseVariantLoader, GeneCharacterisation
import utils
import config


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
    # find_error_files("data/VariPred/output/")
