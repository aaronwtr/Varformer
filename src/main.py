import utils
from dataloader import MissenseVariantLoader, GeneCharacterisation


def load_missense_variants():
    MVL = MissenseVariantLoader()
    print("Missense variants loaded!\n")
    return MVL


def gene_characterisation():
    features = GeneCharacterisation()
    print("Gene characterisation features loaded!\n")
    return features


if __name__ == "__main__":
    mvl = load_missense_variants()
    variant = mvl.variant_data
    utils.evaluate_am(variant)
    # gene_characterisation()
