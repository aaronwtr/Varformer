import utils
from dataloader import MissenseVariantLoader, GeneCharacterisation


def load_missense_variants():
    MVL = MissenseVariantLoader(train=True)
    print("Missense variants loaded!\n")
    return 0


def gene_characterisation():
    features = GeneCharacterisation()
    print("Gene characterisation features loaded!\n")
    return features


if __name__ == "__main__":
    load_missense_variants()
    # TODO: Test retraining algorithm on small subset of train and test data
    # utils.find_error_files('data/VariPred/train/')
