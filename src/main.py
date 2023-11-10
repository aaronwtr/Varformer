import pandas as pd
import pickle as pkl

from preprocessing import MissenseVariantPreprocessor, GeneCharacterisationPreprocessor
import plot
import utils


def main():
    MVP = MissenseVariantPreprocessor()
    print("Missense variants loaded!\n")

    GCP = GeneCharacterisationPreprocessor()
    print("Gene characterisation features loaded!\n")


if __name__ == "__main__":
    main()
