import argparse

from dataloader import MissenseVariantLoader, GeneCharacterisation
from utils import split_data, find_error_files


def load_variants(parse=True):
    """
    This function reads in uniparc IDs from the ELGH data set and retrieves the multiple sequence alignment from the
    UNIPROT database. The MSA is then saved as a .fasta file and returned to the main function as a list. This is the
    first preprocessing step for the pathogenicty pipeline where we score a genes pathogenicity based on its variants'
    MSAs.
    """
    parser = argparse.ArgumentParser(description='Script to process variants')
    ELGH_DIR = "data/elgh/"
    MIVA_PATH = f"{ELGH_DIR}all_functional.gatk_PASS.FS_30.DP_0.GQ_20.AB_0.01.functional.missingness_lt_0.genotype_" \
                f"counts.present_in_ELGH.n_transcripts_corrected.txt"
    parser.add_argument('--data', type=str, default=f"{MIVA_PATH}")
    GENOME_PATH = f"data/hg38.fasta"
    parse = False

    if parse:
        args = parser.parse_args()
        data_path = args.data
        if data_path is MIVA_PATH:
            MVL = MissenseVariantLoader(MIVA_PATH, GENOME_PATH)
        else:
            MVL = MissenseVariantLoader(data_path, GENOME_PATH)
    else:
        MVL = MissenseVariantLoader(MIVA_PATH, GENOME_PATH)

    return 0


def gene_characterisation():
    features = GeneCharacterisation()
    return features


if __name__ == "__main__":
    load_variants()
    # find_error_files("data/VariPred/")
