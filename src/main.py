import os

from dataloader import VariantLoader, GeneCharacterisation


def load_proteins():
    """
    This function reads in uniparc IDs from the ELGH data set and retrieves the multiple sequence alignment from the
    UNIPROT database. The MSA is then saved as a .fasta file and returned to the main function as a list. This is the
    first preprocessing step for the pathogenicty pipeline where we score a genes pathogenicity based on its variants'
    MSAs.

    """
    ELGH_DIR = "data/elgh/"
    UNIPARC_PATH = f"{ELGH_DIR}all_chrs.HC_LoF.genotype_counts.after_genotype_filtering.csv"
    UNIPARC_PATH = os.path.normpath(UNIPARC_PATH)
    MSA_OUTPUT = f"{ELGH_DIR}elgh_HC_LoF_MSA.fasta"
    MSA_OUTPUT = os.path.normpath(MSA_OUTPUT)

    VL = VariantLoader(UNIPARC_PATH, MSA_OUTPUT)
    raw_data = VL.data_reader()
    msa_data = VL.parse_data(raw_data)
    return msa_data


def gene_characterisation():
    features = GeneCharacterisation()
    return features


if __name__ == "__main__":
    gc_features = gene_characterisation()


