from dataloader import SequenceDataPreprocessing


def load_proteins():
    """
    This function reads in uniparc IDs from the ELGH data set and retrieves the multiple sequence alignment from the
    UNIPROT database. The MSA is then saved as a .fasta file and returned to the main function as a list. This is the
    first preprocessing step for the pathogenicty pipeline where we score a genes pathogenicity based on its variants'
    MSAs.

    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    UNIPARC_PATH = f"{ROOT_DIR}/all_chrs.HC_LoF.genotype_counts.after_genotype_filtering.csv"
    UNIPARC_PATH = os.path.normpath(UNIPARC_PATH)
    MSA_OUTPUT = f"{ROOT_DIR}/elgh_HC_LoF_MSA.fasta"
    MSA_OUTPUT = os.path.normpath(MSA_OUTPUT)

    DPP = SequenceDataPreprocessing(UNIPARC_PATH, MSA_OUTPUT)
    raw_data = DPP.data_reader()
    msa_data = DPP.parse_data(raw_data)
    return msa_data


if __name__ == "__main__":
    msa_data = load_proteins()
