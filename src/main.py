import os

from dataloader import MissenseVariantLoader, GeneCharacterisation
from plot import tractibility_plot


def load_variants():
    """
    This function reads in uniparc IDs from the ELGH data set and retrieves the multiple sequence alignment from the
    UNIPROT database. The MSA is then saved as a .fasta file and returned to the main function as a list. This is the
    first preprocessing step for the pathogenicty pipeline where we score a genes pathogenicity based on its variants'
    MSAs.

    """
    ELGH_DIR = "data/elgh/"
    MIVA_PATH = f"{ELGH_DIR}all_functional.gatk_PASS.FS_30.DP_0.GQ_20.AB_0.01.functional.missingness_lt_0.genotype_" \
                f"counts.present_in_ELGH.n_transcripts_corrected.txt"
    GENOME_PATH = f"data/hg38.fasta"

    MVL = MissenseVariantLoader(MIVA_PATH, GENOME_PATH)

    return 0


def gene_characterisation():
    features = GeneCharacterisation()
    return features


if __name__ == "__main__":
    gc_features = gene_characterisation()
    raw_tract_data = gc_features.bin_tract_features
    tract_data = gc_features.tract_features
    fda_approvals = gc_features.ot_fda_approvals

    tractibility_plot(tract_data[0], fda_approvals[0], 'SM')
    tractibility_plot(tract_data[1], fda_approvals[1], 'AB')
