import pandas as pd
import requests
from tqdm import tqdm
import os
from Bio import SeqIO


def get_data():
    """
    Get UNIPARC data from the dataset.
    :return: list of UNIPROT IDs and the corresponding gene name in a dictionary. Gene name is key and UNIPROT ID is
    value.
    """
    raw_data = pd.read_csv("all_chrs.HC_LoF.genotype_counts.after_genotype_filtering.csv", sep="\t")
    uniprot_ids = raw_data["UNIPARC"].tolist()
    gene_names = raw_data["SYMBOL"].tolist()
    uniprot_ids_dict = {}
    for i in range(len(uniprot_ids)):
        if gene_names[i] not in uniprot_ids_dict:
            uniprot_ids_dict[gene_names[i]] = [uniprot_ids[i]]
        else:
            uniprot_ids_dict[gene_names[i]].append(uniprot_ids[i])
    return uniprot_ids_dict


def get_msa(uniprot_ids_dict):
    """
    Get the multiple sequence alignment from the UNIPROT database.
    :param uniprot_ids: list of UNIPROT IDs.
    :return: list of multiple sequence alignments.
    """
    for gene_name in tqdm(uniprot_ids_dict):
        for uniprot_id in uniprot_ids_dict[gene_name]:
            url = f"https://www.uniprot.org/uniparc/{uniprot_id}.fasta"
            response = requests.get(url)
            fasta_msa = response.text
            fasta_msa = fasta_msa.replace("status=active", "")
            fasta_msa = fasta_msa.replace("status=inactive", "")
            fasta_msa = f"{fasta_msa[0:14]}|{gene_name}{fasta_msa[14:]}"
            if "elgh_HC_LoF_MSA.fasta" in os.listdir():
                with open("elgh_HC_LoF_MSA.fasta", "a") as f:
                    f.write(fasta_msa)
            else:
                with open("elgh_HC_LoF_MSA.fasta", "w") as f:
                    f.write(fasta_msa)
            f.close()


if __name__ == "__main__":
    preprocess = True
    if preprocess:
        uni_ids = get_data()
        get_msa(uni_ids)
        msa = list(SeqIO.parse("elgh_HC_LoF_MSA.fasta", "fasta"))
    else:
        msa = list(SeqIO.parse("elgh_HC_LoF_MSA.fasta", "fasta"))
