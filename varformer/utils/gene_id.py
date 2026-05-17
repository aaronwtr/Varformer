"""Gene / protein identifier helpers.

Extracted from src/utils/utils.py (Phase 6 refactor).
"""
import warnings

import requests
import biorosetta as br


def map_gene_names(list_of_genes: list, source_type: str, target_type: str) -> dict:
    """Map a list of gene identifiers between two ID types using biorosetta.

    Returns a dict mapping each source ID to the converted target ID.
    Genes without a mapping get the value 'N/A' and a warning is issued.
    """
    idmap = br.IDMapper('all')
    list_of_targets = idmap.convert(list_of_genes, source_type, target_type)
    if 'N/A' in list_of_targets:
        warnings.warn("Some genes were not found in the mapping. Check the input list of genes.")
        missing = [list_of_genes[i] for i, x in enumerate(list_of_targets) if x == 'N/A']
        warnings.warn(f"Number of missing genes: {len(missing)}")
    return dict(zip(list_of_genes, list_of_targets))


def get_protein_length(ensp: str, ensg: str) -> int:
    """Query the Ensembl REST API to retrieve the protein length for a given ENSP/ENSG ID.

    Falls back to ENSG-based lookup if the ENSP request fails.
    """
    ensp_api_url = f"https://rest.ensembl.org/sequence/id/{ensp}"
    ensg_api_url = f"https://rest.ensembl.org/sequence/id/{ensg}?type=protein;multiple_sequences=1"
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.get(ensp_api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            protein = data.get('seq', None)
            protein_length = len(protein)
            if protein_length is not None:
                return protein_length
            raise KeyError(f"Protein length for {ensp} not found in the response.")
        raise KeyError(
            f"Failed to retrieve protein information. Status code: {response.status_code}"
        )
    except KeyError:
        response = requests.get(ensg_api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            protein = data[0].get('seq', None)
            protein_length = len(protein)
            if protein_length is not None:
                return protein_length
            raise KeyError(f"Protein length for {ensg} not found in the response.")
        raise KeyError(
            f"Failed to retrieve protein information. Status code: {response.status_code}"
        )
