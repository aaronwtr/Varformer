"""Amino-acid encoding helpers.

Extracted from src/utils/utils.py (Phase 6 refactor).
"""


def aa_to_idx(aa: str, dna_encoded: bool = False) -> int:
    """Convert a single-letter amino acid code to an integer index.

    NOTE: The standard (non-DNA-encoded) map includes the non-standard residue U.
    """
    if not dna_encoded:
        aa_to_idx_map = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
            'T': 16, 'U': 17, 'V': 18, 'W': 19, 'Y': 20,
        }
    else:
        aa_to_idx_map = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15,
            'T': 16, 'V': 17, 'W': 18, 'Y': 19,
        }
    return aa_to_idx_map[aa]


def three_letter_aa_to_idx(aa: str) -> int:
    """Convert a three-letter amino acid code to an integer index."""
    three_letter_aa_to_idx_map = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
    }
    return three_letter_aa_to_idx_map[aa]


def aa1_to_aa3(single_code: str) -> str:
    """Convert a single-letter amino acid code to the three-letter equivalent."""
    amino_acids = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
    }
    return amino_acids.get(single_code.upper(), 'Unknown')
