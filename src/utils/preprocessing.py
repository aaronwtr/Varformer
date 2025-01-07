import pandas as pd
import numpy as np
import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Tuple, Optional


def count_scaling(counts):
    """
    Implements \(x' = \frac{x - x_{min}}{x_{max} - x_{min}}\).
    :param counts: array of count
    :return: list of features scaled between 0 and 1.
    """
    return [(count - min(counts)) / (max(counts) - min(counts)) for count in counts]


def lognorm(vector, eps=1e-8):
    """
    Takes in a vector and return a log-normalized spectrum
    """
    vector_base = np.array(vector)
    vector = vector_base + eps
    sqrt_sum = np.sqrt(np.sum(vector ** 2))
    vector = -np.log(vector / sqrt_sum)
    vector = vector - np.max(vector)
    vector = [-element for element in vector if element != 0]
    return vector


def translate_sequence(sequence):
    """
    Translate a DNA sequence to an amino acid sequence.
    """
    seq = Seq.Seq(sequence)
    return seq.translate()


def preprocess_varipred_output(varipred_output_path):
    """
    Preprocess the VariPred output to match the original data.
    """
    if os.path.exists("../data/VariPred/varipred_output_data.csv"):
        return pd.read_csv("../data/VariPred/varipred_output_data.csv", sep="\t")
    else:
        files = os.listdir(varipred_output_path)
        files.sort(key=extract_number)
        dataframes = []
        for filename in files[1:]:
            if filename.endswith(".txt"):
                file_path = os.path.join(varipred_output_path, filename)
                df = pd.read_csv(file_path, sep='\t')
                df['target_id'] = df['target_id'].apply(correct_aa_position)
                dataframes.append(df)
        varipred_data = pd.concat(dataframes, ignore_index=True)
        varipred_data.to_csv("../data/VariPred/varipred_output_data.csv", sep="\t", index=False)
        return varipred_data


def combine_varipred_elgh(varipred_data, elgh_data):
    merged_df = pd.merge(varipred_data, elgh_data, left_on='target_id', right_on='varipred_id', how='inner')
    merged_df.drop('varipred_id', axis=1, inplace=True)
    columns = elgh_data.columns.tolist()
    columns.remove('varipred_id')
    columns = columns + ['classification', 'probability']
    merged_df = merged_df[columns]
    new_column_names = {'classification': 'vpgh_classification', 'probability': 'vpgh_probability'}
    merged_df.rename(columns=new_column_names, inplace=True)
    merged_df.to_csv("../data/elgh/varipred_elgh_data.csv", sep="\t", index=False)


def clinvar_filtering(clinvar_data):
    """
    Filters the ClinVar data to only contain rows assembled with GRCh38 and are missense variants.
    """
    if os.path.exists("../data/clinvar/clinvar_filtered.csv"):
        return pd.read_csv("../data/clinvar/clinvar_filtered.csv", sep="\t")
    else:
        columns = ["Name", "GeneSymbol", "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
                   "ClinSigSimple"]
        clinvar_data = clinvar_data[clinvar_data["Assembly"] == "GRCh38"]
        clinvar_data = clinvar_data[clinvar_data["Type"] == "single nucleotide variant"]
        clinvar_data = clinvar_data[
            (clinvar_data["ReviewStatus"] == "criteria provided, single submitter") |
            (clinvar_data["ReviewStatus"] == "criteria provided, multiple submitters, no conflicts")
            ]
        clinvar_data = clinvar_data.dropna(subset=["ReferenceAlleleVCF", "AlternateAlleleVCF"])
        clinvar_data = clinvar_data[columns]
        clinvar_data.to_csv("../data/clinvar/clinvar_filtered.csv", sep="\t", index=False)
        return clinvar_data


def clinvar_varipred_id(varipred_data, clinvar_data):
    """
    Makes new ID for varipred and clinvar overlap: {gene_name}_{genomic_position}_{ref_allele}_{alt_allele}
    """
    if os.path.exists("../data/clinvar/clinvar_varipred_id_final.csv"):
        clinvar_data = pd.read_csv("../data/clinvar/clinvar_varipred_id_final.csv", sep="\t")
    else:
        clinvar_data["vp_cv_id"] = clinvar_data["GeneSymbol"] + "_" + clinvar_data["Start"].astype(str) + "_" + \
                                   clinvar_data["ReferenceAlleleVCF"] + "_" + clinvar_data["AlternateAlleleVCF"]
        clinvar_data = clinvar_data.drop_duplicates(subset=['vp_cv_id'])
        clinvar_data.to_csv("../data/clinvar/clinvar_vp_id_final.csv", sep="\t", index=False)

    if os.path.exists("../data/VariPred/varipred_vp_id_final.csv"):
        varipred_data = pd.read_csv("../data/VariPred/varipred_vp_id_final.csv", sep="\t")
    else:
        ref_aa = varipred_data["REF"]
        alt_aa = varipred_data["ALT"]
        varipred_data["vp_cv_id"] = varipred_data["SYMBOL"] + "_" + varipred_data["POS"].astype(str) + "_" + \
                                    ref_aa + "_" + alt_aa

        varipred_data = varipred_data.drop_duplicates(subset=['vp_cv_id'])
        varipred_data.to_csv("../data/VariPred/varipred_vp_id_final.csv", sep="\t", index=False)

    return varipred_data, clinvar_data


def combine_varipred_clinvar(varipred_data, clinvar_data):
    """
    Combine the VariPred and ClinVar data. The columns we want to keep are: vp_cv_id, SYMBOL, POS, ReferenceAlleleVCF,
    AlternateAlleleVCF, ClinSigSimple, vp_classification, vp_probability
    """
    if not os.path.exists("../data/merged_varipred_clinvar.csv"):
        merged_df = pd.merge(varipred_data, clinvar_data, on='vp_cv_id', how='inner')
        columns = ["vp_cv_id", "SYMBOL", "ReferenceAlleleVCF", "AlternateAlleleVCF", "POS", "UNIPARC", "Amino_acids",
                   "Protein_position", "ClinSigSimple", "vp_classification", "vp_probability"]
        merged_df = merged_df[columns]
    else:
        merged_df = pd.read_csv("../data/merged_varipred_clinvar.csv", sep="\t")
    merged_df.to_csv("../data/merged_varipred_clinvar.csv", sep="\t", index=False)
    return merged_df


def add_varipred_id(config):
    variant_data = pd.read_csv(config['paths']['MIVA_PATH'], sep="\t")
    columns = ["SYMBOL", "Protein_position", "Amino_acids"]
    variant_data_of_interest = variant_data[columns]

    variant_data = variant_data[~variant_data_of_interest['Protein_position'].astype(str).str.contains('-')]
    variant_data_of_interest = variant_data_of_interest[
        ~variant_data_of_interest['Protein_position'].astype(str).str.contains('-')]
    aa_ref = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[0]
    aa_alt = variant_data_of_interest["Amino_acids"].str.split("/", expand=True)[1]
    aa_index = variant_data_of_interest["Protein_position"].astype(str)
    variant_data["varipred_id"] = variant_data_of_interest["SYMBOL"] + "_" + aa_index + "_" + aa_ref + "_" \
                                  + aa_alt
    variant_data.to_csv(config['paths']['MIVA_PATH'], sep="\t", index=False)


def downsampler(data):
    pos_data = data[data.label == 1]
    neg_data = data[data.label == 0]
    num_samples = len(pos_data) * 4
    neg_data = neg_data.sample(n=num_samples, random_state=42)
    downsampled_data = pd.concat([pos_data, neg_data])
    return downsampled_data


def analyze_legacy_pathogenicity(variant_data):
    """
    Analyze the pathogenicity determined by SIFT and PolyPhen.
    """
    sift = variant_data["SIFT"].values
    polyphen = variant_data["PolyPhen"].values

    pathogenicity = pd.DataFrame({"sift": sift, "polyphen": polyphen})

    pathogenicity['sift'] = pathogenicity['sift'].str.extract(r'\((.*?)\)').astype(float)
    pathogenicity['polyphen'] = pathogenicity['polyphen'].str.extract(r'\((.*?)\)').astype(float)
    plot.variant_sparsity_barplot(pathogenicity, save=True)

    num_vars = len(list(pathogenicity['polyphen'].values))

    sift_nan_count = pathogenicity['sift'].isna().sum()
    polyphen_nan_count = pathogenicity['polyphen'].isna().sum()

    pp_sparsity = round(polyphen_nan_count / num_vars, 3)
    sift_sparsity = round(sift_nan_count / num_vars, 3)

    print(f"PolyPhen sparsity: {pp_sparsity}")
    print(f"SIFT sparsity: {sift_sparsity}")

    plot.pathogenicity_correlation_plot(pathogenicity, save=True)

    print("Done analyzing pathogenicity.")


def featurise(features: dict, feature_name: Optional[str] = '') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sparse_feature_dict = {}
    with open("../data/features/raw_miva_feature_matrix.pkl", 'rb') as f:
        feature_matrix = pkl.load(f)

    if isinstance(next(iter(features.values())), csr_matrix):
        for gene, matrix in tqdm(features.items(), total=len(features)):
            dense_vector = matrix.toarray().flatten().reshape(1, -1)
            sparse_vector = csr_matrix(dense_vector)
            sparse_feature_dict[gene] = sparse_vector
        feature_matrix[feature_name] = feature_matrix["ENSG"].map(sparse_feature_dict)
        zero_fill = [0.0] * dense_vector.shape[1]
        zero_fill = np.array(zero_fill, dtype=np.float32)
        if feature_matrix[feature_name].isna().any():
            feature_matrix[feature_name] = feature_matrix[feature_name].apply(
                lambda x: x if x is not np.nan else zero_fill)
    elif isinstance(next(iter(features.values())), np.ndarray):
        feature_matrix[feature_name] = feature_matrix["ENSG"].map(features)
        if feature_name == 'sequence':
            array_length = feature_matrix['sequence'].dropna().iloc[0].size
            feature_matrix['sequence'] = feature_matrix['sequence'].apply(
                lambda x: x if isinstance(x, np.ndarray) else np.zeros(array_length))
        else:
            zero_fill = [0] * len(next(iter(features.values())))
            feature_matrix[feature_name] = feature_matrix[feature_name].apply(
                lambda x: x if not (np.nan or x == '') else zero_fill)
    else:
        for feature, values in features.items():
            feature_matrix[feature] = feature_matrix["ENSG"].map(values)

        feature_matrix.fillna(0, inplace=True)

    ensg_ids = feature_matrix["ENSG"]
    uniprot_ids = feature_matrix["UNIPROT"]

    # feature_matrix.drop_duplicates(subset="ENSG", inplace=True)
    feature_matrix.drop(["ENSG", "UNIPROT", "variant_id"], axis=1, inplace=True)

    # utils.count_zeros(feature_matrix)

    # plot.correlation_heatmap(feature_matrix)

    return feature_matrix, ensg_ids, uniprot_ids