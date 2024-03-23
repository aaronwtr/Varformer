import pandas as pd
import numpy as np
from tqdm import tqdm
import os

IN_DIR = "../data/elgh/gh_parts/raw_vcfs"
OUT_DIR = "../data/elgh/gh_parts/processed_gh_data/all_csqs/"

csq_of_interest = ['splice_acceptor_variant', 'splice_donor_variant', 'start_lost', 'stop_lost', 'stop_gained',
                   'missense_variant', 'inframe_insertion', 'inframe_deletion', 'frameshift_variant', ]

file_names = os.listdir(IN_DIR)

for file_name in file_names:
    if not os.listdir(OUT_DIR):
        with open(f'{IN_DIR}/{file_name}', 'r', encoding='Windows-1252') as file:
                content = file.readlines()

        content = content[1:]

        data = []
        for line in content:
            fields = line.strip().split('\t')
            row = {
                'CHROM': fields[0],
                'POS': fields[1],
                'ID': fields[2],
                'REF': fields[3],
                'ALT': fields[4],
                'QUAL': fields[5],
                'FILTER': fields[6],
                'INFO': fields[7]
            }
            data.append(row)

        df = pd.DataFrame(data)

        info_pairs = df['INFO'].str.split(';')

        info_dict = {}

        for i, pairs in enumerate(info_pairs):
            row_info = {}
            for pair in pairs:
                key, value = pair.split('=')
                row_info[key] = value
            info_dict[i] = row_info

        info_df = pd.DataFrame.from_dict(info_dict, orient='index')

        result_df = pd.concat([df.drop(columns=['INFO']), info_df], axis=1)
        result_df = result_df.iloc[:result_df.shape[0], :]

        def filter_csq_optimized(row):
            return row['CSQ'].split('|')[1] in csq_of_interest  # Remove the row

        filtered_df = result_df[result_df.apply(filter_csq_optimized, axis=1)]

        cols_to_keep_1 = ["CHROM", "POS", "REF", "ALT", "AF", "CSQ"]
        cols_to_keep_csq = ["Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature", "HGVSp", "UNIPARC",
                            "SWISSPROT", "TREMBL", "Protein_position", "Amino_acids", "PHENOTYPES", "Conservation",
                            "LoF", "LoF_filter", "LoF_flags", "LoF_info", "REVEL"]

        columns = "Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|ALLELE_NUM|DISTANCE|STRAND|FLAGS|VARIANT_CLASS|SYMBOL_SOURCE|HGNC_ID|CANONICAL|MANE_SELECT|MANE_PLUS_CLINICAL|TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|UNIPROT_ISOFORM|SOURCE|GENE_PHENO|SIFT|PolyPhen|DOMAINS|miRNA|HGVS_OFFSET|AF|AFR_AF|AMR_AF|EAS_AF|EUR_AF|SAS_AF|AA_AF|EA_AF|gnomAD_AF|gnomAD_AFR_AF|gnomAD_AMR_AF|gnomAD_ASJ_AF|gnomAD_EAS_AF|gnomAD_FIN_AF|gnomAD_NFE_AF|gnomAD_OTH_AF|gnomAD_SAS_AF|MAX_AF|MAX_AF_POPS|CLIN_SIG|SOMATIC|PHENO|PUBMED|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|MOTIF_SCORE_CHANGE|TRANSCRIPTION_FACTORS|SpliceRegion|GeneSplicer|existing_InFrame_oORFs|existing_OutOfFrame_oORFs|existing_uORFs|five_prime_UTR_variant_annotation|five_prime_UTR_variant_consequence|CADD_PHRED|CADD_RAW|Ensembl_transcriptid|LRT_pred|MutationTaster_pred|Polyphen2_HDIV_pred|Polyphen2_HVAR_pred|SIFT_pred|Uniprot_acc|VEP_canonical|DisGeNET_PMID|DisGeNET_SCORE|PHENOTYPES|Conservation|LoF|LoF_filter|LoF_flags|LoF_info|REVEL|SpliceAI_pred_DP_AG|SpliceAI_pred_DP_AL|SpliceAI_pred_DP_DG|SpliceAI_pred_DP_DL|SpliceAI_pred_DS_AG|SpliceAI_pred_DS_AL|SpliceAI_pred_DS_DG|SpliceAI_pred_DS_DL|SpliceAI_pred_SYMBOL|EVE_CLASS|EVE_SCORE|PrimateAI|ClinVar|ClinVar_CLNDN|ClinVar_CLNDNINCL|ClinVar_CLNDISDB|ClinVar_CLNDISDBINCL|ClinVar_AF_ESP|ClinVar_AF_EXAC|ClinVar_AF_TGP|ClinVar_ALLELEID|ClinVar_CLNDN|ClinVar_CLNDNINCL|ClinVar_CLNDISDB|ClinVar_CLNDISDBINCL|ClinVar_CLNHGVS|ClinVar_CLNREVSTAT|ClinVar_CLNSIG|ClinVar_CLNSIGCONF|ClinVar_CLNSIGINCL|ClinVar_CLNVC|ClinVar_CLNVCSO|ClinVar_CLNVI|ClinVar_DBVARID|ClinVar_GENEINFO|ClinVar_MC|ClinVar_ORIGIN|ClinVar_RS"
        column_names = columns.split('|')

        cols_to_keep_csq = [col for col in column_names if col in cols_to_keep_csq]

        header = cols_to_keep_1[:-1] + cols_to_keep_csq
        csq_col_idx = list(enumerate(column_names))
        rm_cols_idx = [idx for idx, col in csq_col_idx if col not in cols_to_keep_csq]

        with open(f'{OUT_DIR}/proc_{file_name}', 'w') as f:
            f.write('\t'.join(header) + '\n')
            filtered_df = filtered_df[cols_to_keep_1]

            for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
                split_data = row['CSQ'].split('|')
                num_columns = len(cols_to_keep_csq)

                split_data = [x for i, x in enumerate(split_data) if i % len(column_names) not in rm_cols_idx]

                split_data = [x if x != '' else np.nan for x in split_data]

                num_chunks = len(split_data) // num_columns

                row_without_csq = row.drop('CSQ').values.tolist()

                row_without_csq = list(row_without_csq)
                for i in range(num_chunks):
                    start = i * num_columns
                    end = (i + 1) * num_columns
                    chunk = split_data[start:end]

                    row_data = row_without_csq + chunk

                    # Write the row to the file as a tab-delimited line
                    f.write('\t'.join(map(str, row_data)) + '\n')
            f.close()

txt_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.txt')]

dfs = []

for txt_file in tqdm(txt_files):
    df = pd.read_csv(os.path.join(OUT_DIR, txt_file), sep='\t', low_memory=False)
    df = df[df['Consequence'].apply(lambda x: any(csq in x for csq in csq_of_interest))]
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_pickle(f'../data/elgh/gh_parts/processed_gh_data/filtered_csqs.pkl')
