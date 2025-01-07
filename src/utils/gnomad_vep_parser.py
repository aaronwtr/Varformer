import pandas as pd
import pysam
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
from typing import List, Dict, Optional


class GnomadVepParser:
    def __init__(self, vcf_path: str):
        self.vcf_path = vcf_path
        self.vcf = pysam.VariantFile(vcf_path)
        self.vep_fields = self._get_vep_fields()
        self.gene_dict = {}

    def _get_vep_fields(self) -> List[str]:
        for record in self.vcf.header.records:
            if record.type == "INFO" and record.get("ID") == "vep":
                return record.get("Description", "").split(":")[-1].strip().split("|")
        raise ValueError("No VEP annotation found in header")

    def _get_af_fields(self) -> List[str]:
        """Get all allele frequency related fields from the VCF header."""
        af_fields = []
        for record in self.vcf.header.records:
            if record.type == "INFO":
                field_id = record.get("ID")
                if field_id and (field_id.startswith("AF_") or field_id in ["AF", "AF_male", "AF_female"]):
                    af_fields.append(field_id)
        return af_fields

    def _build_gene_dict(self, df: pd.DataFrame) -> Dict[str, str]:
        """Build dictionary mapping gene symbols to ENSG IDs."""
        gene_dict = {}
        for _, row in df.iterrows():
            if pd.notna(row['vep_Gene']) and pd.notna(row['vep_SYMBOL']):
                if str(row['vep_Gene']).startswith('ENSG'):
                    gene_dict[row['vep_SYMBOL']] = row['vep_Gene']
        return gene_dict

    def _standardize_gene_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all gene entries to ENSG format using the gene dictionary."""
        if not self.gene_dict:
            self.gene_dict = self._build_gene_dict(df)

        df_standardized = df.copy()

        mask = ~df_standardized['vep_Gene'].str.startswith('ENSG', na=False)
        df_standardized.loc[mask, 'vep_Gene'] = df_standardized.loc[mask, 'vep_SYMBOL'].map(self.gene_dict)

        total_genes = len(df_standardized)
        converted_genes = mask.sum()
        missing_genes = df_standardized['vep_Gene'].isna().sum()

        print(f"\nGene ID Standardization Summary:")
        print(f"Total entries: {total_genes}")
        print(f"Converted to ENSG: {converted_genes}")
        print(f"Missing ENSG mappings: {missing_genes}")
        print(f"Unique ENSG IDs: {df_standardized['vep_Gene'].nunique()}")

        return df_standardized

    def parse_variants(self, chunksize: int = 5000) -> pd.DataFrame:
        chunks = []
        records = []
        af_fields = self._get_af_fields()

        for variant in tqdm(self.vcf.fetch()):
            record = {
                'chrom': variant.chrom,
                'pos': variant.pos,
                'id': variant.id,
                'ref': variant.ref,
                'alt': ','.join(str(a) for a in variant.alts),
                'qual': variant.qual,
                'filter': ','.join(variant.filter.keys()) if variant.filter else 'PASS'
            }

            for field in ['AC', 'AN', 'nhomalt']:
                if field in variant.info:
                    record[field] = variant.info[field][0] if isinstance(variant.info[field], tuple) else variant.info[
                        field]

            for af_field in af_fields:
                if af_field in variant.info:
                    values = variant.info[af_field]
                    if isinstance(values, tuple):
                        record[af_field] = values[0]
                    else:
                        record[af_field] = values

            if 'vep' in variant.info:
                for vep_string in variant.info['vep']:
                    vep_record = record.copy()
                    vep_values = vep_string.split('|')
                    for field, value in zip(self.vep_fields, vep_values):
                        vep_record[f'vep_{field}'] = value if value else None
                    records.append(vep_record)

                if len(records) >= chunksize:
                    chunks.append(pd.DataFrame(records))
                    records = []

        if records:
            chunks.append(pd.DataFrame(records))

        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        numeric_cols = ['pos', 'qual', 'AC', 'AN', 'nhomalt']
        numeric_cols.extend(af_fields)

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = self._standardize_gene_ids(df)

        return df


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract missense variants from gnomAD VCF file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input VCF file path (bgzipped)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output pickle file path (.pkl.gz)'
    )
    parser.add_argument(
        '--chunksize',
        type=int,
        default=5000,
        help='Number of variants to process in each chunk'
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Validate input/output paths
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.output.endswith('.pkl.gz'):
        raise ValueError("Output file must have .pkl.gz extension")

    parser = GnomadVepParser(args.input)
    df = parser.parse_variants(chunksize=args.chunksize)

    df_missense = df[df['vep_Consequence'].str.contains('missense_variant', na=False, regex=False)]
    df_missense.to_pickle(args.output, compression='gzip')

    print(f"\nFinal Summary:")
    print(f"Total variants: {len(df)}")
    print(f"Missense variants: {len(df_missense)}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()