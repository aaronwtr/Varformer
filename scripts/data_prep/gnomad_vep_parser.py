"""Optimized gnomAD VEP annotation parser for population variant data."""
import pandas as pd
import pysam
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
from typing import List, Dict, Optional
import pyarrow as pa
import pyarrow.parquet as pq


class OptimizedGnomadVepParser:
    def __init__(self, vcf_path: str):
        self.vcf_path = vcf_path
        self.vcf = pysam.VariantFile(vcf_path)
        self.vep_fields = self._get_vep_fields()
        self.gene_dict = {}  # Incremental gene dictionary
        self.af_fields = self._get_af_fields()

        # Pre-compute categorical mappings for memory efficiency
        self.consequence_categories = set()
        self.chrom_categories = set()

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

    def _standardize_gene_id(self, gene_id: str, symbol: str) -> str:
        """Incrementally build gene dictionary and standardize a single gene ID."""
        if pd.isna(gene_id) or pd.isna(symbol):
            return gene_id

        # If already ENSG, return as-is
        if str(gene_id).startswith('ENSG'):
            # Update dictionary if we have a symbol mapping
            if symbol and symbol not in self.gene_dict:
                self.gene_dict[symbol] = gene_id
            return gene_id

        # If we have this symbol in our dictionary, use the ENSG ID
        if symbol in self.gene_dict:
            return self.gene_dict[symbol]

        # Otherwise return the original gene_id
        return gene_id

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency and faster I/O."""
        df_opt = df.copy()

        # Convert only low-cardinality string columns to categorical
        # Avoid columns that might have too many unique values for PyArrow
        safe_categorical_cols = ['chrom', 'filter', 'vep_Consequence', 'vep_IMPACT', 'vep_BIOTYPE']

        for col in safe_categorical_cols:
            if col in df_opt.columns:
                # Only convert to categorical if reasonable number of unique values
                unique_count = df_opt[col].nunique()
                if unique_count < 100:  # Conservative limit for PyArrow compatibility
                    df_opt[col] = df_opt[col].astype('category')

        # Optimize integer columns
        int_cols = ['pos', 'AC', 'AN', 'nhomalt']
        for col in int_cols:
            if col in df_opt.columns:
                df_opt[col] = pd.to_numeric(df_opt[col], errors='coerce', downcast='integer')

        # Optimize float columns
        float_cols = ['qual'] + self.af_fields
        for col in float_cols:
            if col in df_opt.columns:
                df_opt[col] = pd.to_numeric(df_opt[col], errors='coerce', downcast='float')

        return df_opt

    def _get_total_variants(self) -> int:
        """Get total number of variants for progress bar."""
        try:
            # Try to get from index if available
            return sum(1 for _ in self.vcf.fetch())
        except:
            # Fallback: estimate based on file size (very rough)
            file_size = Path(self.vcf_path).stat().st_size
            return max(1000, file_size // 10000)  # Rough estimate

    def parse_variants_streaming(self, output_path: str, chunksize: int = 5000):
        """Parse variants with streaming output to avoid memory issues."""

        # Get total variants for progress bar
        print("Counting variants for progress tracking...")
        self.vcf.close()
        self.vcf = pysam.VariantFile(self.vcf_path)  # Reopen
        total_variants = self._get_total_variants()

        # Reopen for actual processing
        self.vcf.close()
        self.vcf = pysam.VariantFile(self.vcf_path)

        records = []
        chunks = []  # Store chunks for final concatenation
        missense_count = 0
        total_processed = 0
        chunk_num = 0

        with tqdm(total=total_variants, desc="Processing variants") as pbar:
            for variant in self.vcf.fetch():
                total_processed += 1
                pbar.update(1)

                # Base record for this variant
                base_record = {
                    'chrom': variant.chrom,
                    'pos': variant.pos,
                    'id': variant.id,
                    'ref': variant.ref,
                    'alt': ','.join(str(a) for a in variant.alts),
                    'qual': variant.qual,
                    'filter': ','.join(variant.filter.keys()) if variant.filter else 'PASS'
                }

                # Add standard INFO fields
                for field in ['AC', 'AN', 'nhomalt']:
                    if field in variant.info:
                        value = variant.info[field]
                        base_record[field] = value[0] if isinstance(value, tuple) else value

                # Add allele frequency fields
                for af_field in self.af_fields:
                    if af_field in variant.info:
                        values = variant.info[af_field]
                        base_record[af_field] = values[0] if isinstance(values, tuple) else values

                # Process VEP annotations
                if 'vep' in variant.info:
                    for vep_string in variant.info['vep']:
                        vep_values = vep_string.split('|')

                        # Create record with VEP annotations
                        vep_record = base_record.copy()
                        for field, value in zip(self.vep_fields, vep_values):
                            vep_record[f'vep_{field}'] = value if value else None

                        # Standardize gene ID incrementally
                        if 'vep_Gene' in vep_record and 'vep_SYMBOL' in vep_record:
                            vep_record['vep_Gene'] = self._standardize_gene_id(
                                vep_record['vep_Gene'],
                                vep_record['vep_SYMBOL']
                            )

                        # Filter for missense variants early
                        consequence = vep_record.get('vep_Consequence', '')
                        if consequence and 'missense_variant' in consequence:
                            records.append(vep_record)
                            missense_count += 1

                # Process chunk when it reaches the specified size
                if len(records) >= chunksize:
                    df_chunk = pd.DataFrame(records)
                    df_chunk = self._optimize_dtypes(df_chunk)
                    chunks.append(df_chunk)

                    records = []
                    chunk_num += 1

                    # Update progress description
                    pbar.set_description(f"Processing variants (Found {missense_count:,} missense)")

        # Process final chunk if any records remain
        if records:
            df_chunk = pd.DataFrame(records)
            df_chunk = self._optimize_dtypes(df_chunk)
            chunks.append(df_chunk)

        # Concatenate all chunks and save
        if chunks:
            print(f"\nConcatenating {len(chunks)} chunks...")
            final_df = pd.concat(chunks, ignore_index=True)

            # Handle problematic columns for parquet
            for col in final_df.columns:
                if final_df[col].dtype == 'object':
                    # Convert object columns to string to avoid null issues
                    final_df[col] = final_df[col].astype('string')

            print("Saving to parquet...")
            final_df.to_parquet(output_path, index=False, engine='pyarrow')
        else:
            # Create empty file if no data
            pd.DataFrame().to_parquet(output_path, index=False, engine='pyarrow')

        print(f"\nProcessing Complete!")
        print(f"Total variants processed: {total_processed:,}")
        print(f"Missense variants found: {missense_count:,}")
        print(f"Gene dictionary entries: {len(self.gene_dict):,}")
        print(f"Output saved to: {output_path}")

        return missense_count


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Extract missense variants from gnomAD VCF file (optimized version)',
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
        help='Output parquet file path (.parquet)'
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
    if not args.output.endswith('.parquet'):
        raise ValueError("Output file must have .parquet extension")

    # Check if pyarrow is available
    try:
        import pyarrow
    except ImportError:
        raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow")

    parser = OptimizedGnomadVepParser(args.input)
    missense_count = parser.parse_variants_streaming(args.output, chunksize=args.chunksize)

    print(f"\nSummary: {missense_count:,} missense variants saved to {args.output}")


if __name__ == "__main__":
    main()
