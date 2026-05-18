# Data preparation scripts

These scripts are one-off offline data-prep tools run by the user directly. They are **not** imported by the `varformer` package.

## Pipeline overview

Raw cohort VCFs (Genes & Health exomes) are pre-processed by `vcf_parser.py` into per-chunk TSV files with VEP consequence fields expanded. gnomAD population VCFs (bgzipped, VEP-annotated) are parsed by `gnomad_vep_parser.py`, which extracts missense variants into a flat parquet file. Both outputs are then filtered to the variants present in the relevant population exomes and feed into `varformer.data.features.PopulationVariantPreprocessor` (via the pre-computed `var_pat_features.pkl` cache files).

## Scripts

### gnomad_vep_parser.py

Takes a bgzipped gnomAD VCF file with VEP annotations and extracts missense variants into a flat Parquet file. Reads VEP fields from the VCF header and resolves gene IDs incrementally. Streams variants in configurable chunks to limit memory usage.

Usage:

```bash
python scripts/data_prep/gnomad_vep_parser.py \
    --input gnomad.exomes.v4.1.<pop>.vcf.bgz \
    --output gnomad_missense_<pop>.parquet \
    [--chunksize 5000]
```

Requires: `pysam`, `pyarrow`.

### vcf_parser.py

Takes raw tab-separated VCF export files (from the SAS/Genes & Health cohort) stored under `data/sas/gh_parts/raw_vcfs/`, expands the `CSQ` INFO field into individual consequence columns, filters to consequences of interest (missense, splice, frameshift, etc.), and writes per-chunk TSV files to `data/sas/gh_parts/processed_gh_data/all_csqs/`. Finally concatenates all chunks into `all_csqs_non_filtered.pkl`.

Usage: run as a script (no CLI arguments — edit `IN_DIR` / `OUT_DIR` at the top of the file if paths differ):

```bash
python scripts/data_prep/vcf_parser.py
```

Note: the input files use Windows-1252 encoding (the script accounts for this).

## Inputs / outputs

| Script | Input | Output |
|---|---|---|
| `gnomad_vep_parser.py` | `gnomad.exomes.v4.1.<pop>.vcf.bgz` | `gnomad_missense_<pop>.parquet` |
| `vcf_parser.py` | `data/sas/gh_parts/raw_vcfs/*.txt` | `data/sas/gh_parts/processed_gh_data/all_csqs/proc_*.txt` + `all_csqs_non_filtered.pkl` |
