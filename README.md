# Varformer

A multimodal transformer framework for **exome-wide prioritisation of population-informed drug targets**. Varformer integrates gene-centric multiomics features with population-scale genetic variation through a cross-modal attention mechanism that autonomously learns representations from raw, variable-length sets of missense variants, addressing both label sparsity and population-specific genetic architecture in target discovery.

Trained across four ancestrally diverse populations: African (AFR), Admixed American (AMR), Non-Finnish European (NFE), and South Asian (SAS, via the Genes & Health cohort).

## Installation

```bash
git clone https://github.com/aaronwtr/Varformer.git
cd Varformer
uv sync          # or: pip install -e .
```

Requires Python 3.9 and a CUDA-capable GPU for inference at scale (CPU works for small batches).

## Inference on a published population

```python
from varformer import Varformer

# Load the best-scoring checkpoint for a population.
model = Varformer.from_pretrained("nfe", seed="best")

# Predict tractability for a list of Ensembl gene IDs.
predictions = model.predict(
    genes=["ENSG00000141510", "ENSG00000139618"],
    return_attention=False,
)
for gene_id, payload in predictions.items():
    print(gene_id, payload["prediction"], payload["classification"])
```

Each `payload` dict contains:

- `prediction` — sigmoid probability of clinical-trial success in `[0, 1]`.
- `classification` — binary 0/1 from the trained decision threshold.
- `z_var` — the variant-informed gene embedding (NumPy array).
- `attn_weights` — per-variant attention scores (only when `return_attention=True`).

The `seed` argument accepts an integer (load that specific seed) or `"best"` (pick the highest-scoring checkpoint by `val_spearman`).

### Available checkpoints

| Code label | Cohort 
|---|---
| `nfe` | Non-Finnish European (gnomAD) 
| `sas` | South Asian — Genes & Health (Bangladeshi + Pakistani), N=44,288 
| `afr` | African (gnomAD) 
| `amr` | Admixed American (gnomAD)

Trained weights will be released on the Hugging Face Hub alongside the paper; `Varformer.from_pretrained(...)` will resolve the requested `(population, seed)` to the corresponding Hub repository and cache the checkpoint locally on first use. Until the public release the call expects checkpoints on disk at `checkpoints/<label>/seed{N}-epoch=*-val_spearman=*.ckpt`.

## Model

Following the design in the paper, Varformer is structured around two complementary modules whose representations are fused by cross-modal attention before classification:

- **Gene Characterisation (GC) module** — captures gene-centric biology independent of the population. In the implementation this is split across two parallel MLP projections: one over a curated multiomics feature set (Open Targets tractability axes, protein–protein interaction context, gene essentiality, mouse-knockout phenotypes), and one over Gene Ontology-derived features. Their outputs are concatenated into a single gene-level representation `z_gene`.
- **Population Variant Characterisation (PVC) module** — encodes the variable-length set of missense variants observed in a population for each gene. Each variant is represented by its AlphaMissense pathogenicity score, its protein position, and its mutation-type index. A small transformer encoder (`VariantEncoder`) produces contextualised variant embeddings.

`z_gene` then attends over the per-variant embeddings (`GeneVariantAttention`), producing a single variant-informed embedding `z_var`. The classification head concatenates `z_gene` with `z_var` and predicts a clinical-success score.

**Loss.** Drug-target labels are positive-only — clinical-trial failure is not a reliable negative signal — so Varformer treats the problem as positive-unlabelled (PU) learning. Training optimises the **non-negative risk estimator (nnPU)**, an unbiased classification-risk estimator that does not require explicit negatives. nnPU yields more stable learning than two-step PU methods under the extreme label sparsity that characterises drug-target identification.

## Training on a new exome-seq dataset

To train Varformer on a different population or cohort, the dataset has to be pre-processed into the same per-gene format used by the published populations:

1. **Variants.** A pickle keyed by Ensembl gene ID, where each value is the gene's missense-variant table (columns: AlphaMissense pathogenicity score, protein position, mutation-type index). The mutation-type index is taken from the shared missense map in `data/<pop>/missense_mutation_map.pkl`.
2. **Gene features.** Per-gene multiomics features (GC) and Gene Ontology features (GO), aligned to the same gene index, in the layout expected by `varformer/data/features/gc.py` and `varformer/data/features/go.py`.
3. **Labels.** Positive drug-target labels for PU training; unlabelled genes are inferred automatically.

Register the new population by adding its label to the `Population` literal in `varformer/config.py` and adding its data paths under the appropriate profile in `configs/paths/{local,hpc}.yml`.

Once the data is in place:

```python
from varformer import Varformer

trainer = Varformer.trainer(
    population="<your-label>",
    config_overrides={"epochs": 50, "lr_start": 3e-5},
    output_dir="./checkpoints/",
)
checkpoint_paths = trainer.fit(seeds=[7, 42, 85])
```

`config_overrides` accepts any field from `varformer.config.Hyperparameters`. Each seed produces an independent checkpoint under `output_dir/<your-label>/`.

## Repository layout

```
varformer/         # the importable package
  models/          # VariantEncoder, GeneVariantAttention, Varformer
  training/        # VarformerLightningModule, training loop, callbacks
  inference/       # predict + evaluate entry points
  data/            # features/, parsers/, datasets, samplers, pipeline, loaders, splits
  utils/           # seeding, aa_codes
  config.py        # Pydantic Config + Hyperparameters
configs/           # default.yml + paths/{hpc,local}.yml
scripts/           # run_training.py, run_inference.py launchers
checkpoints/       # trained model checkpoints (gitignored)
```

## Citation

```bibtex
@article{wenteler2026varformer,
  title   = {Varformer: A Multimodal Transformer Model for the Exome-Wide Prioritisation of Population-Informed Drug Targets},
  author  = {Wenteler, Aaron and Cabrera, Claudia P. and Wei, W. and Neduva, V. and Barnes, Michael R.},
  year    = {2026},
  journal = {TBD},
  url     = {https://github.com/aaronwtr/Varformer}
}
```

## License

MIT — see `LICENSE`.

## Contact

A. Wenteler — <a.wenteler@qmul.ac.uk>
