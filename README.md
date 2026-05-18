# Varformer

A multimodal transformer framework for **exome-wide prioritisation of population-informed drug targets**. Varformer integrates gene-centric multiomics features with population-scale genetic variation through a cross-modal attention mechanism that autonomously learns representations from raw, variable-length sets of missense variants — addressing both label sparsity and population-specific genetic architecture in target discovery.

Trained across four ancestrally diverse populations: African (AFR), Admixed American (AMR), Non-Finnish European (NFE), and South Asian (SAS, via the Genes & Health cohort). Achieves Spearman correlations up to **ρ = 0.62** on disease-agnostic target prioritisation benchmarks.

> Paper: *Varformer: A Multimodal Transformer Model for the Exome-Wide Prioritisation of Population-Informed Drug Targets.* Wenteler, Cabrera, Wei, Neduva, Barnes.

---

## Installation

```bash
git clone https://github.com/aaronwtr/Varformer.git
cd Varformer
uv sync          # or: pip install -e .
```

Requires Python 3.9 and a CUDA-capable GPU for inference at scale (CPU works for small batches).

## Quickstart

```python
from varformer import Varformer

# Load a published checkpoint
model = Varformer.from_pretrained("nfe", seed="best")

# Predict tractability for a list of Ensembl gene IDs
predictions = model.predict(
    genes=["ENSG00000141510", "ENSG00000139618"],
    return_attention=False,
)
for gene_id, payload in predictions.items():
    print(gene_id, payload["prediction"], payload["classification"])
```

### Ensemble inference

Average predictions across all 5 seeds for a population:

```python
ensemble = Varformer.from_pretrained("nfe", seed="ensemble")
preds = ensemble.predict(genes=[...])
```

### Evaluation on labelled holdout

```python
metrics = model.evaluate(test_set="pfam")  # or "rcnt" | "pharos"
print(metrics["auroc"], metrics["auprc"], metrics["spearman"])
```

### Training a new model

```python
trainer = Varformer.trainer(
    population="elgh",                            # see "Population labels" below
    config_overrides={"epochs": 50, "lr_start": 3e-5},
    output_dir="./checkpoints/",
)
checkpoint_paths = trainer.fit(seeds=[7, 42, 85])
```

---

## Model

Varformer is a multimodal architecture combining gene-centric features with population variant context:

- **Gene Characterisation (GC) branch** — projects per-gene multiomics features (Open Targets, PPI, gene essentiality, mouse-KO) through an MLP.
- **Gene Ontology (GO) branch** — projects GO term-derived features through an MLP.
- **Variant branch (`VariantEncoder`)** — a small Transformer encoder over the missense variants of a gene, encoding each variant by its AlphaMissense pathogenicity score, protein position, and mutation type.
- **Gene-Variant cross-attention** — gene features attend over variant embeddings to produce a variant-informed gene representation `z_var`.
- **Classification head** — concatenates the gene representation with `z_var` and predicts a clinical-success score.

**Loss.** Drug-target labels are positive-only — clinical-trial failure is not a reliable negative signal — so Varformer treats the problem as positive-unlabeled learning. We use the **non-negative risk estimator (nnPU)**, a biased-PUL objective that directly estimates classification risk from positive and unlabeled examples without requiring explicit negatives. nnPU yields more stable learning than two-step PUL methods under extreme label sparsity.

## Available checkpoints

| Code label | Cohort | Seeds |
|---|---|---|
| `nfe` | Non-Finnish European (gnomAD) | 42, 85, 482, 589, 612 |
| `elgh` | **South Asian** — Genes & Health (Bangladeshi + Pakistani), N=44,288 | 7, 32, 57, 64, 482 |
| `afr` | African (gnomAD) | (per release) |
| `amr` | Admixed American (gnomAD) | (per release) |

**Population labels.** The internal code label `elgh` corresponds to the **SAS (South Asian)** cohort in the paper, sourced from the East London Genes & Health study. All other populations come from gnomAD. Checkpoints live under `checkpoints/<label>/seed{N}-epoch=*-val_spearman=*.ckpt`.

## Repository layout

```
varformer/         # the importable package
  models/          # VariantEncoder, GeneVariantAttention, Varformer (architecture)
  training/        # VarformerLightningModule, train, tune, callbacks
  inference/       # predict, evaluate, load
  data/            # features/, parsers/, datasets, samplers, pipeline, loaders, splits
  utils/           # seeding, aa_codes, gene_id
  config.py        # Pydantic Config + Hyperparameters
configs/           # default.yml + paths/{hpc,local}.yml
benchmark/         # regression benchmark — reference predictions + compare.py
tests/             # 38 unit tests, CPU-only
scripts/           # run_training.py, run_inference.py launchers
paper/baselines/   # paper baseline trainers (LR, random, DrugnomeAI)
checkpoints/       # trained model checkpoints (gitignored)
```

## Reproducibility

Inference behaviour is gated by `benchmark/compare.py`, which checks every change against frozen reference predictions in `benchmark/reference/`. Tolerances: `max_abs_diff < 1e-5` on predictions and attention weights, `max_rel_diff < 1e-3` on intermediate `z_var`. See `benchmark/README.md` for details.

To re-run the benchmark on a SLURM cluster:

```bash
sbatch benchmark/run_benchmark_andrena.sh compare
```

## Tests

```bash
pytest tests/
```

38 CPU-only unit tests covering model shapes, attention math, config validation, checkpoint legacy-key stripping, and SDK contracts.

## Citation

If you use Varformer in your research, please cite:

```bibtex
@article{wenteler2026varformer,
  title   = {Varformer: A Multimodal Transformer Model for the Exome-Wide Prioritisation of Population-Informed Drug Targets},
  author  = {Wenteler, Aaron and Cabrera, Claudia P. and Wei, W. and Neduva, V. and Barnes, Michael R.},
  year    = {2026},
  journal = {TBD},
  url     = {https://github.com/aaronwtr/Varformer}
}
```

(Replace `journal = {TBD}` and add `doi = {...}` once the venue is finalised.)

## License

MIT — see `LICENSE`.

## Contact

A. Wenteler — <a.wenteler@qmul.ac.uk>
