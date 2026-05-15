# Regression Benchmark

Locks model inference behaviour during the refactor. Every refactor phase must pass `compare.py` against the references stored here.

## Files

- `inputs/{pop}_genes.txt` — frozen gene lists (Ensembl IDs, one per line)
- `reference/{pop}/seed{N}.pkl` — frozen reference outputs per (population, seed): `{gene_id: {prediction, classification, z_var, attn_weights}}`
- `generate_reference.py` — one-time script that writes `reference/` using the current code. **Do not re-run after Phase 0.**
- `compare.py` — runs current package code and compares to `reference/`. Pass = `max_abs_diff < 1e-5` on prediction / attn_weights, exact classification, `max_rel_diff < 1e-3` on z_var.
- `run_benchmark.sh` — SLURM script for the sae partition. Runs `compare.py`.

## How to run after a refactor phase

```bash
# On HPC, after pulling the latest commit:
sbatch benchmark/run_benchmark.sh
# Watch the queue:
squeue -u $USER
# When done, check the .o file for PASS/FAIL.
```
