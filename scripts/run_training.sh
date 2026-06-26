#!/bin/bash
# Varformer training — single (population, seed) run on Apocrita.
#
# Submitted to sae (pilot_sae_gpu A100 80GB).  Defaults reproduce the
# NFE / seed-42 published checkpoint and are the recommended starting
# point for a training-reproducibility test.
#
#   sbatch scripts/run_training.sh                  # nfe / 42 / checkpoints/repro
#   sbatch scripts/run_training.sh sas 7            # sas / 7   / checkpoints/repro
#   sbatch scripts/run_training.sh nfe 42 prod      # nfe / 42  / checkpoints/prod
#
#SBATCH -J varformer_train
#SBATCH -o varformer_train.%j.o
#SBATCH -e varformer_train.%j.e
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --constraint=80G
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:1
#SBATCH -t 36:00:00

set -euo pipefail

POPULATION="${1:-nfe}"
SEED="${2:-42}"
OUT_SUBDIR="${3:-repro}"

REPO=/gpfs/scratch/bty174/globus/varformer
cd "$REPO"
source .venv/bin/activate

OUT_DIR="./checkpoints/${OUT_SUBDIR}/"
mkdir -p "$OUT_DIR"

echo "=== Varformer training ==="
echo "  population : $POPULATION"
echo "  seed       : $SEED"
echo "  out dir    : $OUT_DIR"
echo "  node       : $(hostname)"
echo "  gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "=========================="

python scripts/run_training.py \
    --population "$POPULATION" \
    --seeds "$SEED" \
    --output-dir "$OUT_DIR"

echo "=== Training complete ==="
ls -la "${OUT_DIR}${POPULATION}/" 2>/dev/null || ls -la "$OUT_DIR"
