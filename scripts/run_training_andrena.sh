#!/bin/bash
# Varformer training — single (population, seed) run on Apocrita.
#
# Submitted to andrena (pilot_andrena).  Same workload as
# ``run_training.sh`` but targeting the andrena queue, which typically
# clears faster than sae.  Use whichever queue is shorter at
# submission time.
#
#   sbatch scripts/run_training_andrena.sh                  # nfe / 42 / checkpoints/repro
#   sbatch scripts/run_training_andrena.sh sas 7            # sas / 7   / checkpoints/repro
#   sbatch scripts/run_training_andrena.sh nfe 42 prod      # nfe / 42  / checkpoints/prod
#
#SBATCH -J varformer_train
#SBATCH -o varformer_train.%j.o
#SBATCH -e varformer_train.%j.e
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:1
#SBATCH -t 36:00:00
# andrena: DefCpuPerGPU=12, MaxMemPerCPU=7680M.  Match the cluster default
# exactly — 12 × 7G = 84G total RAM (well above the 60G training peak).

set -euo pipefail

POPULATION="${1:-nfe}"
SEED="${2:-42}"
OUT_SUBDIR="${3:-repro}"

REPO=/gpfs/scratch/bty174/globus/varformer
cd "$REPO"
source .venv/bin/activate

OUT_DIR="./checkpoints/${OUT_SUBDIR}/"
mkdir -p "$OUT_DIR"

echo "=== Varformer training (andrena) ==="
echo "  population : $POPULATION"
echo "  seed       : $SEED"
echo "  out dir    : $OUT_DIR"
echo "  node       : $(hostname)"
echo "  gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "===================================="

python scripts/run_training.py \
    --population "$POPULATION" \
    --seeds "$SEED" \
    --output-dir "$OUT_DIR"

echo "=== Training complete ==="
ls -la "${OUT_DIR}${POPULATION}/" 2>/dev/null || ls -la "$OUT_DIR"
