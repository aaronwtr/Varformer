#!/bin/bash
#SBATCH -J varformer_benchmark
#SBATCH -o benchmark.%j.o
#SBATCH -e benchmark.%j.e
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH --constraint=80G
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00

set -e

REPO=/gpfs/scratch/bty174/globus/varformer
cd "$REPO"
source .venv/bin/activate

MODE="${1:-compare}"

if [ "$MODE" = "generate" ]; then
    python benchmark/generate_reference.py --populations nfe sas
elif [ "$MODE" = "compare" ]; then
    python benchmark/compare.py --populations nfe sas
else
    echo "Unknown mode: $MODE" >&2
    exit 1
fi
