#!/bin/bash
# Varformer benchmark — andrena (pilot_andrena) variant.
# Same workload as run_benchmark.sh; submitted to a different partition that
# typically clears the queue faster. Use whichever returns a GPU sooner.
#
#   sbatch benchmark/run_benchmark_andrena.sh generate   # one-time reference write
#   sbatch benchmark/run_benchmark_andrena.sh compare    # post-refactor verify
#
#SBATCH -J varformer_benchmark
#SBATCH -o benchmark.%j.o
#SBATCH -e benchmark.%j.e
#SBATCH -p andrena
#SBATCH -A pilot_andrena
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00

set -e

REPO=/gpfs/scratch/bty174/globus/varformer
cd "$REPO"
source .venv/bin/activate

MODE="${1:-compare}"

if [ "$MODE" = "generate" ]; then
    python benchmark/generate_reference.py --populations nfe elgh
elif [ "$MODE" = "compare" ]; then
    python benchmark/compare.py --populations nfe elgh
else
    echo "Unknown mode: $MODE" >&2
    exit 1
fi
