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
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
# andrena: DefCpuPerGPU=12, MaxMemPerCPU=7680M. Match the cluster default
# exactly. 12 × 7G = 84G total RAM — well above the OOM at 32G.

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
