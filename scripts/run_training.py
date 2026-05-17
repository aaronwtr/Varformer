"""Thin launcher: argparse -> Varformer.trainer(...).fit(...)."""
import argparse

from varformer import Varformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True, choices=["nfe", "elgh", "afr", "amr"])
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--output-dir", default="./checkpoints/")
    args = p.parse_args()
    paths = Varformer.trainer(population=args.population, output_dir=args.output_dir).fit(seeds=args.seeds)
    for q in paths:
        print(q)


if __name__ == "__main__":
    main()
