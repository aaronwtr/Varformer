"""Thin launcher: argparse -> Varformer.from_pretrained(...).predict(...) -> JSON."""
import argparse
import json
import sys
from pathlib import Path

from varformer import Varformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--population", required=True, choices=["nfe", "elgh", "afr", "amr"])
    p.add_argument("--seed", default="best", help='int, "best", or "ensemble"')
    p.add_argument("--genes-file", required=True, help="text file, one Ensembl ID per line")
    p.add_argument("--out", default="-", help="JSON output path or '-' for stdout")
    p.add_argument("--return-attention", action="store_true")
    args = p.parse_args()

    seed = args.seed
    if seed not in ("best", "ensemble"):
        seed = int(seed)

    genes = Path(args.genes_file).read_text().splitlines()
    model = Varformer.from_pretrained(args.population, seed=seed)
    result = model.predict(genes=genes, return_attention=args.return_attention)

    # Make JSON-serializable: numpy arrays -> lists.
    for gid, payload in result.items():
        for k in ("z_var", "attn_weights"):
            if k in payload and hasattr(payload[k], "tolist"):
                payload[k] = payload[k].tolist()

    out = json.dumps(result, indent=2)
    if args.out == "-":
        sys.stdout.write(out)
    else:
        Path(args.out).write_text(out)


if __name__ == "__main__":
    main()
