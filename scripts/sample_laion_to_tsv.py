# scripts/sample_laion_to_tsv.py
import argparse
import os

import pandas as pd
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="laion/aesthetics_v2_4.75")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=25000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="inputs/laion_sample.tsv")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Columns can differ slightly across LAION subsets; we handle common variants.
    candidates = []
    for row in ds.shuffle(seed=args.seed).take(
        args.n * 2
    ):  # oversample to survive missing fields
        url = row.get("url") or row.get("URL")
        caption = row.get("caption") or row.get("text") or row.get("TEXT")
        if not url or not caption:
            continue
        candidates.append({"url": url, "caption": caption})
        if len(candidates) >= args.n:
            break

    if len(candidates) < args.n:
        raise RuntimeError(
            f"Only collected {len(candidates)} rows; dataset fields may differ."
        )

    df = pd.DataFrame(candidates)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
