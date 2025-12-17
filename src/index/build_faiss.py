from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embeddings_glob",
        default="data/silver/embeddings/model_version=clip_vit_b32_openai/dt=*/**/*.parquet",
    )
    ap.add_argument("--out_dir", default="data/gold/index/clip_vit_b32_openai")
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    import faiss

    paths = glob.glob(args.embeddings_glob, recursive=True)
    if not paths:
        raise RuntimeError(f"No parquet found for glob: {args.embeddings_glob}")

    dfs = [
        pd.read_parquet(p, columns=["media_id", "local_path", "embedding"])
        for p in paths
    ]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["media_id"], keep="first").reset_index(drop=True)

    if args.max_rows and args.max_rows > 0:
        df = df.iloc[: args.max_rows].copy()

    X = np.stack(df["embedding"].to_numpy()).astype(np.float32)
    faiss.normalize_L2(X)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(X)

    os.makedirs(args.out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out_dir, "index.faiss"))

    df[["media_id", "local_path"]].reset_index(drop=True).to_parquet(
        os.path.join(args.out_dir, "mapping.parquet"), index=False
    )

    print(f"Built FAISS index: N={X.shape[0]} D={d} -> {args.out_dir}")


if __name__ == "__main__":
    main()
