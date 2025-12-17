# src/embed/run_embed_ray.py
from __future__ import annotations

import argparse
import os
from datetime import date

import pandas as pd
import ray
from ray.data import ActorPoolStrategy

from src.embed.clip_actor import ClipEmbedder


class EmbedUDF:
    def __init__(self, model_version="clip_vit_b32_openai"):
        self.model_version = model_version
        self.embedder = ClipEmbedder(model_name="ViT-B-32", pretrained="openai")

    def __call__(self, batch: dict) -> dict:
        paths = batch["local_path"]
        vecs = self.embedder.embed_images(paths)  # (B, D)
        return {
            "media_id": batch["media_id"],
            "sha256": batch["sha256"],
            "local_path": batch["local_path"],
            "model_version": [self.model_version] * len(paths),
            "embedding": list(vecs),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/images.parquet")
    ap.add_argument("--out_root", default="data/silver/embeddings")
    ap.add_argument("--model_version", default="clip_vit_b32_openai")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--actors", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.manifest)
    if args.limit and args.limit > 0:
        df = df.iloc[: args.limit].copy()

    dt = date.today().isoformat()
    out_path = os.path.join(
        args.out_root, f"model_version={args.model_version}", f"dt={dt}"
    )
    os.makedirs(out_path, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    ds = ray.data.from_pandas(df)

    ds_emb = ds.map_batches(
        EmbedUDF,
        compute=ActorPoolStrategy(size=args.actors),
        batch_size=args.batch_size,
        batch_format="numpy",
    )

    ds_emb.write_parquet(out_path)
    print(f"Wrote embeddings to: {out_path}")


if __name__ == "__main__":
    main()
