from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd

import torch
import open_clip

def embed_text(q: str, device: str) -> np.ndarray:
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer([q]).to(device)
    with torch.inference_mode():
        feat = model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.detach().cpu().numpy().astype(np.float32)  # (1, D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", default="data/gold/index/clip_vit_b32_openai")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import faiss

    index_path = os.path.join(args.index_dir, "index.faiss")
    map_path = os.path.join(args.index_dir, "mapping.parquet")

    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path)
    if not os.path.exists(map_path):
        raise FileNotFoundError(map_path)

    index = faiss.read_index(index_path)
    mapping = pd.read_parquet(map_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = embed_text(args.q, device)
    faiss.normalize_L2(v)

    scores, ids = index.search(v, args.k)
    for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
        row = mapping.iloc[int(i)]
        print(f"{rank:02d} score={float(s):.4f} media_id={row.media_id} path={row.local_path}")

if __name__ == "__main__":
    main()
