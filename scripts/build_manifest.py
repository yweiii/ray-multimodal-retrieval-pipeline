# scripts/build_manifest.py
import argparse
import hashlib
from pathlib import Path

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw_images")
    ap.add_argument("--out", default="data/manifests/images.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    rows = []
    for p in files:
        h = sha256_file(p)
        rows.append(
            {
                "media_id": h,  # stable idempotency key
                "sha256": h,
                "local_path": str(p),
            }
        )

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"Wrote manifest: {out} rows={len(df)}")


if __name__ == "__main__":
    main()
