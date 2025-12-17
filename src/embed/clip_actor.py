# src/embed/clip_actor.py
from __future__ import annotations

from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image


class ClipEmbedder:
    def __init__(
        self, model_name="ViT-B-32", pretrained="openai", device: str | None = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def embed_images(self, paths: List[str]) -> np.ndarray:
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            imgs.append(self.preprocess(img))
        batch = torch.stack(imgs).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.detach().cpu().numpy().astype(np.float32)
