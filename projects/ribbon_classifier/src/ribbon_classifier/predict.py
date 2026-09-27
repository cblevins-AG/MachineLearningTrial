"""Use a trained run on new photos."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .dataset import build_transforms, load_image
from .device import pick_device
from .localize import RibbonLocalizer
from .model import load_run


class Predictor:
    """Load once, predict many: ``Predictor(run_dir).predict("photo.jpg")``.

    If the model was trained on ribbon crops, pass a :class:`RibbonLocalizer`
    so new photos are cropped the same way (train and inference must match).
    """

    def __init__(self, run_dir: str | Path, device: str = "auto", localizer: RibbonLocalizer | None = None):
        self.device = pick_device(device)
        self.model, self.meta = load_run(run_dir, self.device)
        self.classes: list[str] = self.meta["classes"]
        self.transform = build_transforms(self.meta["image_size"], tuple(self.meta["mean"]), tuple(self.meta["std"]))
        self.localizer = localizer

    @torch.no_grad()
    def predict(self, image: str | Path | Image.Image, top_k: int = 3) -> dict:
        source = str(image) if not isinstance(image, Image.Image) else "<image>"
        if not isinstance(image, Image.Image):
            image = load_image(image)
        box = None
        if self.localizer is not None:
            image, box = self.localizer.crop(image)
        logits = self.model(self.transform(image).unsqueeze(0).to(self.device))
        probs = torch.softmax(logits.float(), dim=1)[0].cpu()
        values, indices = probs.topk(min(top_k, len(self.classes)))
        return {
            "image": source,
            "prediction": self.classes[int(indices[0])],
            "confidence": round(float(values[0]), 4),
            "top_k": [
                {"label": self.classes[int(i)], "probability": round(float(v), 4)}
                for v, i in zip(values, indices, strict=True)
            ],
            "ribbon_box": None if box is None else [box.xmin, box.ymin, box.xmax, box.ymax],
        }
