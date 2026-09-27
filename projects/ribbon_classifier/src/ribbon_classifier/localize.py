"""Find the ribbon in a photo and crop to it, with a zero-shot object detector.

"Zero-shot" means no training: OWLv2 was trained on a huge set of images and
captions, so it can locate objects described in plain words ("a ribbon bow").
Cropping first means the classifier sees ribbon pixels, not the table, hand or
background around it - which matters a lot with a small dataset, where the
model will happily learn "blue tablecloth = box pleat" if it can.

If the detector's boxes are poor on your photos, the next step up is to draw
~200 boxes in Label Studio and fine-tune a small detector (see the README).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from .config import LocalizerConfig
from .dataset import load_image


@dataclass
class Box:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    score: float


class RibbonLocalizer:
    def __init__(self, config: LocalizerConfig, device: str = "cpu", detector: Callable | None = None):
        self.config = config
        if detector is None:
            from transformers import pipeline  # heavy import; only when actually used

            detector = pipeline("zero-shot-object-detection", model=config.model, device=device)
        self._detector = detector

    def detect(self, image: Image.Image) -> Box | None:
        """Highest-scoring box for any of the prompts, or ``None`` below the threshold."""
        results = self._detector(
            image, candidate_labels=list(self.config.prompts), threshold=self.config.score_threshold
        )
        if not results:
            return None
        best = max(results, key=lambda r: r["score"])
        b = best["box"]
        return Box(int(b["xmin"]), int(b["ymin"]), int(b["xmax"]), int(b["ymax"]), float(best["score"]))

    def crop(self, image: Image.Image) -> tuple[Image.Image, Box | None]:
        """Crop to the ribbon plus a margin; returns the original image if nothing is found."""
        box = self.detect(image)
        if box is None:
            return image, None
        mx = (box.xmax - box.xmin) * self.config.margin
        my = (box.ymax - box.ymin) * self.config.margin
        left, top = max(0, int(box.xmin - mx)), max(0, int(box.ymin - my))
        right, bottom = min(image.width, int(box.xmax + mx)), min(image.height, int(box.ymax + my))
        if right - left < 8 or bottom - top < 8:
            return image, None
        return image.crop((left, top, right, bottom)), box


def crop_manifest(
    df: pd.DataFrame, data_dir: str | Path, crops_dir: str | Path, localizer: RibbonLocalizer
) -> pd.DataFrame:
    """Crop every image in *df*; returns a manifest pointing at the crops.

    Adds ``original_path``, ``crop_found`` and ``crop_score`` so you can sort by
    score and eyeball the weakest crops.
    """
    data_dir, crops_dir = Path(data_dir), Path(crops_dir)
    rows = []
    for row in tqdm(df.to_dict("records"), desc="cropping", unit="img"):
        image = load_image(data_dir / row["image_path"])
        cropped, box = localizer.crop(image)
        target = crops_dir / Path(row["image_path"]).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(target)
        rows.append(
            {
                **row,
                "original_path": row["image_path"],
                "image_path": target.resolve().relative_to(data_dir.resolve()).as_posix(),
                "crop_found": box is not None,
                "crop_score": round(box.score, 4) if box else 0.0,
            }
        )
    return pd.DataFrame(rows)
