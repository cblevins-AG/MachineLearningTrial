"""Synthetic data so the whole pipeline can be tested in seconds, offline, on a CPU.

Three fake "techniques" are drawn as different shapes/colours; a tiny,
randomly initialised timm test model (``test_convnext``) stands in for the
real backbone, so no weights are downloaded.
"""

from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

CLASSES = {
    "box_pleat": ((200, 30, 40), "rectangle"),
    "dovetail_cut": ((30, 60, 200), "triangle"),
    "loop_fold": ((30, 160, 60), "ellipse"),
}


def draw_sample(label: str, variant: int, size: int = 96) -> Image.Image:
    color, shape = CLASSES[label]
    image = Image.new("RGB", (size, size), (235, 230, 220))
    draw = ImageDraw.Draw(image)
    offset = (variant * 7) % 20
    box = (10 + offset, 15, 70 + offset // 2, 80)
    if shape == "rectangle":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(box[0], box[3]), ((box[0] + box[2]) // 2, box[1]), (box[2], box[3])], fill=color)
    else:
        draw.ellipse(box, fill=color)
    return image


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A throwaway project folder: configs/test.yaml + data/raw/examples/<label>/<group>/*.png."""
    examples = tmp_path / "data" / "raw" / "examples"
    for label in CLASSES:
        for group in range(6):  # 6 physical "ribbons" per class ...
            for shot in range(2):  # ... photographed twice each
                path = examples / label / f"roll_{group}" / f"img_{shot}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                draw_sample(label, group * 2 + shot).save(path)

    config = {
        "model": {"backbone": "test_convnext", "pretrained": False, "image_size": 64},
        "train": {"epochs": 3, "head_only_epochs": 1, "batch_size": 8, "patience": 2},
        "probe": {"backbone": "test_convnext", "pretrained": False, "image_size": 64, "batch_size": 8},
        "split": {"val_fraction": 0.2, "test_fraction": 0.2},
    }
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "test.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path
