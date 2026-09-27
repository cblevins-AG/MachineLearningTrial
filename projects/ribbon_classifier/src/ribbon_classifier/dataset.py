"""Label manifests, image loading, validation and PyTorch datasets.

The *manifest* is a CSV with one row per labelled image::

    image_path,label,group,source
    raw/examples/dovetail_cut/roll_17/IMG_0001.jpg,dovetail_cut,roll_17,examples

* ``image_path`` - relative to the data folder, forward slashes.
* ``label``      - the class name (fold or cut technique).
* ``group``      - images that must stay together in one split, e.g. several
                   photos of the same physical ribbon. Splitting them across
                   train and test would let the model "recognise the ribbon"
                   instead of the technique, and inflate test scores.
* ``source``     - where it came from (``examples``, ``reference_pdf``, ...).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import AugmentConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
MIN_IMAGES_PER_CLASS = 10  # below this, per-class results are mostly noise

_heif_registered = False


def load_image(path: str | Path) -> Image.Image:
    """Open any supported image as RGB, rotated upright.

    Phone photos store rotation as an EXIF tag instead of rotating pixels;
    ``exif_transpose`` applies it so the model sees what a person sees.
    """
    global _heif_registered
    if not _heif_registered:
        from pillow_heif import register_heif_opener  # adds iPhone HEIC support to Pillow

        register_heif_opener()
        _heif_registered = True
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            return background
        return image.convert("RGB")


def read_manifest(csv_path: str | Path) -> pd.DataFrame:
    """Read a manifest and normalise it (trimmed labels, forward slashes, default group/source)."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = {"image_path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing column(s): {', '.join(sorted(missing))}")
    df["image_path"] = df["image_path"].str.strip().str.replace("\\", "/", regex=False)
    df["label"] = df["label"].str.strip()
    if "source" not in df.columns:
        df["source"] = "unknown"
    if "group" not in df.columns:
        df["group"] = ""
    df["group"] = df["group"].str.strip()
    df.loc[df["group"] == "", "group"] = df["image_path"]  # no group given: the image is its own group
    return df


def _file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    df: pd.DataFrame, data_dir: str | Path, open_images: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """Drop unusable rows and report problems a person should look at.

    Returns the cleaned manifest and a list of human-readable messages. Exact
    duplicate files are put into the same group so they cannot leak across
    splits.
    """
    data_dir = Path(data_dir)
    problems: list[str] = []
    df = df.copy()

    unlabelled = df["label"] == ""
    if unlabelled.any():
        problems.append(f"{int(unlabelled.sum())} row(s) have no label and were skipped.")
        df = df[~unlabelled]

    keep, hashes = [], []
    for row in df.itertuples():
        path = data_dir / row.image_path
        if not path.is_file():
            problems.append(f"Missing file: {row.image_path}")
            keep.append(False)
            hashes.append("")
            continue
        if open_images:
            try:
                load_image(path)  # decoding fails loudly on corrupt files
            except Exception as error:  # corrupt file, unsupported format, not an image
                problems.append(f"Unreadable image: {row.image_path} ({type(error).__name__})")
                keep.append(False)
                hashes.append("")
                continue
        keep.append(True)
        hashes.append(_file_hash(path))
    df = df.assign(sha1=hashes)[keep]

    # Labels that differ only by case/spacing are almost always typos.
    spellings: dict[str, set[str]] = {}
    for label in df["label"].unique():
        spellings.setdefault(label.casefold().replace(" ", "_").replace("-", "_"), set()).add(label)
    for variants in spellings.values():
        if len(variants) > 1:
            problems.append(f"Labels that look like the same class: {sorted(variants)}")

    # Identical files: same group (no leakage); conflicting labels are an error to fix.
    for _, dupes in df[df.duplicated("sha1", keep=False)].groupby("sha1"):
        df.loc[dupes.index, "group"] = dupes["group"].iloc[0]
        if dupes["label"].nunique() > 1:
            problems.append(
                f"Same image with different labels {sorted(dupes['label'].unique())}: {', '.join(dupes['image_path'])}"
            )

    for label, count in df["label"].value_counts().items():
        if count < MIN_IMAGES_PER_CLASS:
            problems.append(
                f"Class '{label}' has only {count} image(s); aim for 50+ (at least {MIN_IMAGES_PER_CLASS})."
            )
    return df.drop(columns="sha1").reset_index(drop=True), problems


def manifest_from_folders(root: str | Path, data_dir: str | Path, source: str = "examples") -> pd.DataFrame:
    """Build manifest rows from a folder tree where each top-level folder is a label::

        root/dovetail_cut/IMG_001.jpg              -> label dovetail_cut, own group
        root/dovetail_cut/roll_17/IMG_002.jpg      -> label dovetail_cut, group roll_17

    Put several photos of the same physical ribbon in one sub-folder to group them.
    """
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    rows = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            relative = path.relative_to(data_dir).as_posix()
            inner = path.relative_to(label_dir).parts
            group = f"{label_dir.name}/{inner[0]}" if len(inner) > 1 else relative
            rows.append({"image_path": relative, "label": label_dir.name, "group": group, "source": source})
    return pd.DataFrame(rows, columns=["image_path", "label", "group", "source"])


def build_transforms(
    image_size: int, mean: tuple[float, ...], std: tuple[float, ...], augment: AugmentConfig | None = None
) -> Callable:
    """Image -> normalised tensor. Pass *augment* for training, ``None`` for evaluation.

    Augmentation shows the model a slightly different version of each photo
    every epoch (crop, flip, rotation, lighting), which is the cheapest defence
    against over-fitting a small dataset.
    """
    if augment is None:
        steps = [v2.Resize(int(image_size / 0.95)), v2.CenterCrop(image_size)]
    else:
        steps = [v2.RandomResizedCrop(image_size, scale=(augment.min_crop_scale, 1.0))]
        if augment.horizontal_flip:
            steps.append(v2.RandomHorizontalFlip())
        if augment.vertical_flip:
            steps.append(v2.RandomVerticalFlip())
        if augment.rotation_degrees:
            steps.append(v2.RandomRotation(augment.rotation_degrees, fill=255))
        if augment.color_jitter:
            j = augment.color_jitter
            steps.append(v2.ColorJitter(brightness=j, contrast=j, saturation=j, hue=min(0.05, j / 4)))
    return v2.Compose([v2.ToImage(), *steps, v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])


class ImageDataset(Dataset):
    """(image tensor, class index) pairs for a manifest DataFrame."""

    def __init__(self, df: pd.DataFrame, data_dir: str | Path, classes: list[str], transform: Callable):
        self.paths = [Path(data_dir) / p for p in df["image_path"]]
        index = {name: i for i, name in enumerate(classes)}
        self.targets = [index[label] for label in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(load_image(self.paths[i])), self.targets[i]
