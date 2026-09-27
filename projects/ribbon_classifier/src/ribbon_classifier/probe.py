"""Linear probe: the fastest honest answer to "is this task learnable from my data?".

A frozen, self-supervised backbone (DINOv2) turns each image into a feature
vector once; then a plain logistic regression learns to separate the classes.
No fine-tuning, no GPU needed, minutes on a laptop CPU. Use it to:

* sanity-check labels and splits before spending hours on training,
* get a baseline number that fine-tuning must beat,
* compare backbones cheaply.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .dataset import ImageDataset, build_transforms
from .device import describe_device, pick_device


@torch.no_grad()
def _features(backbone: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    chunks = [
        backbone(images.to(device)).float().cpu().numpy() for images, _ in tqdm(loader, desc="features", unit="batch")
    ]
    return np.concatenate(chunks)


def run_probe(config: Config, csv_path: str | Path, device: str = "auto") -> dict:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    classes = sorted(df["label"].unique())
    pc = config.probe
    dev = pick_device(device)
    print(f"Device: {describe_device(dev)}")

    extra = {"img_size": pc.image_size} if pc.backbone.startswith("vit") else {}  # ViTs can run at a smaller size
    backbone = timm.create_model(pc.backbone, pretrained=pc.pretrained, num_classes=0, **extra).to(dev).eval()
    data_cfg = timm.data.resolve_model_data_config(backbone)
    transform = build_transforms(pc.image_size, tuple(data_cfg["mean"]), tuple(data_cfg["std"]))
    data_dir = config.path(config.paths.data_dir)

    def loader(frame: pd.DataFrame) -> DataLoader:
        return DataLoader(ImageDataset(frame, data_dir, classes, transform), batch_size=pc.batch_size)

    x_train, x_val = _features(backbone, loader(train_df), dev), _features(backbone, loader(val_df), dev)
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()

    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_val)
    result = {
        "backbone": pc.backbone,
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "val_accuracy": round(float(accuracy_score(y_val, predicted)), 4),
        "val_macro_f1": round(float(f1_score(y_val, predicted, average="macro", zero_division=0)), 4),
        "per_class": classification_report(y_val, predicted, output_dict=True, zero_division=0),
    }
    out_dir = config.path(config.paths.models_dir) / f"probe_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"Linear probe on {pc.backbone}: val accuracy {result['val_accuracy']:.3f}, macro-F1 {result['val_macro_f1']:.3f}"
    )
    print(f"Report: {out_dir / 'probe_report.json'}")
    return result
