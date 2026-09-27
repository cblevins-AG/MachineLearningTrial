"""Score a trained run on a held-out split and write an error analysis.

Outputs (in the run folder):
* ``eval_<split>.json``   accuracy, macro-F1, per-class precision/recall/F1
* ``confusion_<split>.png`` which classes get mistaken for which
* ``errors_<split>.csv``  every misclassified image, most confident mistakes first
  (open these images: they are usually label errors, ambiguous photos, or a
  missing class)

Why macro-F1: with imbalanced classes, plain accuracy rewards ignoring the
rare ones. Macro-F1 averages each class's F1 equally, so a class the model
never gets right drags the score down.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ImageDataset, build_transforms
from .device import pick_device
from .model import load_run
from .train import predict_loader


def evaluate_run(
    config: Config, run_dir: str | Path, split: str = "test", csv_path: str | Path | None = None, device: str = "auto"
) -> dict:
    run_dir = Path(run_dir)
    dev = pick_device(device)
    model, meta = load_run(run_dir, dev)
    csv_path = Path(csv_path or meta["training_csv"])
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df = df[df["split"] == split]
    if df.empty:
        raise ValueError(f"No rows with split == '{split}' in {csv_path}.")
    classes = meta["classes"]
    unknown = sorted(set(df["label"]) - set(classes))
    if unknown:
        raise ValueError(f"Labels not seen in training: {unknown}")

    transform = build_transforms(meta["image_size"], tuple(meta["mean"]), tuple(meta["std"]))
    dataset = ImageDataset(df, config.path(config.paths.data_dir), classes, transform)
    y_true, probs = predict_loader(model, DataLoader(dataset, batch_size=config.train.batch_size), dev)
    y_pred = probs.argmax(1)

    label_ids = list(range(len(classes)))
    result = {
        "split": split,
        "images": int(len(df)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", labels=label_ids, zero_division=0)), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=label_ids, target_names=classes, output_dict=True, zero_division=0
        ),
    }
    (run_dir / f"eval_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    size = max(6, 0.6 * len(classes))
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=label_ids, display_labels=classes, ax=ax, xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"{split}: accuracy {result['accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(run_dir / f"confusion_{split}.png", dpi=120)
    plt.close(fig)

    errors = df.assign(predicted=[classes[i] for i in y_pred], confidence=probs.max(1).round(4))[
        lambda d: d["predicted"] != d["label"]
    ].sort_values("confidence", ascending=False)
    errors[["image_path", "label", "predicted", "confidence"]].to_csv(run_dir / f"errors_{split}.csv", index=False)
    return result
