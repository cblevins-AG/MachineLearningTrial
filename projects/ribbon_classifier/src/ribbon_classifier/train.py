"""Fine-tune a pretrained backbone on the labelled ribbon images.

One *epoch* = one pass over the training images, in *batches*. For each batch:
forward pass -> loss (how wrong the predictions are) -> backward pass
(gradients: which way to nudge each weight) -> optimizer step (nudge them).
After each epoch the model is scored on the validation split; the best epoch
is saved, and training stops early once validation stops improving - the
classic sign that the model has started memorising instead of learning.

Two phases (standard transfer-learning practice):
1. ``head_only_epochs``: backbone frozen, only the new head learns. A random
   head would otherwise send large, noisy gradients into good pretrained
   weights and damage them.
2. Remaining epochs: everything trains, backbone at a 10x smaller learning rate.
"""

from __future__ import annotations

import csv
import platform
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ImageDataset, build_transforms
from .device import describe_device, pick_device, seed_everything
from .model import create_model, normalization, save_run, split_parameters


def _loader(dataset: ImageDataset, batch_size: int, shuffle: bool, workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def predict_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """(true class indices, predicted probabilities) for every item in *loader*."""
    model.eval()
    targets, probabilities = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(probabilities)


def train(config: Config, csv_path: str | Path, run_name: str | None = None, device: str = "auto") -> Path:
    """Train, keep the best checkpoint by validation macro-F1, return the run folder."""
    seed_everything(config.train.seed)
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if "split" not in df.columns:
        raise ValueError(f"{csv_path} has no 'split' column - run `ribbon split` first.")
    train_df, val_df = df[df["split"] == "train"], df[df["split"] == "val"]
    if train_df.empty or val_df.empty:
        raise ValueError("Need images in both the train and val splits.")
    classes = sorted(df["label"].unique())

    dev = pick_device(device)
    print(f"Device: {describe_device(dev)}")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Images: {len(train_df)} train, {len(val_df)} val")

    mc, tc = config.model, config.train
    model = create_model(mc.backbone, len(classes), mc.pretrained, mc.drop_rate).to(dev)
    mean, std = normalization(model)
    data_dir = config.path(config.paths.data_dir)
    train_set = ImageDataset(train_df, data_dir, classes, build_transforms(mc.image_size, mean, std, config.augment))
    val_set = ImageDataset(val_df, data_dir, classes, build_transforms(mc.image_size, mean, std))
    train_loader = _loader(train_set, tc.batch_size, True, tc.num_workers, dev)
    val_loader = _loader(val_set, tc.batch_size, False, tc.num_workers, dev)

    # Class weights ("balanced"): rare classes count more in the loss, so the
    # model cannot score well by always guessing the most common class.
    counts = train_df["label"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype=np.float32)
    weights = len(train_df) / (len(classes) * np.maximum(counts, 1))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=dev), label_smoothing=tc.label_smoothing)

    backbone_params, head_params = split_parameters(model)
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": tc.backbone_lr}, {"params": head_params, "lr": tc.head_lr}],
        weight_decay=tc.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, tc.epochs))
    use_amp = tc.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{mc.backbone.split('.')[0]}"
    run_dir = config.path(config.paths.models_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0

    for epoch in range(1, tc.epochs + 1):
        backbone_trainable = epoch > tc.head_only_epochs
        for p in backbone_params:
            p.requires_grad_(backbone_trainable)

        model.train()
        running_loss, seen, correct = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(labels)
            seen += len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        scheduler.step()

        y_true, probs = predict_loader(model, val_loader, dev)
        y_pred = probs.argmax(1)
        row = {
            "epoch": epoch,
            "phase": "full" if backbone_trainable else "head-only",
            "train_loss": round(running_loss / seen, 4),
            "train_acc": round(correct / seen, 4),
            "val_acc": round(accuracy_score(y_true, y_pred), 4),
            "val_macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        }
        history.append(row)
        improved = row["val_macro_f1"] > best_f1
        print(
            f"epoch {epoch:3d} [{row['phase']:9}] loss {row['train_loss']:.3f}  train acc {row['train_acc']:.3f}  "
            f"val acc {row['val_acc']:.3f}  val F1 {row['val_macro_f1']:.3f}{'  *best*' if improved else ''}"
        )
        if improved:
            best_f1, best_epoch, bad_epochs = row["val_macro_f1"], epoch, 0
            save_run(
                model,
                run_dir,
                {
                    "backbone": mc.backbone,
                    "classes": classes,
                    "image_size": mc.image_size,
                    "mean": list(mean),
                    "std": list(std),
                    "trained_on_crops": "crop_found" in df.columns,
                    "training_csv": csv_path.resolve().as_posix(),
                    "best_epoch": epoch,
                    "val_accuracy": row["val_acc"],
                    "val_macro_f1": row["val_macro_f1"],
                    "created": datetime.now().isoformat(timespec="seconds"),
                    "config": config.to_dict(),
                    "versions": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "timm": timm.__version__,
                    },
                },
            )
        elif backbone_trainable:
            bad_epochs += 1
            if bad_epochs >= tc.patience:
                print(f"Early stopping: no validation improvement for {tc.patience} epochs.")
                break

    with open(run_dir / "history.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"Best epoch {best_epoch}: val macro-F1 {best_f1:.3f}. Saved to {run_dir}")
    return run_dir
