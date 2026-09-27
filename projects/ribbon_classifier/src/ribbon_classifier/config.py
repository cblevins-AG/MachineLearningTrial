"""Typed configuration loaded from ``configs/default.yaml``.

Relative paths in the YAML are resolved against the folder *above* the one
holding the YAML file - for anything in ``configs/`` that is the project
folder (``projects/ribbon_classifier``) - so commands work from any working
directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "default.yaml"


@dataclass
class PathsConfig:
    data_dir: str = "data"  # image_path values in the CSVs are relative to this
    labels_csv: str = "data/labels.csv"  # every labelled image (you build this)
    splits_csv: str = "data/splits.csv"  # labels.csv + a train/val/test column (ribbon split)
    cropped_csv: str = "data/splits_cropped.csv"  # same, pointing at ribbon crops (ribbon crop)
    crops_dir: str = "data/processed/crops"
    models_dir: str = "models"


@dataclass
class SplitConfig:
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    # Sources that always go to the training split. The test set should look
    # like real-world input (phone photos), so the clean reference-document
    # images are used for learning but never for grading.
    train_only_sources: list[str] = field(default_factory=lambda: ["reference_pdf"])


@dataclass
class LocalizerConfig:
    model: str = "google/owlv2-base-patch16-ensemble"
    prompts: list[str] = field(default_factory=lambda: ["a photo of a ribbon", "a ribbon bow", "a piece of ribbon"])
    score_threshold: float = 0.10
    margin: float = 0.10  # extra border around the detected box, as a fraction of its size


@dataclass
class ModelConfig:
    backbone: str = "convnext_nano.in12k_ft_in1k"
    pretrained: bool = True
    image_size: int = 224
    drop_rate: float = 0.2


@dataclass
class AugmentConfig:
    # Every augmentation must keep the label true. Turn flips off if your
    # taxonomy has left/right-specific techniques (e.g. "left-angled cut").
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation_degrees: float = 15.0
    min_crop_scale: float = 0.6
    color_jitter: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 30
    head_only_epochs: int = 3  # train only the new classifier head first, backbone frozen
    batch_size: int = 32
    head_lr: float = 1.0e-3
    backbone_lr: float = 1.0e-4
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    patience: int = 7  # early stopping: epochs without val improvement
    num_workers: int = 0  # 0 is safest on Windows and in notebooks; try 4 once things work
    amp: bool = True  # mixed precision on NVIDIA GPUs (ignored elsewhere)
    seed: int = 42


@dataclass
class ProbeConfig:
    backbone: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    localizer: LocalizerConfig = field(default_factory=LocalizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    project_dir: Path = PROJECT_DIR

    def path(self, relative: str | Path) -> Path:
        """Resolve a config path against the project folder."""
        p = Path(relative)
        return p if p.is_absolute() else self.project_dir / p

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("project_dir")
        return data


def _build(cls: type, values: dict, where: str):
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown key(s) in {where}: {', '.join(sorted(unknown))}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if hasattr(default, "__dataclass_fields__") and isinstance(value, dict):
            value = _build(type(default), value, f"{where}.{name}")
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load YAML (default: configs/default.yaml). Missing keys keep their defaults."""
    if path is None:
        path = DEFAULT_CONFIG
        values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    else:
        path = Path(path)
        values = yaml.safe_load(path.read_text(encoding="utf-8"))  # a missing explicit file is an error
    config = _build(Config, values or {}, path.name)
    config.project_dir = path.resolve().parent.parent
    return config
