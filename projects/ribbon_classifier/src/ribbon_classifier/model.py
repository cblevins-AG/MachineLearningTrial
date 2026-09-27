"""Build the classifier and save/load trained runs.

Transfer learning in one paragraph: a *backbone* network pretrained on
millions of images already turns pixels into useful features (edges ->
textures -> shapes -> parts). We keep it, replace its last layer (the *head*)
with a new one sized for our classes, and train mostly the head plus a gentle
nudge to the backbone. That is why a few hundred labelled photos can be
enough, where training from scratch would need tens of thousands.
"""

from __future__ import annotations

import json
from pathlib import Path

import timm
import torch
from safetensors.torch import load_model, save_model

WEIGHTS_FILE = "model.safetensors"  # safetensors: weights only, no code execution on load (unlike pickle .pt files)
META_FILE = "meta.json"


def create_model(backbone: str, num_classes: int, pretrained: bool = True, drop_rate: float = 0.0) -> torch.nn.Module:
    """A timm backbone with a fresh classification head for *num_classes*.

    ``pretrained=True`` downloads ImageNet weights from the Hugging Face Hub
    on first use (cached under %USERPROFILE%\\.cache\\huggingface).
    """
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def split_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """(backbone parameters, head parameters) - trained at different learning rates."""
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    return [p for p in model.parameters() if id(p) not in head_ids], head


def normalization(model: torch.nn.Module) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The mean/std the backbone was pretrained with; inputs must be normalised the same way."""
    config = timm.data.resolve_model_data_config(model)
    return tuple(config["mean"]), tuple(config["std"])


def save_run(model: torch.nn.Module, run_dir: Path, meta: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(run_dir / WEIGHTS_FILE))
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_run(run_dir: str | Path, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Rebuild the architecture from meta.json and load the trained weights."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / META_FILE).read_text(encoding="utf-8"))
    model = create_model(meta["backbone"], len(meta["classes"]), pretrained=False)
    load_model(model, str(run_dir / WEIGHTS_FILE), device=str(device))
    return model.to(device).eval(), meta
