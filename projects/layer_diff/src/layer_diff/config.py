"""Tunable settings, with defaults that suit typical print/packaging PDFs.

``configs/default.yaml`` documents every field and mirrors these defaults (a
test keeps the two in sync). Copy it, edit it, and pass it with ``--config``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Settings:
    # Rendering resolution. 150 DPI = 0.17 mm per pixel: fine enough to catch a
    # nudged hairline, and a US-letter page is ~2 million pixels. Use 300 for
    # tiny text changes (4x the pixels, so ~4x slower).
    dpi: float = 150.0

    # Two pixels are "the same colour" when their CIE76 colour difference
    # (delta E, in CIELAB space) is at most this. ~2.3 is the smallest
    # difference a trained eye can see; 10 ignores rendering noise but still
    # catches any colour change a person would call a change.
    color_tolerance: float = 10.0

    # A pixel counts as ink (not blank paper) when its delta E from pure white
    # exceeds this.
    ink_threshold: float = 8.0

    # Changed areas smaller than this (in mm^2) are treated as rendering noise.
    min_region_mm2: float = 0.5

    # Changed pixels closer together than this are reported as one difference
    # (so a re-typed word is one region, not one region per letter).
    merge_distance_mm: float = 2.0

    # How far (mm) to look for an element that moved rather than changed.
    move_search_mm: float = 10.0

    # "pairwise": every file against every other file.
    # "baseline": the first file is the reference; each other file is compared to it.
    mode: str = "pairwise"

    # "local": sentences built from the measurements; nothing leaves the machine.
    # "claude": also send the layer images to the Claude API for a richer description.
    describer: str = "local"
    claude_model: str = "claude-opus-5"

    # Write PNG renders and colour-coded overlays next to the report.
    save_images: bool = True

    def mm_per_px(self) -> float:
        return 25.4 / self.dpi

    def validate(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if self.mode not in ("pairwise", "baseline"):
            raise ValueError(f"mode must be 'pairwise' or 'baseline', not {self.mode!r}")
        if self.describer not in ("local", "claude"):
            raise ValueError(f"describer must be 'local' or 'claude', not {self.describer!r}")


def load_settings(path: str | Path | None = None, **overrides: object) -> Settings:
    """Defaults <- YAML file (if given) <- explicit overrides (e.g. CLI flags)."""
    values: dict[str, object] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(Settings)}
        unknown = set(loaded) - known
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {', '.join(sorted(unknown))}")
        values.update(loaded)
    values.update({k: v for k, v in overrides.items() if v is not None})
    settings = Settings(**values)  # type: ignore[arg-type]
    settings.validate()
    return settings
