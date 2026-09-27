"""Colour-coded overlay images and the JSON / Markdown report files."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from .compare import PIXEL_ADDED, PIXEL_CHANGED, PIXEL_REMOVED, PageDiff

# Overlay colours (RGB). Red/blue rather than red/green so the two most
# common kinds stay distinguishable with red-green colour blindness; every
# region also gets a numbered box, so colour is never the only cue.
COLOR_REMOVED = (214, 39, 40)
COLOR_ADDED = (31, 119, 180)
COLOR_CHANGED = (255, 170, 0)
COLOR_MOVED = (148, 103, 189)
_BOX_COLORS = {"removed": COLOR_REMOVED, "added": COLOR_ADDED, "changed": COLOR_CHANGED, "moved": COLOR_MOVED}


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "layer"


def overlay_image(image_a: np.ndarray, image_b: np.ndarray, diff: PageDiff) -> np.ndarray:
    """Faded union of both renders, changed pixels coloured, numbered boxes on each region."""
    height, width = diff.pixel_kind.shape
    canvas_a = np.full((height, width, 3), 255, np.uint8)
    canvas_b = canvas_a.copy()
    canvas_a[: image_a.shape[0], : image_a.shape[1]] = image_a
    canvas_b[: image_b.shape[0], : image_b.shape[1]] = image_b

    gray = cv2.cvtColor(np.minimum(canvas_a, canvas_b), cv2.COLOR_RGB2GRAY).astype(np.float32)
    faded = (255 - (255 - gray) * 0.3).astype(np.uint8)
    out = cv2.cvtColor(faded, cv2.COLOR_GRAY2RGB)
    out[diff.pixel_kind == PIXEL_REMOVED] = COLOR_REMOVED
    out[diff.pixel_kind == PIXEL_ADDED] = COLOR_ADDED
    out[diff.pixel_kind == PIXEL_CHANGED] = COLOR_CHANGED

    thickness = max(1, round(min(height, width) / 500))
    pad = 3 * thickness
    font_scale = max(0.5, min(height, width) / 700)
    for region in diff.regions:
        color = _BOX_COLORS.get(region.kind, COLOR_CHANGED)
        top_left = (max(0, region.x - pad), max(0, region.y - pad))
        bottom_right = (min(width - 1, region.x + region.width + pad), min(height - 1, region.y + region.height + pad))
        cv2.rectangle(out, top_left, bottom_right, color, thickness, cv2.LINE_AA)
        label_y = top_left[1] - 4 if top_left[1] > 20 else bottom_right[1] + int(20 * font_scale) + 4
        cv2.putText(
            out,
            f"#{region.id}",
            (top_left[0], label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Could not encode {path}")
    path.write_bytes(data.tobytes())  # imencode + write_bytes handles non-ASCII Windows paths


def _fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.1f}%"


def write_json(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_markdown(report: dict, path: Path) -> None:
    """Human-readable version of the JSON report (renders nicely on GitHub / VS Code preview)."""
    s = report["settings"]
    lines = [
        "# Layer difference report",
        "",
        f"Generated {report['generated_at']} by {report['tool']}. "
        f"Rendered at {s['dpi']:g} DPI; colour tolerance delta E {s['color_tolerance']:g}; "
        f"mode: {s['mode']}; describer: {s['describer']}.",
        "",
        "Overlay colours: **red** = only in the first file of the pair, **blue** = only in the second, "
        "**amber** = in both but different; boxes are numbered to match the text (purple box = moved).",
        "",
        "## Files",
        "",
        "| # | File | Pages | Layers | Warnings |",
        "|---|------|-------|--------|----------|",
    ]
    for i, info in enumerate(report["files"], start=1):
        warnings = []
        if info["unlayered_content_pages"]:
            pages = ", ".join(map(str, info["unlayered_content_pages"]))
            warnings.append(f"content outside any layer on page(s) {pages}")
        if not info["layers"]:
            warnings.append("no layers defined")
        layers = ", ".join(info["layers"]) or "-"
        lines.append(f"| {i} | {info['name']} | {info['pages']} | {layers} | {'; '.join(warnings) or '-'} |")

    for layer in report["layers"]:
        names = [Path(p).name for p in layer["matrix"]["files"]]
        lines += [
            "",
            f"## Layer pattern: `{layer['pattern']}`",
            "",
            f"**Overall difference score: {_fmt(layer['overall_score_pct'])}** "
            f"(mean of {len(layer['pairs'])} comparison(s); largest {_fmt(layer['max_score_pct'])}).",
            "",
            "| | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for name, row in zip(names, layer["matrix"]["scores"], strict=True):
            lines.append(f"| **{name}** | " + " | ".join(_fmt(v) for v in row) + " |")
        for pair in layer["pairs"]:
            lines += [
                "",
                f"### {Path(pair['file_a']).name} vs {Path(pair['file_b']).name}: {_fmt(pair['score_pct'])}",
                "",
            ]
            lines.append(pair["summary"])
            if pair["details"]:
                lines.append("")
                lines += [f"- {d}" for d in pair["details"]]
            for page in pair["pages"]:
                if page.get("overlay"):
                    lines += ["", f"![Page {page['page']} overlay]({page['overlay']})"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
