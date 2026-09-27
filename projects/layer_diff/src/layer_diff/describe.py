"""Turn measured differences into plain-English sentences.

Two describers share one interface, ``describe(pair, settings) -> Description``:

* :class:`LocalDescriber` (default): deterministic sentences built from the
  measurements - position, size in millimetres, colour names, direction of
  movement. Nothing leaves the machine, it costs nothing, and the same input
  always yields the same text.
* :class:`ClaudeDescriber` (optional, ``--describer claude``): sends the two
  layer renders, the colour-coded overlay and the measured regions to the
  Claude API and asks for a description in production vocabulary. Richer
  wording ("the left notch of the dieline is now deeper"), but it needs an
  API key, costs money per call and sends artwork off the machine. If the
  call fails, the local description is used instead.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

import cv2
import numpy as np

from .compare import PairResult, Region
from .config import Settings

# Small, print-friendly palette for naming colours. Matching is done in CIELAB
# so "nearest" means "looks most similar", not "closest RGB numbers".
_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "dark gray": (80, 80, 80),
    "gray": (140, 140, 140),
    "light gray": (205, 205, 205),
    "white": (255, 255, 255),
    "red": (220, 30, 35),
    "dark red": (130, 20, 25),
    "pink": (245, 150, 180),
    "magenta": (220, 0, 140),
    "orange": (245, 130, 30),
    "yellow": (250, 220, 30),
    "gold": (200, 160, 40),
    "brown": (120, 75, 35),
    "light green": (150, 215, 120),
    "green": (40, 150, 60),
    "dark green": (20, 80, 35),
    "cyan": (0, 175, 220),
    "light blue": (140, 190, 235),
    "blue": (30, 80, 200),
    "navy": (20, 30, 90),
    "purple": (120, 50, 150),
}
_NAME_LIST = list(_NAMED_COLORS)
_PALETTE_LAB = cv2.cvtColor(np.array([list(_NAMED_COLORS.values())], dtype=np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0]


@dataclass
class Description:
    summary: str
    details: list[str] = field(default_factory=list)
    source: str = "local"  # "local" or the Claude model id that wrote it


def color_name(rgb: tuple[int, int, int]) -> str:
    lab = cv2.cvtColor(np.array([[rgb]], dtype=np.float32) / 255.0, cv2.COLOR_RGB2Lab)[0, 0]
    return _NAME_LIST[int(np.argmin(np.linalg.norm(_PALETTE_LAB - lab, axis=1)))]


def _position(region: Region, page_w_px: int, page_h_px: int) -> str:
    cx = (region.x + region.width / 2) / max(page_w_px, 1)
    cy = (region.y + region.height / 2) / max(page_h_px, 1)
    vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    if vertical == "middle" and horizontal == "center":
        return "center"
    if vertical == "middle":
        return f"middle-{horizontal}"
    return f"{vertical}-{horizontal}"


def _describe_region(region: Region, page_shape: tuple[int, int], mm_per_px: float) -> str:
    h_px, w_px = page_shape
    size = f"{region.width * mm_per_px:.1f} x {region.height * mm_per_px:.1f} mm"
    where = (
        f"{_position(region, w_px, h_px)} of the page "
        f"(x {region.x * mm_per_px:.1f} mm, y {region.y * mm_per_px:.1f} mm from the top-left corner)"
    )
    label = f"#{region.id}"
    if region.kind == "added":
        color = f" {color_name(region.color_after)}" if region.color_after else ""
        return f"{label} Added: new{color} artwork, about {size}, at the {where}."
    if region.kind == "removed":
        color = f" {color_name(region.color_before)}" if region.color_before else ""
        return f"{label} Removed:{color} artwork, about {size}, is gone from the {where}."
    if region.kind == "moved" and region.shift_px is not None:
        dx, dy = (v * mm_per_px for v in region.shift_px)
        steps = []
        if abs(dx) >= 0.05:
            steps.append(f"{abs(dx):.1f} mm {'right' if dx > 0 else 'left'}")
        if abs(dy) >= 0.05:
            steps.append(f"{abs(dy):.1f} mm {'down' if dy > 0 else 'up'}")
        return f"{label} Moved: an element about {size} at the {where} shifted {' and '.join(steps)}."
    before = color_name(region.color_before) if region.color_before else None
    after = color_name(region.color_after) if region.color_after else None
    if before and after and before != after:
        return f"{label} Changed: artwork about {size} at the {where} changed colour from {before} to {after}."
    return f"{label} Changed: the shape or content of artwork about {size} at the {where} is different."


class LocalDescriber:
    """Deterministic, offline descriptions from the measured regions."""

    def describe(self, pair: PairResult, settings: Settings, images: object = None) -> Description:
        a, b = pair.file_a.name, pair.file_b.name
        if not pair.comparable:
            return Description(summary=" ".join(pair.notes) or f"'{pair.pattern}' could not be compared.")

        regions = [(page, region) for page in pair.pages for region in page.regions]
        score = pair.score_pct or 0.0
        if not regions:
            summary = (
                f"'{pair.pattern}': {a} and {b} are visually identical at {settings.dpi:g} DPI "
                f"(colour tolerance delta E {settings.color_tolerance:g})."
            )
            return Description(summary=summary, details=list(pair.notes))

        counts: dict[str, int] = {}
        for _, region in regions:
            counts[region.kind] = counts.get(region.kind, 0) + 1
        breakdown = ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))
        summary = (
            f"'{pair.pattern}': {a} vs {b} - {score:.1f}% of the layer's artwork differs "
            f"({len(regions)} difference{'s' if len(regions) != 1 else ''}: {breakdown}; SSIM {pair.ssim:.3f})."
        )
        details = list(pair.notes)
        multi_page = len(pair.pages) > 1
        for page, region in regions:
            sentence = _describe_region(region, page.pixel_kind.shape, settings.mm_per_px())
            details.append(f"Page {page.page}, {sentence}" if multi_page else sentence)
        return Description(summary=summary, details=details)


# --------------------------------------------------------------------------- #
# Optional cloud describer
# --------------------------------------------------------------------------- #

_CLAUDE_PROMPT = """You are reviewing print-production artwork. The images show ONE layer \
("{pattern}") rendered from two PDF files, followed by an overlay where red = only in \
{a}, blue = only in {b}, amber = present in both but different, with numbered boxes.

Measured differences (pixel-accurate, from the software - trust these numbers):
{regions}

Describe each numbered difference in one plain-English sentence a prepress operator \
would understand (what the element appears to be, what changed, where). Do not invent \
differences that are not in the list. Then give a one-sentence overall summary."""

_CLAUDE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "differences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"region_id": {"type": "integer"}, "description": {"type": "string"}},
                "required": ["region_id", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "differences"],
    "additionalProperties": False,
}


def _png_block(rgb: np.ndarray, max_side: int = 1568) -> dict:
    """Encode an RGB image as a base64 PNG content block, downscaled if large."""
    scale = max_side / max(rgb.shape[:2])
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    data = base64.standard_b64encode(png.tobytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


class ClaudeDescriber:
    """Asks Claude to phrase the measured differences in production language.

    Only the first page of each pair is sent (to bound cost); the measured
    regions for every page are still included as text.
    """

    def __init__(self, model: str = "claude-opus-5", client: object | None = None) -> None:
        self.model = model
        self._client = client  # injectable for tests; created lazily otherwise
        self._fallback = LocalDescriber()

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported here so the package works without the SDK installed

            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        return self._client

    def describe(
        self, pair: PairResult, settings: Settings, images: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    ) -> Description:
        local = self._fallback.describe(pair, settings)
        if not pair.comparable or not any(page.regions for page in pair.pages) or images is None:
            return local  # nothing for a model to add

        mm = settings.mm_per_px()
        lines = [
            f"#{r.id} (page {p.page}): {r.kind}, {r.width * mm:.1f} x {r.height * mm:.1f} mm "
            f"at x {r.x * mm:.1f} mm, y {r.y * mm:.1f} mm"
            + (f", shifted ({r.shift_px[0] * mm:.1f}, {r.shift_px[1] * mm:.1f}) mm" if r.shift_px else "")
            for p in pair.pages
            for r in p.regions
        ]
        prompt = _CLAUDE_PROMPT.format(
            pattern=pair.pattern, a=pair.file_a.name, b=pair.file_b.name, regions="\n".join(lines)
        )
        try:
            response = self._get_client().beta.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": _CLAUDE_SCHEMA}},
                # If a safety classifier declines, let the API retry on its recommended fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[
                    {
                        "role": "user",
                        "content": [*(_png_block(img) for img in images), {"type": "text", "text": prompt}],
                    }
                ],
            )
            if response.stop_reason == "refusal":
                raise RuntimeError("the request was declined")
            text = next(block.text for block in response.content if block.type == "text")
            data = json.loads(text)
        except Exception as error:  # network, auth, quota, parsing: fall back, never crash the report
            local.details.append(f"(Claude describer unavailable: {type(error).__name__}: {error})")
            return local

        details = list(pair.notes) + [f"#{d['region_id']} {d['description']}" for d in data["differences"]]
        return Description(summary=data["summary"], details=details, source=response.model)
