"""Measure the differences between renders of the same layer from different PDFs.

Because every file is rendered from vector data at the same DPI, identical
artwork produces identical pixels: no photo alignment is needed, and any
pixel that differs is a real change. The pipeline per page is:

1. Convert both renders to CIELAB, a colour space where straight-line
   distance approximates how different two colours *look* (delta E).
2. Mark pixels whose colour differs by more than ``color_tolerance``.
3. Group nearby changed pixels into regions and drop specks smaller than
   ``min_region_mm2`` (anti-aliasing noise).
4. Classify each region: *added* (ink only in B), *removed* (ink only in A),
   *changed* (ink in both, but different), or *moved* (A's content reappears
   in B at an offset: found by template matching).

Difference score (the headline number)::

    score % = 100 * changed pixels / pixels that have ink in A or in B

It is normalised by the *artwork* area, not the page area, so "a third of
the dieline changed" reads as ~33% whether the dieline fills the page or a
corner of it. SSIM (structural similarity, 1.0 = identical) is reported
alongside as a second opinion that is less sensitive to tiny shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .config import Settings
from .pdf_layers import LayerRender

# Per-pixel classification codes, used for the colour-coded overlay image.
PIXEL_SAME, PIXEL_REMOVED, PIXEL_ADDED, PIXEL_CHANGED = 0, 1, 2, 3

_WHITE_LAB = np.array([100.0, 0.0, 0.0], dtype=np.float32)
# Template matching on regions larger than this many pixels is skipped (slow, and
# an element that big "moving" is better described as a layout change).
_MAX_SHIFT_TEMPLATE_PX = 4_000_000


@dataclass
class Region:
    """One contiguous area that differs between file A and file B."""

    id: int
    kind: str  # "added" | "removed" | "changed" | "moved"
    x: int  # bounding box, pixels from the top-left corner of the page
    y: int
    width: int
    height: int
    changed_px: int
    color_before: tuple[int, int, int] | None = None  # mean RGB of A's ink in the region
    color_after: tuple[int, int, int] | None = None  # mean RGB of B's ink in the region
    shift_px: tuple[int, int] | None = None  # (dx, dy) for "moved"; +x right, +y down


@dataclass
class PageDiff:
    page: int  # 1-based
    changed_px: int
    ink_px: int  # pixels with ink in A or B (the score's denominator)
    ssim: float
    regions: list[Region]
    size_a_pt: tuple[float, float]
    size_b_pt: tuple[float, float]
    pixel_kind: np.ndarray = field(repr=False)  # PIXEL_* code per pixel

    @property
    def score_pct(self) -> float:
        return 100.0 * self.changed_px / self.ink_px if self.ink_px else 0.0


@dataclass
class PairResult:
    """All pages of one layer pattern, file A versus file B."""

    file_a: Path
    file_b: Path
    pattern: str
    layers_a: list[str]
    layers_b: list[str]
    pages: list[PageDiff]
    notes: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return bool(self.layers_a) and bool(self.layers_b)

    @property
    def changed_px(self) -> int:
        return sum(p.changed_px for p in self.pages)

    @property
    def ink_px(self) -> int:
        return sum(p.ink_px for p in self.pages)

    @property
    def score_pct(self) -> float | None:
        """Pixel-weighted across pages; ``None`` when the layer is missing from a file."""
        if not self.comparable:
            return None
        return 100.0 * self.changed_px / self.ink_px if self.ink_px else 0.0

    @property
    def ssim(self) -> float | None:
        return float(np.mean([p.ssim for p in self.pages])) if self.pages else None


@dataclass
class LayerComparison:
    """One layer pattern compared across the whole set of files."""

    pattern: str
    renders: list[LayerRender]
    pairs: list[PairResult]
    mode: str

    @property
    def overall_score_pct(self) -> float | None:
        """Mean of the pair scores: "how different these files are from one another"."""
        scores = [p.score_pct for p in self.pairs if p.score_pct is not None]
        return float(np.mean(scores)) if scores else None

    @property
    def max_score_pct(self) -> float | None:
        scores = [p.score_pct for p in self.pairs if p.score_pct is not None]
        return max(scores) if scores else None

    def matrix(self) -> list[list[float | None]]:
        """N x N score matrix (symmetric; diagonal 0). Unknown cells are ``None``."""
        index = {r.pdf_path: i for i, r in enumerate(self.renders)}
        n = len(self.renders)
        grid: list[list[float | None]] = [[0.0 if i == j else None for j in range(n)] for i in range(n)]
        for pair in self.pairs:
            i, j = index[pair.file_a], index[pair.file_b]
            grid[i][j] = grid[j][i] = pair.score_pct
        return grid


# --------------------------------------------------------------------------- #
# Page level
# --------------------------------------------------------------------------- #


def _pad_white(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _mean_rgb(image: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
    if not mask.any():
        return None
    r, g, b = image[mask].mean(axis=0)
    return int(round(r)), int(round(g)), int(round(b))


def _classify(member: np.ndarray, ink_a: np.ndarray, ink_b: np.ndarray) -> str:
    total = int(member.sum())
    only_b = int((member & ink_b & ~ink_a).sum())
    only_a = int((member & ink_a & ~ink_b).sum())
    if only_b >= 0.8 * total:
        return "added"
    if only_a >= 0.8 * total:
        return "removed"
    return "changed"


def _find_shift(
    light_a: np.ndarray, light_b: np.ndarray, box: tuple[int, int, int, int], search_px: int
) -> tuple[int, int] | None:
    """Does A's content inside *box* reappear in B at an offset?

    Slides A's patch over a window of B (template matching) and accepts the
    best offset only if it explains at least 80% of the pixel difference.
    """
    x, y, w, h = box
    if w * h > _MAX_SHIFT_TEMPLATE_PX or search_px < 1:
        return None
    height, width = light_b.shape
    template = light_a[y : y + h, x : x + w]
    x0, y0 = max(0, x - search_px), max(0, y - search_px)
    x1, y1 = min(width, x + w + search_px), min(height, y + h + search_px)
    window = light_b[y0:y1, x0:x1]
    result = cv2.matchTemplate(window, template, cv2.TM_SQDIFF)
    best_sq_diff, _, best_loc, _ = cv2.minMaxLoc(result)
    dx, dy = x0 + best_loc[0] - x, y0 + best_loc[1] - y
    if dx == 0 and dy == 0:
        return None
    unshifted = float(np.sum((template - light_b[y : y + h, x : x + w]) ** 2))
    if unshifted <= 0 or best_sq_diff > 0.2 * unshifted:
        return None
    return int(dx), int(dy)


def _contains_center(outer: tuple[int, int, int, int], region: Region, slack: int) -> bool:
    ox, oy, ow, oh = outer
    cx, cy = region.x + region.width / 2, region.y + region.height / 2
    return ox - slack <= cx <= ox + ow + slack and oy - slack <= cy <= oy + oh + slack


def compare_pages(
    image_a: np.ndarray,
    image_b: np.ndarray,
    settings: Settings,
    page: int = 1,
    size_a_pt: tuple[float, float] = (0.0, 0.0),
    size_b_pt: tuple[float, float] = (0.0, 0.0),
) -> PageDiff:
    """Compare two RGB renders of the same page and layer."""
    height = max(image_a.shape[0], image_b.shape[0])
    width = max(image_a.shape[1], image_b.shape[1])
    a = _pad_white(image_a, height, width)
    b = _pad_white(image_b, height, width)

    lab_a, lab_b = _to_lab(a), _to_lab(b)
    ink_a = np.linalg.norm(lab_a - _WHITE_LAB, axis=2) > settings.ink_threshold
    ink_b = np.linalg.norm(lab_b - _WHITE_LAB, axis=2) > settings.ink_threshold
    changed = np.linalg.norm(lab_a - lab_b, axis=2) > settings.color_tolerance

    mm_per_px = settings.mm_per_px()
    min_px = max(1, round(settings.min_region_mm2 / mm_per_px**2))
    merge_px = max(1, round(settings.merge_distance_mm / mm_per_px))
    search_px = round(settings.move_search_mm / mm_per_px)

    # Dilating ("fattening") the changed pixels makes nearby changes touch, so
    # connected-component labelling returns one region per visual change.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1, 2 * merge_px + 1))
    grouped = cv2.dilate(changed.astype(np.uint8), kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    pixel_kind = np.zeros((height, width), dtype=np.uint8)
    regions: list[Region] = []
    for label in range(1, count):
        bx, by, bw, bh = (int(v) for v in stats[label, :4])
        window = (slice(by, by + bh), slice(bx, bx + bw))
        member = (labels[window] == label) & changed[window]
        n_changed = int(member.sum())
        if n_changed < min_px:
            continue
        wa, wb = ink_a[window], ink_b[window]
        ys, xs = np.nonzero(member)
        kind_view = pixel_kind[window]
        kind_view[member] = PIXEL_CHANGED
        kind_view[member & wa & ~wb] = PIXEL_REMOVED
        kind_view[member & wb & ~wa] = PIXEL_ADDED
        regions.append(
            Region(
                id=0,
                kind=_classify(member, wa, wb),
                x=bx + int(xs.min()),
                y=by + int(ys.min()),
                width=int(xs.max() - xs.min() + 1),
                height=int(ys.max() - ys.min() + 1),
                changed_px=n_changed,
                color_before=_mean_rgb(a[window], member & wa),
                color_after=_mean_rgb(b[window], member & wb),
            )
        )

    # Movement: a "removed" or "changed" region whose A-content reappears in B
    # nearby is a move. The matching "added" region at the destination is
    # folded into the move instead of being reported separately.
    absorbed: set[int] = set()
    for i, region in enumerate(regions):
        if region.kind not in ("removed", "changed"):
            continue
        box = (region.x, region.y, region.width, region.height)
        shift = _find_shift(lab_a[:, :, 0], lab_b[:, :, 0], box, search_px)
        if shift is None:
            continue
        region.kind, region.shift_px = "moved", shift
        destination = (region.x + shift[0], region.y + shift[1], region.width, region.height)
        for j, other in enumerate(regions):
            if j != i and other.kind == "added" and _contains_center(destination, other, merge_px):
                absorbed.add(j)
                region.changed_px += other.changed_px

    regions = [r for j, r in enumerate(regions) if j not in absorbed]
    regions.sort(key=lambda r: (r.y, r.x))  # reading order: top to bottom, left to right
    for number, region in enumerate(regions, start=1):
        region.id = number

    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    ssim = 1.0 if np.array_equal(gray_a, gray_b) else float(structural_similarity(gray_a, gray_b, data_range=255))

    return PageDiff(
        page=page,
        changed_px=sum(r.changed_px for r in regions),
        ink_px=int((ink_a | ink_b).sum()),
        ssim=ssim,
        regions=regions,
        size_a_pt=size_a_pt,
        size_b_pt=size_b_pt,
        pixel_kind=pixel_kind,
    )


# --------------------------------------------------------------------------- #
# File and set level
# --------------------------------------------------------------------------- #


def _pt_to_mm(size: tuple[float, float]) -> str:
    return f"{size[0] * 25.4 / 72:.1f} x {size[1] * 25.4 / 72:.1f} mm"


def compare_pair(render_a: LayerRender, render_b: LayerRender, settings: Settings) -> PairResult:
    result = PairResult(
        file_a=render_a.pdf_path,
        file_b=render_b.pdf_path,
        pattern=render_a.pattern,
        layers_a=render_a.matched_layers,
        layers_b=render_b.matched_layers,
        pages=[],
    )
    for render in (render_a, render_b):
        if not render.found:
            result.notes.append(f"No layer matching '{render.pattern}' exists in {render.pdf_path.name}.")
    if not result.comparable:
        return result

    n_a, n_b = len(render_a.pages), len(render_b.pages)
    if n_a != n_b:
        longer = render_a if n_a > n_b else render_b
        result.notes.append(
            f"{longer.pdf_path.name} has {abs(n_a - n_b)} more page(s); "
            f"only the first {min(n_a, n_b)} page(s) were compared."
        )
    if set(render_a.matched_layers) != set(render_b.matched_layers):
        result.notes.append(
            f"The pattern matched different layer names: {', '.join(render_a.matched_layers)} "
            f"in {render_a.pdf_path.name} vs {', '.join(render_b.matched_layers)} in {render_b.pdf_path.name}."
        )

    for index in range(min(n_a, n_b)):
        size_a, size_b = render_a.page_sizes_pt[index], render_b.page_sizes_pt[index]
        if max(abs(size_a[0] - size_b[0]), abs(size_a[1] - size_b[1])) > 0.5:
            result.notes.append(
                f"Page {index + 1} size differs: {_pt_to_mm(size_a)} vs {_pt_to_mm(size_b)}; "
                "pages were aligned at their top-left corners."
            )
        result.pages.append(
            compare_pages(render_a.pages[index], render_b.pages[index], settings, index + 1, size_a, size_b)
        )
    return result


def compare_renders(renders: list[LayerRender], settings: Settings) -> LayerComparison:
    """Compare one layer pattern across all files (pairwise, or against the first file)."""
    if len(renders) < 2:
        raise ValueError("Need at least two files to compare.")
    if settings.mode == "baseline":
        index_pairs = [(0, j) for j in range(1, len(renders))]
    else:
        index_pairs = list(combinations(range(len(renders)), 2))
    pairs = [compare_pair(renders[i], renders[j], settings) for i, j in index_pairs]
    return LayerComparison(renders[0].pattern, renders, pairs, settings.mode)
