"""End-to-end run: inspect files -> render layers -> compare -> describe -> write report."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .compare import LayerComparison, PairResult, compare_renders
from .config import Settings
from .describe import ClaudeDescriber, Description, LocalDescriber
from .pdf_layers import find_unlayered_content, list_layers, list_spot_colors, page_count, render_layer
from .report import overlay_image, save_png, slug, write_json, write_markdown


def inspect_file(pdf_path: Path) -> dict:
    """Everything worth knowing about a file before comparing it."""
    return {
        "path": str(pdf_path),
        "name": pdf_path.name,
        "pages": page_count(pdf_path),
        "layers": [layer.name for layer in list_layers(pdf_path)],
        "unlayered_content_pages": find_unlayered_content(pdf_path),
        "spot_colors": list_spot_colors(pdf_path),
    }


def _pair_to_dict(pair: PairResult, description: Description, overlays: dict[int, str]) -> dict:
    return {
        "file_a": str(pair.file_a),
        "file_b": str(pair.file_b),
        "layers_a": pair.layers_a,
        "layers_b": pair.layers_b,
        "score_pct": pair.score_pct,
        "ssim": pair.ssim,
        "summary": description.summary,
        "details": description.details,
        "description_source": description.source,
        "notes": pair.notes,
        "pages": [
            {
                "page": page.page,
                "score_pct": page.score_pct,
                "ssim": page.ssim,
                "changed_px": page.changed_px,
                "ink_px": page.ink_px,
                "overlay": overlays.get(page.page),
                "regions": [asdict(region) for region in page.regions],
            }
            for page in pair.pages
        ],
    }


def run(
    pdf_paths: list[str | Path],
    patterns: list[str],
    settings: Settings,
    out_dir: str | Path,
    pages: list[int] | None = None,
    describer: LocalDescriber | ClaudeDescriber | None = None,
) -> dict:
    """Compare every layer pattern across every file and write the report.

    *pages* are 0-based page indices (``None`` = all pages). Returns the report
    as a dict (the same content as ``report.json``).
    """
    paths = [Path(p) for p in pdf_paths]
    if len(paths) < 2:
        raise ValueError("Give at least two PDF files to compare.")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Not found: {', '.join(missing)}")
    if not patterns:
        raise ValueError("Give at least one layer name or keyword to compare (e.g. --layer Dieline).")

    out_dir = Path(out_dir)
    if describer is None:
        describer = ClaudeDescriber(settings.claude_model) if settings.describer == "claude" else LocalDescriber()

    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tool": f"layer-diff {__version__}",
        "settings": asdict(settings),
        "files": [inspect_file(p) for p in paths],
        "layers": [],
    }
    tags = [f"{i}_{slug(p.stem)}" for i, p in enumerate(paths, start=1)]  # unique even if stems repeat

    for pattern in patterns:
        renders = [render_layer(p, pattern, settings.dpi, pages) for p in paths]
        comparison: LayerComparison = compare_renders(renders, settings)
        image_dir = out_dir / "images" / slug(pattern)
        if settings.save_images:
            for tag, render in zip(tags, renders, strict=True):
                for page_number, image in enumerate(render.pages, start=1):
                    save_png(image_dir / f"{tag}_p{page_number}.png", image)

        pairs = []
        for pair in comparison.pairs:
            ia, ib = paths.index(pair.file_a), paths.index(pair.file_b)
            overlays: dict[int, str] = {}
            first_images = None
            for page in pair.pages:
                image_a, image_b = renders[ia].pages[page.page - 1], renders[ib].pages[page.page - 1]
                overlay = overlay_image(image_a, image_b, page)
                if first_images is None:
                    first_images = (image_a, image_b, overlay)
                if settings.save_images:
                    target = image_dir / f"{tags[ia]}__vs__{tags[ib]}_p{page.page}.png"
                    save_png(target, overlay)
                    overlays[page.page] = target.relative_to(out_dir).as_posix()
            description = describer.describe(pair, settings, images=first_images)
            pairs.append(_pair_to_dict(pair, description, overlays))

        report["layers"].append(
            {
                "pattern": pattern,
                "mode": comparison.mode,
                "overall_score_pct": comparison.overall_score_pct,
                "max_score_pct": comparison.max_score_pct,
                "matched": {str(r.pdf_path): r.matched_layers for r in renders},
                "matrix": {"files": [str(p) for p in paths], "scores": comparison.matrix()},
                "pairs": pairs,
            }
        )

    write_json(report, out_dir / "report.json")
    write_markdown(report, out_dir / "report.md")
    return report
