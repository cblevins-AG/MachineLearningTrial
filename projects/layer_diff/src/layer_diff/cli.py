"""Command-line interface: ``layer-diff inspect | compare | make-samples``.

Built on argparse (standard library; comparable to System.CommandLine).
Run ``layer-diff --help`` or ``layer-diff compare --help`` for usage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pikepdf

from .config import load_settings
from .pdf_layers import find_unlayered_content, list_layers, list_spot_colors, page_count
from .pipeline import run
from .samples import make_samples

# projects/layer_diff/ (this file is projects/layer_diff/src/layer_diff/cli.py)
PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_pages(text: str | None) -> list[int] | None:
    """'1,3-5' -> [0, 2, 3, 4] (0-based). None/'' -> None (all pages)."""
    if not text:
        return None
    pages: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(v) for v in part.split("-", 1))
            pages.update(range(start, end + 1))
        elif part:
            pages.add(int(part))
    if any(p < 1 for p in pages):
        raise ValueError("Page numbers start at 1.")
    return sorted(p - 1 for p in pages)


def _cmd_inspect(args: argparse.Namespace) -> int:
    for pdf in args.pdfs:
        print(f"\n{pdf}")
        print(f"  pages: {page_count(pdf)}")
        layers = list_layers(pdf)
        if layers:
            print("  layers:")
            for layer in layers:
                state = "on" if layer.visible_by_default else "off"
                print(f"    - {layer.name!r}  (visible by default: {state})")
        else:
            print("  layers: none (this PDF has no optional content groups)")
        unlayered = find_unlayered_content(pdf)
        if unlayered:
            print(
                f"  WARNING: page(s) {unlayered} draw content outside every layer; it will appear in every layer render."
            )
        spots = list_spot_colors(pdf)
        if spots:
            print(f"  spot colours: {', '.join(spots)}")
    return 0


def _fmt(score: float | None) -> str:
    return "  n/a " if score is None else f"{score:5.1f}%"


def _cmd_compare(args: argparse.Namespace) -> int:
    settings = load_settings(
        args.config,
        dpi=args.dpi,
        color_tolerance=args.tolerance,
        min_region_mm2=args.min_region_mm2,
        mode=args.mode,
        describer=args.describer,
        claude_model=args.claude_model,
        save_images=False if args.no_images else None,
    )
    out_dir = args.out or PROJECT_DIR / "data" / "output" / datetime.now().strftime("%Y%m%d-%H%M%S")
    report = run(args.pdfs, args.layers, settings, out_dir, pages=parse_pages(args.pages))

    names = [Path(f["path"]).name for f in report["files"]]
    for info in report["files"]:
        if info["unlayered_content_pages"]:
            print(
                f"WARNING: {info['name']} has content outside any layer on page(s) {info['unlayered_content_pages']}."
            )
    for layer in report["layers"]:
        print(f"\n=== Layer '{layer['pattern']}'  overall difference: {_fmt(layer['overall_score_pct']).strip()}")
        width = max(len(n) for n in names)
        print(" " * (width + 2) + "  ".join(f"{i + 1:>6}" for i in range(len(names))))
        for i, (name, row) in enumerate(zip(names, layer["matrix"]["scores"], strict=True), start=1):
            print(f"{name:<{width}} {i:>1} " + "  ".join(_fmt(v) for v in row))
        for pair in layer["pairs"]:
            print(f"\n  {pair['summary']}")
            for line in pair["details"]:
                print(f"    - {line}")
    print(f"\nReport: {Path(out_dir) / 'report.md'}")
    return 0


def _cmd_samples(args: argparse.Namespace) -> int:
    for path in make_samples(args.out):
        print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="layer-diff",
        description="Compare named layers across production PDFs, describe the differences and score them.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="List pages, layers, spot colours and warnings for PDF files.")
    p.add_argument("pdfs", nargs="+", type=Path)
    p.set_defaults(func=_cmd_inspect)

    p = sub.add_parser("compare", help="Compare one or more layers across 2..N PDF files.")
    p.add_argument("pdfs", nargs="+", type=Path, help="Two or more PDF files.")
    p.add_argument(
        "-l",
        "--layer",
        dest="layers",
        action="append",
        required=True,
        metavar="NAME",
        help="Layer name or keyword (repeatable). Case-insensitive; wildcards (Cut*) and 're:<regex>' allowed.",
    )
    p.add_argument("-o", "--out", type=Path, help="Output folder (default: data/output/<timestamp>).")
    p.add_argument("--pages", help="Pages to compare, 1-based, e.g. '1,3-5' (default: all).")
    p.add_argument("--config", type=Path, help="YAML settings file (see configs/default.yaml).")
    p.add_argument("--dpi", type=float, help="Render resolution (default 150).")
    p.add_argument("--tolerance", type=float, help="Colour difference (delta E) treated as 'same' (default 10).")
    p.add_argument("--min-region-mm2", type=float, help="Ignore changed areas smaller than this (default 0.5).")
    p.add_argument("--mode", choices=["pairwise", "baseline"], help="All pairs, or each file vs the first.")
    p.add_argument("--describer", choices=["local", "claude"], help="How to write the plain-English text.")
    p.add_argument("--claude-model", help="Claude model id for --describer claude.")
    p.add_argument("--no-images", action="store_true", help="Skip writing PNG renders and overlays.")
    p.set_defaults(func=_cmd_compare)

    p = sub.add_parser("make-samples", help="Write three small layered sample PDFs to try the tool on.")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "data" / "samples")
    p.set_defaults(func=_cmd_samples)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except pikepdf.PasswordError as error:
        print(f"error: a PDF is password-protected ({error}); remove the security in Acrobat first.", file=sys.stderr)
        return 2
    except pikepdf.PdfError as error:
        print(f"error: could not read a PDF ({error}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
