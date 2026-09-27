"""Command-line interface: ``ribbon <step> [options]``. Run ``ribbon --help``.

Every step reads ``configs/default.yaml`` unless ``--config`` points elsewhere,
and prints where it wrote its output, so the steps chain together.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import Config, load_config


def _training_csv(config: Config, explicit: str | None) -> Path:
    """Explicit --csv, else the cropped manifest if `ribbon crop` has run, else splits.csv."""
    if explicit:
        return Path(explicit)
    cropped = config.path(config.paths.cropped_csv)
    chosen = cropped if cropped.exists() else config.path(config.paths.splits_csv)
    print(f"Using {chosen} (pass --csv to choose another file)")
    return chosen


def _cmd_extract_reference(args: argparse.Namespace, config: Config) -> None:
    from .reference_pdf import extract_reference_pdf

    out = Path(args.out) if args.out else config.path("data/interim/reference_pdf")
    df = extract_reference_pdf(args.pdf, out, config.path(config.paths.data_dir), args.min_side, args.render_pages_dpi)
    print(f"Extracted {len(df)} image(s) from {df['page'].nunique() if len(df) else 0} page(s) into {out}")
    print(
        f"Next: open {out / 'reference_candidates.csv'} in Excel, fill in 'label', delete non-ribbon rows, save as CSV UTF-8."
    )


def _cmd_manifest(args: argparse.Namespace, config: Config) -> None:
    from .dataset import manifest_from_folders, read_manifest

    data_dir = config.path(config.paths.data_dir)
    frames = [manifest_from_folders(folder, data_dir, args.source) for folder in args.folders or []]
    for csv_file in args.csv or []:
        frame = read_manifest(csv_file)
        skipped = int((frame["label"] == "").sum())
        if skipped:
            print(f"{csv_file}: skipping {skipped} unlabelled row(s)")
        frames.append(frame[frame["label"] != ""][["image_path", "label", "group", "source"]])
    if not frames:
        raise ValueError("Give at least one --folders DIR or --csv FILE.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("image_path")
    out = Path(args.out) if args.out else config.path(config.paths.labels_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df["label"].value_counts().to_string())
    print(f"Wrote {len(df)} row(s), {df['label'].nunique()} class(es) to {out}")


def _cmd_split(args: argparse.Namespace, config: Config) -> None:
    from .dataset import read_manifest, validate_manifest
    from .split import assign_splits, split_summary

    labels = Path(args.labels) if args.labels else config.path(config.paths.labels_csv)
    df, problems = validate_manifest(
        read_manifest(labels), config.path(config.paths.data_dir), open_images=not args.quick
    )
    for problem in problems:
        print(f"  ! {problem}")
    sc = config.split
    df = assign_splits(df, sc.val_fraction, sc.test_fraction, sc.seed, sc.train_only_sources)
    out = Path(args.out) if args.out else config.path(config.paths.splits_csv)
    df.to_csv(out, index=False)
    print(split_summary(df).to_string())
    print(f"Wrote {out}")


def _cmd_crop(args: argparse.Namespace, config: Config) -> None:
    from .device import pick_device
    from .localize import RibbonLocalizer, crop_manifest

    source = Path(args.csv) if args.csv else config.path(config.paths.splits_csv)
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    localizer = RibbonLocalizer(config.localizer, device=str(pick_device(args.device)))
    out_df = crop_manifest(df, config.path(config.paths.data_dir), config.path(config.paths.crops_dir), localizer)
    out = Path(args.out) if args.out else config.path(config.paths.cropped_csv)
    out_df.to_csv(out, index=False)
    missed = int((~out_df["crop_found"]).sum())
    print(f"Cropped {len(out_df) - missed}/{len(out_df)} image(s); {missed} kept whole (no ribbon found). Wrote {out}")
    if missed:
        print("Review rows with crop_found == False, and the lowest crop_score values, before training.")


def _cmd_probe(args: argparse.Namespace, config: Config) -> None:
    from .probe import run_probe

    run_probe(config, _training_csv(config, args.csv), args.device)


def _cmd_train(args: argparse.Namespace, config: Config) -> None:
    from .train import train

    if args.epochs:
        config.train.epochs = args.epochs
    train(config, _training_csv(config, args.csv), args.run_name, args.device)


def _cmd_evaluate(args: argparse.Namespace, config: Config) -> None:
    from .evaluate import evaluate_run

    result = evaluate_run(config, args.run_dir, args.split, args.csv, args.device)
    print(
        f"{result['split']}: {result['images']} images, accuracy {result['accuracy']:.3f}, macro-F1 {result['macro_f1']:.3f}"
    )
    print(f"Details, confusion matrix and errors_{args.split}.csv are in {args.run_dir}")


def _cmd_predict(args: argparse.Namespace, config: Config) -> None:
    from .device import pick_device
    from .model import META_FILE
    from .predict import Predictor

    meta = json.loads((Path(args.run_dir) / META_FILE).read_text(encoding="utf-8"))
    localizer = None
    if meta.get("trained_on_crops") and not args.no_localize:
        from .localize import RibbonLocalizer

        localizer = RibbonLocalizer(config.localizer, device=str(pick_device(args.device)))
    predictor = Predictor(args.run_dir, args.device, localizer)
    for image in args.images:
        print(json.dumps(predictor.predict(image, args.top_k)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ribbon", description="Ribbon fold/cut classifier pipeline.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="YAML config (default: configs/default.yaml).")
    common.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "extract-reference", parents=[common], help="Extract candidate images + page text from the reference PDF."
    )
    p.add_argument("pdf")
    p.add_argument("--out", help="Output folder (default: data/interim/reference_pdf).")
    p.add_argument("--min-side", type=int, default=128, help="Skip images smaller than this (px).")
    p.add_argument("--render-pages-dpi", type=float, help="Also save whole pages at this DPI (for scanned PDFs).")
    p.set_defaults(func=_cmd_extract_reference)

    p = sub.add_parser("manifest", parents=[common], help="Combine labelled folders and CSVs into labels.csv.")
    p.add_argument("--folders", nargs="*", help="Folders whose sub-folders are label names.")
    p.add_argument("--csv", nargs="*", help="Labelled CSVs (e.g. the finished reference_candidates.csv).")
    p.add_argument("--source", default="examples", help="Source name recorded for --folders images.")
    p.add_argument("--out", help="Default: data/labels.csv")
    p.set_defaults(func=_cmd_manifest)

    p = sub.add_parser("split", parents=[common], help="Validate labels.csv and assign train/val/test.")
    p.add_argument("--labels", help="Default: data/labels.csv")
    p.add_argument("--out", help="Default: data/splits.csv")
    p.add_argument("--quick", action="store_true", help="Skip opening every image (faster, less thorough).")
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("crop", parents=[common], help="Crop every image to the ribbon (zero-shot OWLv2 detector).")
    p.add_argument("--csv", help="Default: data/splits.csv")
    p.add_argument("--out", help="Default: data/splits_cropped.csv")
    p.set_defaults(func=_cmd_crop)

    p = sub.add_parser("probe", parents=[common], help="Fast baseline: frozen DINOv2 features + logistic regression.")
    p.add_argument("--csv")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("train", parents=[common], help="Fine-tune the classifier.")
    p.add_argument("--csv")
    p.add_argument("--run-name", help="Folder name under models/ (default: timestamp_backbone).")
    p.add_argument("--epochs", type=int, help="Override train.epochs from the config.")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("evaluate", parents=[common], help="Score a run on the test split and write an error analysis.")
    p.add_argument("run_dir")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--csv", help="Default: the CSV the run was trained on.")
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("predict", parents=[common], help="Classify new photos with a trained run.")
    p.add_argument("run_dir")
    p.add_argument("images", nargs="+")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--no-localize", action="store_true", help="Do not crop to the ribbon first.")
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args, load_config(args.config))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
