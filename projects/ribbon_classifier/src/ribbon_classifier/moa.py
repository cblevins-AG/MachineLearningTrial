"""Turn a reference deck of "method of assembly" (MOA) examples into training labels.

Some reference documents describe each example design as a *list* of assembly
operations with counts ("cut into 2 pieces", "insert through 3x", "tie 1x")
rather than one technique name. Two hand-written YAML files capture such a
deck (see ``data/reference/README.md`` for their format):

* the **catalogue**: every operation, the regexes that recognise its phrase,
  whether it carries a count, plus the older one-phrase attachment methods;
* the **designs**: one entry per example page, with the phrases copied as
  printed and the image files ``ribbon extract-reference`` wrote for it.

:func:`build_moa_labels` checks every phrase against the catalogue and writes
one row per image with a count column per operation (multi-label targets) and
a single ``label`` column, so the file also feeds ``ribbon manifest --csv``.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

LABEL_CHOICES = ("legacy", "section", "none")


@dataclass
class Operation:
    id: str
    family: str
    counted: bool
    max_count: int | None
    patterns: list[re.Pattern]


@dataclass
class Catalog:
    operations: list[Operation]
    legacy: list[Operation] = field(default_factory=list)

    @property
    def operation_ids(self) -> list[str]:
        return [op.id for op in self.operations]


def _normalise(phrase: str) -> str:
    """Lower-case, straight quotes/apostrophes, single spaces, no trailing full stop."""
    text = phrase.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"')
    return " ".join(text.split()).strip().rstrip(".").casefold()


def _compile(entries: list[dict], kind: str) -> list[Operation]:
    ops = []
    for entry in entries:
        if not entry.get("patterns"):
            raise ValueError(f"Catalogue {kind} '{entry.get('id')}' has no patterns.")
        ops.append(
            Operation(
                id=entry["id"],
                family=entry.get("family", kind),
                counted=bool(entry.get("counted", False)),
                max_count=entry.get("max_count"),
                patterns=[re.compile(p, re.IGNORECASE) for p in entry["patterns"]],
            )
        )
    return ops


def load_catalog(path: str | Path) -> Catalog:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Catalog(_compile(raw["operations"], "operation"), _compile(raw.get("legacy_methods", []), "legacy"))


def _match(phrase: str, ops: list[Operation]) -> tuple[Operation, re.Match] | None:
    text = _normalise(phrase)
    for op in ops:
        for pattern in op.patterns:
            found = pattern.search(text)
            if found:
                return op, found
    return None


def parse_operation(phrase: str, catalog: Catalog) -> tuple[str, int]:
    """``"Insert Thru 3x's"`` -> ``("insert_thru", 3)``. Uncounted operations count 1."""
    hit = _match(phrase, catalog.operations)
    if hit is None:
        raise ValueError(f"Unknown MOA phrase: {phrase!r}")
    op, found = hit
    if not op.counted:
        return op.id, 1
    if not found.groups() or found.group(1) is None:
        raise ValueError(f"MOA phrase {phrase!r} matched '{op.id}' but no count was captured.")
    return op.id, int(found.group(1))


def parse_legacy(phrase: str, catalog: Catalog) -> str:
    hit = _match(phrase, catalog.legacy)
    if hit is None:
        raise ValueError(f"Unknown legacy attachment method: {phrase!r}")
    return hit[0].id


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def build_moa_labels(
    designs_path: str | Path, catalog_path: str | Path, data_dir: str | Path, label_by: str = "legacy"
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (one row per image, one row per design, problems for a person to review).

    Counts are summed over a design's ribbons: two ribbons each "cut into 1 piece"
    give ``cut_pieces == 2``. ``label`` is the design's single legacy method
    (``label_by="legacy"``; left empty when its ribbons disagree), its section
    heading (``"section"``), or empty (``"none"``).
    """
    if label_by not in LABEL_CHOICES:
        raise ValueError(f"label_by must be one of {LABEL_CHOICES}, not {label_by!r}")
    catalog = load_catalog(catalog_path)
    spec = yaml.safe_load(Path(designs_path).read_text(encoding="utf-8"))
    data_dir = Path(data_dir)
    images_dir = spec["source"]["images_dir"]
    op_ids = catalog.operation_ids
    problems: list[str] = []
    design_rows, image_rows = [], []

    entries = [(d, "design", "") for d in spec.get("designs", [])] + [
        (d, "illustration", f"_ill{n}") for n, d in enumerate(spec.get("illustrations", []), start=1)
    ]
    for design, kind, suffix in entries:
        where = f"page {design['page']}" + (f" (material {design['material']})" if design.get("material") else "")
        counts = dict.fromkeys(op_ids, 0)
        sequences, legacy_ids = [], set()
        for ribbon in design["ribbons"]:
            steps = []
            for phrase in ribbon.get("moas", []):
                try:
                    op_id, n = parse_operation(phrase, catalog)
                except ValueError as error:
                    problems.append(f"{where}: {error}")
                    continue
                op = next(o for o in catalog.operations if o.id == op_id)
                if op.max_count and n > op.max_count:
                    problems.append(f"{where}: {phrase!r} is above the catalogue's top bucket ({op.max_count}+).")
                counts[op_id] += n
                steps.append(f"{op_id}={n}" if op.counted else op_id)
            for phrase in ribbon.get("simplified", []):
                try:
                    parse_operation(phrase, catalog)  # validate only; not a label
                except ValueError as error:
                    problems.append(f"{where} (simplified): {error}")
            legacy = ribbon.get("legacy") or []
            for phrase in [legacy] if isinstance(legacy, str) else legacy:
                try:
                    legacy_ids.add(parse_legacy(phrase, catalog))
                except ValueError as error:
                    problems.append(f"{where}: {error}")
            name = ribbon.get("name")
            sequences.append((f"{name}: " if name else "") + " > ".join(steps))

        group = f"moa_p{design['page']:03d}{suffix}"
        if label_by == "legacy":
            label = next(iter(legacy_ids)) if len(legacy_ids) == 1 else ""
            if kind == "design" and len(legacy_ids) != 1:
                problems.append(f"{where}: {len(legacy_ids)} legacy methods {sorted(legacy_ids)}; label left empty.")
        elif label_by == "section":
            label = _slug(design.get("section", ""))
        else:
            label = ""
        row = {
            "group": group,
            "kind": kind,
            "page": design["page"],
            "material": str(design.get("material") or ""),
            "section": design.get("section", ""),
            "production_pdf": design.get("production_pdf", ""),
            "n_ribbons": len(design["ribbons"]),
            "legacy_methods": "+".join(sorted(legacy_ids)),
            "moa_sequence": " | ".join(sequences),
            **counts,
            "notes": design.get("notes", ""),
            "questions": " / ".join(design.get("questions", [])),
        }
        design_rows.append(row)

        for image in design.get("images", []):
            image = {"file": image} if isinstance(image, str) else image
            rel = f"{images_dir}/{image['file']}"
            if not (data_dir / rel).is_file():
                problems.append(f"{where}: missing image {rel} (run `ribbon extract-reference` first).")
            image_rows.append(
                {
                    "image_path": rel,
                    "label": label,
                    "group": group,
                    "source": "reference_pdf",
                    **{k: row[k] for k in ("kind", "page", "material", "section", "legacy_methods", "moa_sequence")},
                    **counts,
                    "image_note": image.get("note", ""),
                    "same_as": image.get("same_as", ""),
                }
            )

    images = pd.DataFrame(image_rows)
    dupes = images[images.duplicated("image_path", keep=False)]
    for path in sorted(set(dupes["image_path"])):
        problems.append(f"Image listed more than once: {path}")

    # Every extracted image must be used or excluded on purpose. Files identical
    # to an excluded one (a logo repeated on every page) are excluded too.
    folder = data_dir / images_dir
    if folder.is_dir():
        excluded = set(spec.get("excluded_images", {}))
        excluded_hashes = {_sha1(folder / f) for f in excluded if (folder / f).is_file()}
        used = {Path(p).name for p in images["image_path"]}
        for path in sorted(folder.glob("*.png")):
            if path.name not in used and path.name not in excluded and _sha1(path) not in excluded_hashes:
                problems.append(f"Extracted image not used or excluded: {images_dir}/{path.name}")

    unused = [op for op in op_ids if not any(r[op] for r in design_rows)]
    if unused:
        problems.append(f"Catalogue operations with no example image: {', '.join(unused)}")
    return images, pd.DataFrame(design_rows), problems


def operation_summary(designs: pd.DataFrame, catalog_path: str | Path) -> pd.DataFrame:
    """Per operation: how many designs use it, and which per-ribbon counts appear (vs. the catalogue's range)."""
    catalog = load_catalog(catalog_path)
    per_ribbon = re.findall(r"(\w+)=(\d+)", " ".join(designs["moa_sequence"]))
    rows = []
    for op in catalog.operations:
        used = designs[designs[op.id] > 0]
        seen = sorted({int(n) for op_id, n in per_ribbon if op_id == op.id})
        rows.append(
            {
                "operation": op.id,
                "family": op.family,
                "designs": len(used),
                "counts_seen": ",".join(map(str, seen)) if op.counted else "",
                "counts_missing": ",".join(str(n) for n in range(1, (op.max_count or 0) + 1) if n not in seen)
                if op.counted
                else "",
            }
        )
    return pd.DataFrame(rows)
