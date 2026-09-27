"""Generate small layered PDFs that mimic a ribbon-packaging label.

Used by the tests, the walkthrough notebook and ``layer-diff make-samples``,
so the tool can be tried before any real production files are available.
Each file has three layers - "Artwork", "Dieline" and "Varnish":

* ``label_v1.pdf`` - the baseline
* ``label_v2.pdf`` - the dieline notch moved 7 mm up; product text 25 mm -> 38 mm
* ``label_v3.pdf`` - header band recoloured; fold line removed; varnish patch added

Page size is 4 x 6 inches; PDF coordinates are points (1/72 inch) measured
from the bottom-left corner.
"""

from __future__ import annotations

from pathlib import Path

import pikepdf
from pikepdf import Array, Dictionary, Name, String

PAGE_SIZE = (288, 432)  # 4 x 6 inches


def _circle(cx: float, cy: float, r: float) -> bytes:
    k = 0.5523 * r  # control-point distance that makes four Bezier curves look circular
    return (
        f"{cx + r} {cy} m "
        f"{cx + r} {cy + k} {cx + k} {cy + r} {cx} {cy + r} c "
        f"{cx - k} {cy + r} {cx - r} {cy + k} {cx - r} {cy} c "
        f"{cx - r} {cy - k} {cx - k} {cy - r} {cx} {cy - r} c "
        f"{cx + k} {cy - r} {cx + r} {cy - k} {cx + r} {cy} c f"
    ).encode()


def _artwork(band_rgb: str = "0.85 0.1 0.15", product: str = "Satin ribbon - 25 mm") -> bytes:
    bars = b" ".join(
        f"{x} 48 {w} 44 re".encode()
        for x, w in [(40, 3), (46, 2), (51, 4), (58, 2), (63, 3), (70, 5), (78, 2), (83, 3), (89, 4)]
    )
    return b"\n".join(
        [
            f"{band_rgb} rg 24 330 240 70 re f".encode(),  # coloured header band
            b"1 1 1 rg " + _circle(60, 365, 22),  # logo disc
            b"BT /F1 20 Tf 1 1 1 rg 95 358 Td (RIBBON CO.) Tj ET",
            f"BT /F1 14 Tf 0.1 0.1 0.1 rg 40 290 Td ({product}) Tj ET".encode(),
            b"0 g " + bars + b" f",  # barcode-like bars
        ]
    )


def _dieline(notch_y: float = 206, fold_line: bool = True) -> bytes:
    ops = [
        b"0.9 0 0.6 RG 1.2 w",  # magenta stroke, as dielines are usually shown
        b"12 12 264 408 re S",  # outer cut
        f"12 {notch_y} m 32 {notch_y + 10} l 12 {notch_y + 20} l S".encode(),  # V-notch on the left edge
    ]
    if fold_line:
        ops.append(b"[6 4] 0 d 12 150 m 276 150 l S [] 0 d")  # dashed fold line
    return b"\n".join(ops)


def _varnish(extra_patch: bool = False) -> bytes:
    ops = [b"0.7 0.9 1 rg 36 341 48 48 re f"]  # spot varnish over the logo
    if extra_patch:
        ops.append(b"0.7 0.9 1 rg 150 40 100 60 re f")
    return b"\n".join(ops)


def make_layered_pdf(
    path: str | Path,
    layers: dict[str, bytes],
    unlayered: bytes = b"",
    page_size: tuple[float, float] = PAGE_SIZE,
    pages: int = 1,
) -> Path:
    """Write a PDF whose content streams are wrapped in named optional content groups."""
    path = Path(path)
    pdf = pikepdf.new()
    font = pdf.make_indirect(
        Dictionary(Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica, Encoding=Name.WinAnsiEncoding)
    )
    ocgs = [pdf.make_indirect(Dictionary(Type=Name.OCG, Name=String(name))) for name in layers]
    properties = Dictionary({f"/L{i}": ocg for i, ocg in enumerate(ocgs)})
    parts = [unlayered] if unlayered else []
    parts += [b"/OC /L%d BDC\n%s\nEMC" % (i, ops) for i, ops in enumerate(layers.values())]
    for _ in range(pages):
        page = pdf.add_blank_page(page_size=page_size)
        page.obj.Resources = Dictionary(Properties=properties, Font=Dictionary(F1=font))
        page.obj.Contents = pdf.make_stream(b"\n".join(parts))
    pdf.Root.OCProperties = Dictionary(OCGs=Array(ocgs), D=Dictionary(Order=Array(ocgs), ON=Array(ocgs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(path, deterministic_id=True)  # same input -> byte-identical file
    return path


def make_samples(out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    return [
        make_layered_pdf(
            out_dir / "label_v1.pdf",
            {"Artwork": _artwork(), "Dieline": _dieline(), "Varnish": _varnish()},
        ),
        make_layered_pdf(
            out_dir / "label_v2.pdf",
            {
                "Artwork": _artwork(product="Satin ribbon - 38 mm"),
                "Dieline": _dieline(notch_y=226),
                "Varnish": _varnish(),
            },
        ),
        make_layered_pdf(
            out_dir / "label_v3.pdf",
            {
                "Artwork": _artwork(band_rgb="0.1 0.2 0.55"),
                "Dieline": _dieline(fold_line=False),
                "Varnish": _varnish(extra_patch=True),
            },
        ),
    ]
