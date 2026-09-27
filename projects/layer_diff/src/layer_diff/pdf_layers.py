"""Find the layers in a PDF and render only the ones you ask for.

How it works
------------
What Acrobat and Illustrator call a "layer" is, inside the PDF, an *Optional
Content Group* (OCG). Drawing commands on a page are wrapped in markers that
say "this belongs to OCG X", and the document carries a default configuration
(``/OCProperties /D``) listing which OCGs are ON and which are OFF.

To render a single layer we:

1. open the PDF with **pikepdf** (a structural PDF editor),
2. rewrite that default configuration *in memory* so the requested layers are
   ON and every other layer is OFF, and
3. hand the modified bytes to **pdfium** (the PDF engine inside Chrome, via
   pypdfium2), which honours the configuration when it rasterises the page.

The source file on disk is never modified. Both libraries are permissively
licensed (MPL-2.0 / Apache-2.0 / BSD), unlike the AGPL-licensed PyMuPDF.

Content that is *not* inside any layer is always drawn, whatever the layer
settings. :func:`find_unlayered_content` detects that case so it can be
reported instead of silently polluting a layer comparison.
"""

from __future__ import annotations

import fnmatch
import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium

POINTS_PER_INCH = 72.0

# A pixel counts as "ink" when it differs from pure white by more than this
# (0-255 per channel). Used only for the unlayered-content check.
_INK_THRESHOLD = 8


@dataclass(frozen=True)
class Layer:
    """One optional content group (layer) in a PDF."""

    name: str
    object_id: tuple[int, int]  # PDF object number + generation; unique even if names repeat
    visible_by_default: bool


@dataclass
class LayerRender:
    """Pixels for one layer-pattern of one PDF: one RGB array per page."""

    pdf_path: Path
    pattern: str
    matched_layers: list[str]
    pages: list[np.ndarray]  # each (height, width, 3) uint8, RGB
    page_sizes_pt: list[tuple[float, float]]  # (width, height) in PDF points (1/72 inch)
    dpi: float

    @property
    def found(self) -> bool:
        return bool(self.matched_layers)


def list_layers(pdf_path: str | Path) -> list[Layer]:
    """Return every layer defined in the PDF, in document order."""
    with pikepdf.open(pdf_path) as pdf:
        return _layers(pdf)


def _layers(pdf: pikepdf.Pdf) -> list[Layer]:
    props = pdf.Root.get("/OCProperties")
    if props is None:
        return []
    default_config = props.get("/D", pikepdf.Dictionary())
    base_on = str(default_config.get("/BaseState", "/ON")) != "/OFF"
    on_ids = {o.objgen for o in default_config.get("/ON", [])}
    off_ids = {o.objgen for o in default_config.get("/OFF", [])}

    layers: list[Layer] = []
    seen: set[tuple[int, int]] = set()
    for ocg in props.get("/OCGs", []):
        if ocg.objgen in seen:
            continue
        seen.add(ocg.objgen)
        visible = ocg.objgen in on_ids or (base_on and ocg.objgen not in off_ids)
        layers.append(Layer(str(ocg.get("/Name", "")), ocg.objgen, visible))
    return layers


def match_layers(layers: list[Layer], pattern: str) -> list[Layer]:
    """Select layers by name, case-insensitively.

    * ``"Dieline"``   exact name if one exists, otherwise every name containing it
    * ``"Cut*"``      shell-style wildcard (``*``, ``?``, ``[abc]``)
    * ``"re:^die"``   regular expression (after the ``re:`` prefix)

    Preferring an exact match means ``"Cut"`` does not also sweep in
    ``"Cut Contour"`` when a layer literally named ``Cut`` exists.
    """
    if pattern.startswith("re:"):
        regex = re.compile(pattern[3:], re.IGNORECASE)
        return [layer for layer in layers if regex.search(layer.name)]

    wanted = pattern.strip().casefold()
    if any(ch in wanted for ch in "*?["):
        return [layer for layer in layers if fnmatch.fnmatchcase(layer.name.strip().casefold(), wanted)]

    exact = [layer for layer in layers if layer.name.strip().casefold() == wanted]
    return exact or [layer for layer in layers if wanted in layer.name.casefold()]


def with_only_layers_visible(pdf: pikepdf.Pdf, visible_ids: set[tuple[int, int]]) -> bytes:
    """Serialise a copy of *pdf* whose default view shows only *visible_ids*.

    The default configuration is rebuilt from scratch, which also drops
    auto-state (/AS) rules and radio-button groups that could otherwise
    switch layers back on during rendering.
    """
    props = pdf.Root.get("/OCProperties")
    if props is not None:
        ocgs = list(props.get("/OCGs", []))
        props.D = pikepdf.Dictionary(
            Name=pikepdf.String("layer-diff isolation"),
            BaseState=pikepdf.Name.OFF,
            ON=pikepdf.Array([o for o in ocgs if o.objgen in visible_ids]),
            OFF=pikepdf.Array([o for o in ocgs if o.objgen not in visible_ids]),
        )
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def render_pdf_bytes(
    data: bytes, dpi: float, pages: list[int] | None = None
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    """Rasterise pages (0-based indices; ``None`` = all) to RGB uint8 arrays.

    Annotations (sticky notes, review comments) are not drawn: they are not
    part of the printed artwork.
    """
    document = pdfium.PdfDocument(data)
    try:
        indices = range(len(document)) if pages is None else [p for p in pages if p < len(document)]
        images, sizes = [], []
        for index in indices:
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / POINTS_PER_INCH, draw_annots=False, rev_byteorder=True)
                images.append(np.array(bitmap.to_numpy()[:, :, :3], copy=True))
                sizes.append(page.get_size())
            finally:
                page.close()
        return images, sizes
    finally:
        document.close()


def page_count(pdf_path: str | Path) -> int:
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)


def render_layer(pdf_path: str | Path, pattern: str, dpi: float = 150, pages: list[int] | None = None) -> LayerRender:
    """Render every page of *pdf_path* with only the layers matching *pattern* visible.

    If nothing matches, ``matched_layers`` is empty and ``pages`` is empty:
    callers report "layer not found" rather than comparing blank pages.
    """
    pdf_path = Path(pdf_path)
    with pikepdf.open(pdf_path) as pdf:
        matched = match_layers(_layers(pdf), pattern)
        if not matched:
            return LayerRender(pdf_path, pattern, [], [], [], dpi)
        data = with_only_layers_visible(pdf, {layer.object_id for layer in matched})
    images, sizes = render_pdf_bytes(data, dpi, pages)
    names = list(dict.fromkeys(layer.name for layer in matched))  # de-duplicate, keep order
    return LayerRender(pdf_path, pattern, names, images, sizes, dpi)


def find_unlayered_content(pdf_path: str | Path, dpi: float = 50) -> list[int]:
    """Return 1-based page numbers that draw something outside every layer.

    Renders each page with *all* layers switched off: anything still visible
    is not on a layer, and would appear in every single-layer render. A low
    DPI is plenty for a yes/no answer.
    """
    with pikepdf.open(pdf_path) as pdf:
        if not _layers(pdf):
            return []  # no layers at all: the question does not apply
        data = with_only_layers_visible(pdf, set())
    images, _ = render_pdf_bytes(data, dpi)
    return [i + 1 for i, img in enumerate(images) if (255 - img.min(axis=2) > _INK_THRESHOLD).any()]


def list_spot_colors(pdf_path: str | Path) -> list[str]:
    """Names of spot colours (Separation / DeviceN inks) used in the PDF.

    Print files often carry things like a dieline or varnish as a *spot
    colour* rather than (or as well as) a layer. If a "layer" you expect is
    missing from :func:`list_layers`, look for it here.
    """
    names: set[str] = set()
    visited: set[tuple[int, int]] = set()

    def visit_colorspace(cs: object) -> None:
        if isinstance(cs, pikepdf.Array) and len(cs) >= 2:
            family = str(cs[0])
            if family == "/Separation":
                names.add(str(cs[1]).lstrip("/"))
            elif family == "/DeviceN":
                names.update(str(n).lstrip("/") for n in cs[1])

    def visit_resources(resources: object, depth: int = 0) -> None:
        if not isinstance(resources, pikepdf.Dictionary) or depth > 8:
            return
        for cs in resources.get("/ColorSpace", pikepdf.Dictionary()).values():
            visit_colorspace(cs)
        for xobject in resources.get("/XObject", pikepdf.Dictionary()).values():
            if xobject.objgen in visited:
                continue
            visited.add(xobject.objgen)
            if xobject.get("/Subtype") == pikepdf.Name.Form:
                visit_resources(xobject.get("/Resources"), depth + 1)
            elif xobject.get("/Subtype") == pikepdf.Name.Image:
                visit_colorspace(xobject.get("/ColorSpace"))

    with pikepdf.open(pdf_path) as pdf:
        for page in pdf.pages:
            visit_resources(page.obj.get("/Resources"))
    names.discard("All")
    names.discard("None")
    return sorted(names)
