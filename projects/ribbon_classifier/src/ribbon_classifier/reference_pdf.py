"""Pull candidate training images (and nearby text) out of the reference PDF.

The output is a CSV you finish in Excel: each extracted image gets a row with
its page number and that page's text, which usually names the technique shown.
Fill in the ``label`` column (or delete rows that are logos, icons, tables),
then feed the CSV to ``ribbon manifest --csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")


def extract_reference_pdf(
    pdf_path: str | Path,
    out_dir: str | Path,
    data_dir: str | Path,
    min_side_px: int = 128,
    render_pages_dpi: float | None = None,
) -> pd.DataFrame:
    """Save every embedded image at least *min_side_px* on its shorter side.

    If *render_pages_dpi* is set, each whole page is also saved as an image -
    useful when the PDF is a scan (one big picture per page) or when figures
    are vector drawings rather than embedded photos.
    """
    pdf_path, out_dir, data_dir = Path(pdf_path), Path(out_dir), Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    document = pdfium.PdfDocument(pdf_path)
    try:
        for page_index in range(len(document)):
            page = document[page_index]
            textpage = page.get_textpage()
            text = " ".join(textpage.get_text_range().split())
            textpage.close()
            page_no = page_index + 1
            images: list[tuple[str, Image.Image]] = []
            for k, obj in enumerate(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]), start=1):
                try:
                    image = obj.get_bitmap(render=True).to_pil()  # applies masks and transforms
                except Exception:
                    try:
                        image = obj.get_bitmap(render=False).to_pil()
                    except Exception:
                        continue  # unusual encoding pdfium cannot decode; skip it
                if min(image.size) >= min_side_px:
                    images.append((f"page_{page_no:03d}_img_{k:02d}.png", image))
            if render_pages_dpi:
                bitmap = page.render(scale=render_pages_dpi / 72, draw_annots=False)
                images.append((f"page_{page_no:03d}_full.png", bitmap.to_pil()))
            page.close()

            for name, image in images:
                target = out_dir / name
                image = _to_rgb(image)
                image.save(target)
                rows.append(
                    {
                        "image_path": target.resolve().relative_to(data_dir.resolve()).as_posix(),
                        "label": "",
                        "group": f"refdoc_p{page_no:03d}",  # images on one page usually show one technique
                        "source": "reference_pdf",
                        "page": page_no,
                        "width": image.width,
                        "height": image.height,
                        "page_text": text[:400],
                    }
                )
    finally:
        document.close()

    df = pd.DataFrame(rows, columns=["image_path", "label", "group", "source", "page", "width", "height", "page_text"])
    # utf-8-sig adds a byte-order mark so Excel opens accented text correctly.
    df.to_csv(out_dir / "reference_candidates.csv", index=False, encoding="utf-8-sig")
    return df
