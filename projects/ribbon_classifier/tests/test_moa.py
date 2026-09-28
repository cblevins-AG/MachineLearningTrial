from pathlib import Path

import pytest
import yaml
from PIL import Image

from ribbon_classifier.moa import build_moa_labels, load_catalog, operation_summary, parse_legacy, parse_operation

CATALOG = {
    "operations": [
        {
            "id": "cut_pieces",
            "family": "cut",
            "counted": True,
            "max_count": 3,
            "patterns": [r"^cut into (\d+)\+? pieces?"],
        },
        {
            "id": "insert_thru",
            "family": "insert",
            "counted": True,
            "max_count": 4,
            "patterns": [r"^insert thru (\d+) ?x"],
        },
        {"id": "glue_flat", "family": "other", "patterns": ["^glue flat"]},
        {"id": "heat_seal", "family": "other", "patterns": ["^heat seal"]},
    ],
    "legacy_methods": [
        {"id": "weave", "patterns": ["^weave thru"]},
        {"id": "tie_knot", "patterns": ["^thru slit tie knot"]},
    ],
}


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    """Catalogue + designs YAML and three extracted images (one a repeated logo)."""
    images = tmp_path / "data" / "interim" / "deck"
    images.mkdir(parents=True)
    Image.new("RGB", (40, 40), "red").save(images / "page_002_img_01.png")  # logo
    Image.new("RGB", (40, 40), "red").save(images / "page_003_img_01.png")  # same logo again
    Image.new("RGB", (40, 40), "blue").save(images / "page_003_img_02.png")
    Image.new("RGB", (40, 40), "green").save(images / "page_004_img_01.png")
    designs = {
        "source": {"images_dir": "interim/deck"},
        "excluded_images": {"page_002_img_01.png": "logo"},
        "designs": [
            {
                "page": 3,
                "material": "111",
                "section": "Insert Thru - Multi",
                "ribbons": [{"legacy": "WEAVE THRu slits", "moas": ["Cut into 1 Piece", "Insert Thru 3x’s."]}],
                "images": ["page_003_img_02.png"],
            },
            {
                "page": 4,
                "section": "Two ribbons",
                "ribbons": [
                    {"name": "red", "legacy": "WEAVE THRU", "moas": ["Cut into 1 Piece", "Glue Flat"]},
                    {"name": "blue", "legacy": "THRU SLIT TIE KNOT", "moas": ["Cut into 2 Pieces", "Staple it"]},
                ],
                "images": [{"file": "page_004_img_01.png", "note": "front"}, "page_009_img_01.png"],
            },
        ],
    }
    (tmp_path / "catalog.yaml").write_text(yaml.safe_dump(CATALOG), encoding="utf-8")
    (tmp_path / "designs.yaml").write_text(yaml.safe_dump(designs, allow_unicode=True), encoding="utf-8")
    return tmp_path


def test_parse_operation_normalises_phrases(reference):
    catalog = load_catalog(reference / "catalog.yaml")
    assert parse_operation("Cut  into 2 Pieces", catalog) == ("cut_pieces", 2)
    assert parse_operation("Insert Thru 4 x’s.", catalog) == ("insert_thru", 4)
    assert parse_operation("Glue Flat Across", catalog) == ("glue_flat", 1)
    assert parse_legacy("Thru Slit Tie Knot - grey", catalog) == "tie_knot"
    with pytest.raises(ValueError, match="Unknown MOA phrase"):
        parse_operation("Staple it", catalog)


def test_build_moa_labels_counts_labels_and_problems(reference):
    images, designs, problems = build_moa_labels(
        reference / "designs.yaml", reference / "catalog.yaml", reference / "data"
    )
    first = images.iloc[0]
    assert first["image_path"] == "interim/deck/page_003_img_02.png" and first["label"] == "weave"
    assert (first["cut_pieces"], first["insert_thru"], first["glue_flat"]) == (1, 3, 0)
    assert first["group"] == "moa_p003" and first["source"] == "reference_pdf"

    two = designs.set_index("page").loc[4]
    assert two["cut_pieces"] == 3 and two["glue_flat"] == 1  # summed over both ribbons
    assert two["moa_sequence"] == "red: cut_pieces=1 > glue_flat | blue: cut_pieces=2"
    assert set(images[images["page"] == 4]["label"]) == {""}  # ribbons disagree on the legacy method

    text = "\n".join(problems)
    assert "Unknown MOA phrase: 'Staple it'" in text
    assert "2 legacy methods" in text
    assert "missing image interim/deck/page_009_img_01.png" in text
    assert "not used or excluded" not in text  # the repeated logo matches the excluded file's hash
    assert "no example image: heat_seal" in text

    summary = operation_summary(designs, reference / "catalog.yaml").set_index("operation")
    assert summary.loc["cut_pieces", "counts_seen"] == "1,2" and summary.loc["cut_pieces", "counts_missing"] == "3"


def test_label_by_section(reference):
    images, _, _ = build_moa_labels(
        reference / "designs.yaml", reference / "catalog.yaml", reference / "data", "section"
    )
    assert images.iloc[0]["label"] == "insert_thru_multi"
