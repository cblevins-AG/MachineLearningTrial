from pathlib import Path

import pandas as pd
import pikepdf
import yaml
from PIL import Image

from ribbon_classifier.config import DEFAULT_CONFIG, Config, LocalizerConfig, load_config
from ribbon_classifier.dataset import load_image, manifest_from_folders, read_manifest, validate_manifest
from ribbon_classifier.localize import RibbonLocalizer
from ribbon_classifier.reference_pdf import extract_reference_pdf
from ribbon_classifier.split import assign_splits


def test_default_yaml_matches_code_defaults():
    loaded = load_config(DEFAULT_CONFIG)
    assert loaded.to_dict() == Config().to_dict()
    assert set(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))) == set(Config().to_dict())


def test_config_paths_resolve_against_config_location(project):
    config = load_config(project / "configs" / "test.yaml")
    assert config.path("data") == project / "data"
    assert config.model.backbone == "test_convnext" and config.train.epochs == 3
    assert config.train.patience == 2 and config.augment.horizontal_flip  # unspecified keys keep defaults


def test_manifest_from_folders_uses_subfolders_as_groups(project):
    df = manifest_from_folders(project / "data" / "raw" / "examples", project / "data")
    assert len(df) == 36 and set(df["label"]) == {"box_pleat", "dovetail_cut", "loop_fold"}
    first = df.iloc[0]
    assert first["image_path"].startswith("raw/examples/") and first["group"] == "box_pleat/roll_0"
    assert df.groupby("group").size().eq(2).all()


def test_validate_manifest_reports_problems(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    Image.new("RGB", (32, 32), "red").save(data / "a.png")
    (data / "b.png").write_bytes((data / "a.png").read_bytes())  # exact duplicate of a.png
    (data / "broken.jpg").write_bytes(b"not an image")
    pd.DataFrame(
        {
            "image_path": ["a.png", "b.png", "broken.jpg", "missing.png", "a.png"],
            "label": ["Box Pleat", "box_pleat", "box_pleat", "box_pleat", ""],
            "group": ["g1", "g2", "g3", "g4", "g5"],
        }
    ).to_csv(tmp_path / "labels.csv", index=False)

    clean, problems = validate_manifest(read_manifest(tmp_path / "labels.csv"), data)
    text = "\n".join(problems)
    assert list(clean["image_path"]) == ["a.png", "b.png"]
    assert clean["group"].nunique() == 1  # duplicates merged into one group
    assert "Missing file: missing.png" in text and "Unreadable image: broken.jpg" in text
    assert "look like the same class" in text and "different labels" in text and "no label" in text


def test_split_keeps_groups_together_and_forces_reference_to_train(project):
    df = manifest_from_folders(project / "data" / "raw" / "examples", project / "data")
    reference = pd.DataFrame(
        [
            {"image_path": f"ref_{i}.png", "label": "box_pleat", "group": f"refdoc_p{i}", "source": "reference_pdf"}
            for i in range(3)
        ]
    )
    out = assign_splits(pd.concat([df, reference]), 0.2, 0.2, seed=1, train_only_sources=["reference_pdf"])
    assert out.groupby("group")["split"].nunique().eq(1).all()  # no group in two splits
    assert set(out.loc[out["source"] == "reference_pdf", "split"]) == {"train"}
    assert set(out["split"]) == {"train", "val", "test"}
    assert (out.groupby(["split"])["label"].nunique() == 3).all()  # every class in every split


def test_load_image_applies_exif_rotation_and_flattens_alpha(tmp_path):
    image = Image.new("RGB", (40, 20), "blue")
    exif = image.getexif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    image.save(tmp_path / "rotated.jpg", exif=exif)
    assert load_image(tmp_path / "rotated.jpg").size == (20, 40)

    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    assert load_image(tmp_path / "clear.png").getpixel((5, 5)) == (255, 255, 255)


def _pdf_with_image(path: Path) -> None:
    pdf = pikepdf.new()
    page = pdf.add_blank_page(page_size=(300, 300))
    pixels = Image.new("RGB", (200, 150), (180, 20, 60)).tobytes()
    image = pikepdf.Stream(
        pdf,
        pixels,
        Type=pikepdf.Name.XObject,
        Subtype=pikepdf.Name.Image,
        Width=200,
        Height=150,
        ColorSpace=pikepdf.Name.DeviceRGB,
        BitsPerComponent=8,
    )
    font = pikepdf.Dictionary(Type=pikepdf.Name.Font, Subtype=pikepdf.Name.Type1, BaseFont=pikepdf.Name.Helvetica)
    page.obj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im1=image), Font=pikepdf.Dictionary(F1=font))
    page.obj.Contents = pdf.make_stream(
        b"q 200 0 0 150 50 120 cm /Im1 Do Q BT /F1 12 Tf 50 100 Td (Figure 3: Dovetail cut) Tj ET"
    )
    pdf.save(path)


def test_extract_reference_pdf(tmp_path):
    _pdf_with_image(tmp_path / "ref.pdf")
    out = tmp_path / "data" / "interim" / "reference_pdf"
    df = extract_reference_pdf(tmp_path / "ref.pdf", out, tmp_path / "data")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["image_path"] == "interim/reference_pdf/page_001_img_01.png"
    assert "Dovetail cut" in row["page_text"] and row["group"] == "refdoc_p001" and row["label"] == ""
    assert (out / "reference_candidates.csv").is_file()


def test_localizer_crops_best_box_with_margin():
    fake = lambda image, candidate_labels, threshold: [  # noqa: E731 - stands in for the OWLv2 pipeline
        {"score": 0.3, "label": candidate_labels[0], "box": {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20}},
        {"score": 0.8, "label": candidate_labels[1], "box": {"xmin": 40, "ymin": 30, "xmax": 80, "ymax": 70}},
    ]
    localizer = RibbonLocalizer(LocalizerConfig(margin=0.25), detector=fake)
    crop, box = localizer.crop(Image.new("RGB", (100, 100)))
    assert box.score == 0.8 and crop.size == (60, 60)  # 40x40 box + 25% margin on each side

    nothing = RibbonLocalizer(LocalizerConfig(), detector=lambda *a, **k: [])
    image = Image.new("RGB", (50, 50))
    assert nothing.crop(image) == (image, None)
