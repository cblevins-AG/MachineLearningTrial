import numpy as np
import pytest

from layer_diff.compare import compare_pages, compare_renders
from layer_diff.config import Settings
from layer_diff.pdf_layers import render_layer

SETTINGS = Settings(dpi=150)


def _blank(h: int = 300, w: int = 300) -> np.ndarray:
    return np.full((h, w, 3), 255, np.uint8)


def test_identical_images_score_zero():
    image = _blank()
    image[50:100, 50:150] = (200, 0, 0)
    diff = compare_pages(image, image.copy(), SETTINGS)
    assert diff.score_pct == 0.0 and diff.regions == [] and diff.ssim == 1.0


def test_added_and_removed_are_classified():
    a, b = _blank(), _blank()
    a[20:60, 20:60] = 0  # only in A -> removed
    b[200:260, 200:260] = 0  # only in B -> added
    diff = compare_pages(a, b, SETTINGS)
    assert sorted(r.kind for r in diff.regions) == ["added", "removed"]
    # everything that has ink differs, so the score is 100%
    assert diff.score_pct == pytest.approx(100.0)


def test_recolour_is_changed_with_colours():
    a, b = _blank(), _blank()
    a[50:150, 50:150] = (220, 30, 35)
    b[50:150, 50:150] = (20, 30, 90)
    (region,) = compare_pages(a, b, SETTINGS).regions
    assert region.kind == "changed"
    assert region.color_before == (220, 30, 35) and region.color_after == (20, 30, 90)


def test_small_move_is_detected_with_offset():
    a, b = _blank(), _blank()
    a[100:140, 100:130] = 0
    b[100:140, 112:142] = 0  # 12 px to the right
    (region,) = compare_pages(a, b, SETTINGS).regions
    assert region.kind == "moved" and region.shift_px == (12, 0)


def test_far_move_absorbs_the_destination_region():
    a, b = _blank(400, 400), _blank(400, 400)
    a[50:70, 50:90] = 0
    b[50:70, 50 + 45 : 90 + 45] = 0  # 45 px (~7.6 mm at 150 DPI) away: two separate blobs
    diff = compare_pages(a, b, SETTINGS)
    assert [r.kind for r in diff.regions] == ["moved"]
    assert diff.regions[0].shift_px == (45, 0)


def test_specks_below_min_region_are_ignored():
    a, b = _blank(), _blank()
    a[100:140, 100:140] = 0
    b[100:140, 100:140] = 0
    b[10, 10] = 0  # one stray pixel
    assert compare_pages(a, b, SETTINGS).regions == []


def test_different_sizes_are_padded():
    a = _blank(300, 300)
    b = _blank(320, 300)
    diff = compare_pages(a, b, SETTINGS)
    assert diff.pixel_kind.shape == (320, 300) and diff.score_pct == 0.0


def test_sample_set_pairwise_matrix(sample_pdfs):
    renders = [render_layer(p, "Dieline", dpi=100) for p in sample_pdfs]
    comparison = compare_renders(renders, Settings(dpi=100))
    matrix = comparison.matrix()
    assert len(comparison.pairs) == 3
    assert matrix[0][0] == 0.0 and matrix[0][1] == matrix[1][0] and matrix[0][1] > 0
    kinds_v1_v2 = [r.kind for p in comparison.pairs[0].pages for r in p.regions]
    kinds_v1_v3 = [r.kind for p in comparison.pairs[1].pages for r in p.regions]
    assert kinds_v1_v2 == ["moved"]
    assert kinds_v1_v3 == ["removed"]


def test_baseline_mode_compares_against_first_file(sample_pdfs):
    renders = [render_layer(p, "Varnish", dpi=72) for p in sample_pdfs]
    comparison = compare_renders(renders, Settings(dpi=72, mode="baseline"))
    assert [(p.file_a.name, p.file_b.name) for p in comparison.pairs] == [
        ("label_v1.pdf", "label_v2.pdf"),
        ("label_v1.pdf", "label_v3.pdf"),
    ]
    assert comparison.pairs[0].score_pct == 0.0 and comparison.pairs[1].score_pct > 0


def test_missing_layer_gives_none_score(sample_pdfs, tmp_path):
    from layer_diff.samples import make_layered_pdf

    other = make_layered_pdf(tmp_path / "no_varnish.pdf", {"Artwork": b"0 g 10 10 50 50 re f"})
    renders = [render_layer(sample_pdfs[0], "Varnish"), render_layer(other, "Varnish")]
    (pair,) = compare_renders(renders, SETTINGS).pairs
    assert pair.score_pct is None and "no_varnish.pdf" in pair.notes[0]
