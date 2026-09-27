import numpy as np

from layer_diff.pdf_layers import (
    Layer,
    find_unlayered_content,
    list_layers,
    list_spot_colors,
    match_layers,
    render_layer,
)
from layer_diff.samples import make_layered_pdf

RED_BOX = b"1 0 0 rg 20 20 60 60 re f"
BLUE_BOX = b"0 0 1 rg 100 100 50 50 re f"


def _layers(*names: str) -> list[Layer]:
    return [Layer(name, (i + 1, 0), True) for i, name in enumerate(names)]


def test_list_layers_reads_names_in_order(sample_pdfs):
    assert [layer.name for layer in list_layers(sample_pdfs[0])] == ["Artwork", "Dieline", "Varnish"]


def test_match_prefers_exact_name_over_substring():
    layers = _layers("Cut", "Cut Contour", "Kiss Cut")
    assert [layer.name for layer in match_layers(layers, "cut")] == ["Cut"]


def test_match_falls_back_to_substring_then_supports_glob_and_regex():
    layers = _layers("Die Line", "Dieline 2", "Artwork")
    assert [layer.name for layer in match_layers(layers, "dieline")] == ["Dieline 2"]
    assert [layer.name for layer in match_layers(layers, "die*")] == ["Die Line", "Dieline 2"]
    assert [layer.name for layer in match_layers(layers, r"re:^die\s?line")] == ["Die Line", "Dieline 2"]
    assert match_layers(layers, "varnish") == []


def test_render_layer_draws_only_the_requested_layer(tmp_path):
    pdf = make_layered_pdf(tmp_path / "two.pdf", {"Red": RED_BOX, "Blue": BLUE_BOX}, page_size=(200, 200))
    red = render_layer(pdf, "red", dpi=72).pages[0]
    blue = render_layer(pdf, "blue", dpi=72).pages[0]

    assert red.shape == (200, 200, 3)
    is_red = (red[..., 0] > 200) & (red[..., 2] < 50)
    is_blue_in_red_render = (red[..., 2] > 200) & (red[..., 0] < 50)
    assert is_red.sum() > 3000 and is_blue_in_red_render.sum() == 0

    is_blue = (blue[..., 2] > 200) & (blue[..., 0] < 50)
    is_red_in_blue_render = (blue[..., 0] > 200) & (blue[..., 2] < 50)
    assert is_blue.sum() > 2000 and is_red_in_blue_render.sum() == 0


def test_render_layer_reports_missing_layer(sample_pdfs):
    render = render_layer(sample_pdfs[0], "White Ink")
    assert not render.found and render.pages == []


def test_unlayered_content_is_detected(tmp_path):
    clean = make_layered_pdf(tmp_path / "clean.pdf", {"Blue": BLUE_BOX}, page_size=(200, 200))
    dirty = make_layered_pdf(tmp_path / "dirty.pdf", {"Blue": BLUE_BOX}, unlayered=RED_BOX, page_size=(200, 200))
    assert find_unlayered_content(clean) == []
    assert find_unlayered_content(dirty) == [1]


def test_source_file_is_not_modified(sample_pdfs):
    before = sample_pdfs[0].read_bytes()
    render_layer(sample_pdfs[0], "Dieline")
    assert sample_pdfs[0].read_bytes() == before


def test_spot_colors_found(tmp_path):
    ops = b"/CS0 cs 1 scn 10 10 50 50 re f"
    pdf = make_layered_pdf(tmp_path / "spot.pdf", {"Cut": ops}, page_size=(100, 100))
    import pikepdf

    with pikepdf.open(pdf, allow_overwriting_input=True) as doc:
        tint = doc.make_stream(b"{dup 0 exch 0}", FunctionType=4, Domain=[0, 1], Range=[0, 1, 0, 1, 0, 1, 0, 1])
        separation = pikepdf.Array(
            [pikepdf.Name.Separation, pikepdf.Name("/CutContour"), pikepdf.Name.DeviceCMYK, tint]
        )
        doc.pages[0].obj.Resources.ColorSpace = pikepdf.Dictionary(CS0=separation)
        doc.save(pdf)
    assert list_spot_colors(pdf) == ["CutContour"]
    assert np.asarray(render_layer(pdf, "cut").pages[0]).min() < 255  # still renders
