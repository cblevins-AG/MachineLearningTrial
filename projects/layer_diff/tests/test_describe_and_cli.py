import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from layer_diff.cli import main, parse_pages
from layer_diff.compare import compare_pages, compare_pair
from layer_diff.config import Settings, load_settings
from layer_diff.describe import ClaudeDescriber, LocalDescriber, color_name
from layer_diff.pdf_layers import render_layer

DEFAULT_YAML = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_color_names():
    assert color_name((0, 0, 0)) == "black"
    assert color_name((215, 35, 40)) == "red"
    assert color_name((25, 35, 95)) == "navy"


def test_local_description_mentions_move_direction_and_distance(sample_pdfs):
    settings = Settings(dpi=150)
    pair = compare_pair(render_layer(sample_pdfs[0], "Dieline"), render_layer(sample_pdfs[1], "Dieline"), settings)
    text = LocalDescriber().describe(pair, settings)
    assert "label_v1.pdf vs label_v2.pdf" in text.summary
    assert len(text.details) == 1 and "Moved" in text.details[0] and "mm up" in text.details[0]


def test_default_yaml_matches_code_defaults():
    assert load_settings(DEFAULT_YAML) == Settings()
    assert set(yaml.safe_load(DEFAULT_YAML.read_text(encoding="utf-8"))) == set(Settings().__dict__)


def test_parse_pages():
    assert parse_pages("1,3-4") == [0, 2, 3]
    assert parse_pages(None) is None


def test_cli_compare_writes_reports(sample_pdfs, tmp_path, capsys):
    out = tmp_path / "out"
    code = main(["compare", *map(str, sample_pdfs), "--layer", "Varnish", "--dpi", "72", "-o", str(out)])
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    layer = report["layers"][0]
    assert layer["pattern"] == "Varnish" and len(layer["pairs"]) == 3
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Layer difference report")
    assert (out / layer["pairs"][1]["pages"][0]["overlay"]).is_file()
    assert "overall difference" in capsys.readouterr().out


def test_cli_reports_missing_file_cleanly(tmp_path, capsys):
    assert main(["compare", str(tmp_path / "a.pdf"), str(tmp_path / "b.pdf"), "-l", "x"]) == 2
    assert "Not found" in capsys.readouterr().err


class _FakeMessages:
    def __init__(self, reply: dict | None, error: Exception | None = None):
        self.reply, self.error, self.kwargs = reply, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        block = SimpleNamespace(type="text", text=json.dumps(self.reply))
        return SimpleNamespace(content=[block], stop_reason="end_turn", model=kwargs["model"])


def _changed_pair():
    a = np.full((200, 200, 3), 255, np.uint8)
    b = a.copy()
    b[50:100, 50:100] = 0
    settings = Settings(dpi=150)
    from layer_diff.pdf_layers import LayerRender

    ra = LayerRender(Path("a.pdf"), "Cut", ["Cut"], [a], [(96.0, 96.0)], 150)
    rb = LayerRender(Path("b.pdf"), "Cut", ["Cut"], [b], [(96.0, 96.0)], 150)
    pair = compare_pair(ra, rb, settings)
    overlay = a
    return pair, settings, (a, b, overlay)


def test_claude_describer_builds_request_and_parses_json():
    pair, settings, images = _changed_pair()
    messages = _FakeMessages(
        {"summary": "A square was added.", "differences": [{"region_id": 1, "description": "New square."}]}
    )
    describer = ClaudeDescriber(model="claude-opus-5", client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    text = describer.describe(pair, settings, images=images)
    assert text.summary == "A square was added." and text.details == ["#1 New square."]
    assert text.source == "claude-opus-5"
    content = messages.kwargs["messages"][0]["content"]
    assert [block["type"] for block in content] == ["image", "image", "image", "text"]
    assert messages.kwargs["output_config"]["format"]["type"] == "json_schema"


def test_claude_describer_falls_back_to_local_on_error():
    pair, settings, images = _changed_pair()
    messages = _FakeMessages(None, error=RuntimeError("no network"))
    describer = ClaudeDescriber(client=SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    text = describer.describe(pair, settings, images=images)
    assert text.source == "local" and "Added" in text.details[0]
    assert "no network" in text.details[-1]


def test_compare_pages_does_not_modify_inputs():
    a = np.full((50, 50, 3), 255, np.uint8)
    b = a.copy()
    b[10:20, 10:20] = 0
    before = b.copy()
    compare_pages(a, b, Settings())
    assert np.array_equal(b, before)
