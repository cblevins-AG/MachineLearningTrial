"""End-to-end smoke test through the CLI: manifest -> split -> probe -> train -> evaluate -> predict."""

import json

from ribbon_classifier.cli import main
from ribbon_classifier.predict import Predictor


def test_full_pipeline_on_synthetic_data(project, capsys):
    config = str(project / "configs" / "test.yaml")
    common = ["--config", config, "--device", "cpu"]

    assert main(["manifest", *common, "--folders", str(project / "data" / "raw" / "examples")]) == 0
    assert (project / "data" / "labels.csv").is_file()

    assert main(["split", *common]) == 0
    assert (project / "data" / "splits.csv").is_file()

    assert main(["probe", *common]) == 0
    assert list((project / "models").glob("probe_*/probe_report.json"))

    assert main(["train", *common, "--run-name", "smoke"]) == 0
    run_dir = project / "models" / "smoke"
    for name in ("model.safetensors", "meta.json", "history.csv"):
        assert (run_dir / name).is_file()
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["classes"] == ["box_pleat", "dovetail_cut", "loop_fold"] and meta["trained_on_crops"] is False

    assert main(["evaluate", *common, str(run_dir)]) == 0
    for name in ("eval_test.json", "confusion_test.png", "errors_test.csv"):
        assert (run_dir / name).is_file()

    image = project / "data" / "raw" / "examples" / "loop_fold" / "roll_0" / "img_0.png"
    capsys.readouterr()
    assert main(["predict", *common, str(run_dir), str(image)]) == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["prediction"] in meta["classes"] and 0 <= result["confidence"] <= 1

    # Same answer through the Python API, and probabilities sum to ~1.
    direct = Predictor(run_dir, device="cpu").predict(image, top_k=3)
    assert direct["prediction"] == result["prediction"]
    assert abs(sum(item["probability"] for item in direct["top_k"]) - 1) < 1e-3


def test_cli_errors_are_friendly(project, capsys):
    config = str(project / "configs" / "test.yaml")
    assert main(["train", "--config", config, "--csv", str(project / "nope.csv")]) == 2
    assert "error:" in capsys.readouterr().err
