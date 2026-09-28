# 05 · Repository inventory

Every file and folder in the repository, and what it contains. The last section lists what
gets **created on your machine** but is deliberately kept out of git. To confirm this document
still matches the repo after changes, run `python scripts/check_inventory.py`. It fails if any
tracked path isn't mentioned here.

## Tree

```
MachineLearningTrial/
├── .editorconfig  .gitattributes  .gitignore  pytest.ini  ruff.toml
├── README.md  requirements.txt  environment.yml
├── .vscode/            settings.json, extensions.json
├── docs/               01–05 guides (this file is 05)
├── requirements/       common.txt, ribbon_classifier.txt, layer_diff.txt, dev.txt, lock.txt
├── scripts/            setup_env.ps1, check_env.py, check_inventory.py
└── projects/
    ├── ribbon_classifier/          PROJECT 1: fold/cut classifier
    │   ├── README.md  pyproject.toml
    │   ├── configs/default.yaml
    │   ├── data/                   README.md; raw/{reference_pdf,examples}/, interim/, processed/
    │   ├── models/README.md
    │   ├── notebooks/              01_explore_dataset.ipynb, 02_review_model.ipynb
    │   ├── src/ribbon_classifier/  15 modules
    │   └── tests/                  conftest.py, test_data.py, test_moa.py, test_pipeline.py
    └── layer_diff/                 PROJECT 2: PDF layer difference tool
        ├── README.md  pyproject.toml
        ├── configs/default.yaml
        ├── data/                   README.md; input/, samples/ (3 PDFs), output/
        ├── notebooks/              01_layer_diff_walkthrough.ipynb
        ├── src/layer_diff/         10 modules
        └── tests/                  conftest.py + 3 test files
```

## Repository root

| Path | What it is |
|------|------------|
| `README.md` | Overview: the two projects, why they are separate, quick start, assumptions, doc index |
| `requirements.txt` | One-stop install for both projects + dev tools. Applies `requirements/lock.txt` as constraints, pulls in the per-project pin files, and installs both packages in editable mode. Run from the repo root. |
| `environment.yml` | Optional conda alternative: conda provides Python 3.14, pip installs `requirements.txt` |
| `pytest.ini` | Test configuration: where tests live (both projects), `--import-mode=importlib` so both can have same-named test files |
| `ruff.toml` | Linter/formatter rules (line length 120; pycodestyle, pyflakes, import sorting, bugbear, pyupgrade) |
| `.gitignore` | Keeps out virtual environments, caches, **data contents, trained models, outputs**, secrets (`.env`) and editor files; keeps folder READMEs, `.gitkeep` files and the sample PDFs |
| `.gitattributes` | Line-ending normalisation (LF in the repo, CRLF for `.ps1`), and marks PDFs/images/weights as binary |
| `.editorconfig` | Whitespace conventions (UTF-8, 4-space indent, 2 for YAML/JSON) for any editor |

## `.vscode/`

| Path | What it is |
|------|------------|
| `.vscode/` | Shared VS Code workspace settings (other files in this folder are git-ignored) |
| `.vscode/settings.json` | Uses `.venv\Scripts\python.exe`, enables pytest in the Testing panel, Ruff format-on-save, notebooks run from their own folder |
| `.vscode/extensions.json` | Recommended extensions: Python, Pylance, Jupyter, Ruff, YAML |

## `docs/`

| Path | What it is |
|------|------------|
| `docs/` | The five guides |
| `docs/01_GETTING_STARTED.md` | Python/ML for a C#/.NET developer: concept map, libraries, how training works end to end, building the labelled dataset from the reference PDF + examples, data storage, cheat sheet |
| `docs/02_INSTALLATION.md` | What to install on Windows 11, from which official source, and why; GPU decision table; manual equivalents of the setup script |
| `docs/03_LEARNING_RESOURCES.md` | Tiered (beginner → intermediate → advanced) resources: Python, ML fundamentals, image classification, image comparison, production/.NET |
| `docs/04_TROUBLESHOOTING.md` | Symptom → cause → fix for environment, installs, GPU/drivers, data loading, over-fitting, cropping, PDF layers, notebooks and git |
| `docs/05_REPO_INVENTORY.md` | This file |

## `requirements/`

| Path | What it is |
|------|------------|
| `requirements/` | Pinned dependency lists (the "PackageReference" layer) and the full lock |
| `requirements/common.txt` | Shared by both projects: numpy, pandas, pillow, pyyaml, tqdm, matplotlib |
| `requirements/ribbon_classifier.txt` | Project 1: torch, torchvision, timm, transformers, safetensors, scikit-learn, pypdfium2, pillow-heif (+ `common.txt`) |
| `requirements/layer_diff.txt` | Project 2: pikepdf, pypdfium2, opencv-python-headless, scikit-image, anthropic (optional use) (+ `common.txt`) |
| `requirements/dev.txt` | Dev tools: jupyterlab, ipykernel, ipywidgets, pytest, ruff, nbstripout, uv |
| `requirements/lock.txt` | Generated: all 181 packages (including indirect ones) at exact versions for Python 3.14 on Windows/macOS/Linux, with OS markers. Regeneration command at the top |

## `scripts/`

| Path | What it is |
|------|------------|
| `scripts/` | Repo-level helper scripts |
| `scripts/setup_env.ps1` | Windows setup: checks Python 3.14, creates `.venv`, detects an NVIDIA GPU and installs the matching PyTorch build (cu130 / cu126 / cpu), installs `requirements.txt`, registers nbstripout, runs `check_env.py`. Options: `-Torch`, `-PythonVersion`, `-Recreate` |
| `scripts/check_env.py` | Health report: interpreter, every package version, PyTorch build and GPU (with a real GPU calculation), PDF layer rendering self-test, model cache location, API key presence |
| `scripts/check_inventory.py` | Fails if any tracked file or folder is missing from this document |

## `projects/`

| Path | What it is |
|------|------------|
| `projects/` | One folder per tool; each is an independently installable Python package with its own data, notebooks and tests |

### Project 1: `projects/ribbon_classifier/`

| Path | What it is |
|------|------------|
| `projects/ribbon_classifier/` | The ribbon fold/cut classifier |
| `projects/ribbon_classifier/README.md` | Approach and trade-offs (transfer learning vs from scratch, localisation options), assumptions, workflow commands, deployment notes |
| `projects/ribbon_classifier/pyproject.toml` | Package definition (`ribbon-classifier`), loose dependency ranges, and the `ribbon` console command |
| `projects/ribbon_classifier/configs/` | Configuration files |
| `projects/ribbon_classifier/configs/default.yaml` | Every setting, commented: paths, split fractions, localiser prompts/threshold, backbone (default ConvNeXt-Nano), augmentation, training hyper-parameters, probe backbone |
| `projects/ribbon_classifier/data/` | All data for Project 1. Contents git-ignored; folders kept |
| `projects/ribbon_classifier/data/README.md` | What goes in each data folder and which command creates each file |
| `projects/ribbon_classifier/data/raw/` | Original, read-only inputs |
| `projects/ribbon_classifier/data/raw/reference_pdf/` | The 100-page reference document goes here |
| `projects/ribbon_classifier/data/raw/reference_pdf/.gitkeep` | Empty marker so git keeps the folder |
| `projects/ribbon_classifier/data/raw/examples/` | Labelled example photos: one sub-folder per label, optional nested sub-folder per physical ribbon |
| `projects/ribbon_classifier/data/raw/examples/.gitkeep` | Empty marker so git keeps the folder |
| `projects/ribbon_classifier/data/reference/` | Hand-written YAML for a reference deck that lists assembly operations (MOAs) per example design: `moa_catalog.yaml` (vocabulary) and `moa_designs.yaml` (one entry per example page). Git-ignored like the rest of `data/` |
| `projects/ribbon_classifier/data/reference/README.md` | Format of the two MOA YAML files and how `ribbon moa-labels` uses them |
| `projects/ribbon_classifier/data/interim/` | Intermediate files, e.g. `reference_pdf/` images + `reference_candidates.csv` from `ribbon extract-reference` |
| `projects/ribbon_classifier/data/interim/.gitkeep` | Empty marker so git keeps the folder |
| `projects/ribbon_classifier/data/processed/` | Model-ready data, e.g. `crops/` from `ribbon crop` |
| `projects/ribbon_classifier/data/processed/.gitkeep` | Empty marker so git keeps the folder |
| `projects/ribbon_classifier/models/` | Trained runs, one folder each. Contents git-ignored |
| `projects/ribbon_classifier/models/README.md` | What each file in a run folder is (`model.safetensors`, `meta.json`, `history.csv`, evaluation outputs) |
| `projects/ribbon_classifier/notebooks/` | Jupyter notebooks for looking at data and results |
| `projects/ribbon_classifier/notebooks/01_explore_dataset.ipynb` | Before training: class balance, image sizes, sample grid per class, group/leakage check |
| `projects/ribbon_classifier/notebooks/02_review_model.ipynb` | After training: learning curves, evaluation + confusion matrix, gallery of misclassified images |
| `projects/ribbon_classifier/src/` | Source root (the "src layout" keeps tests honest: they import the installed package, not loose files) |
| `projects/ribbon_classifier/src/ribbon_classifier/` | The Python package |
| `projects/ribbon_classifier/src/ribbon_classifier/__init__.py` | Package marker, version, and an overview of the pipeline steps |
| `projects/ribbon_classifier/src/ribbon_classifier/__main__.py` | Enables `python -m ribbon_classifier` |
| `projects/ribbon_classifier/src/ribbon_classifier/cli.py` | The `ribbon` command: `extract-reference`, `moa-labels`, `manifest`, `split`, `crop`, `probe`, `train`, `evaluate`, `predict` |
| `projects/ribbon_classifier/src/ribbon_classifier/config.py` | Typed config dataclasses and YAML loading; resolves relative paths against the project folder |
| `projects/ribbon_classifier/src/ribbon_classifier/device.py` | Picks CUDA / Apple MPS / CPU; seeds random generators |
| `projects/ribbon_classifier/src/ribbon_classifier/dataset.py` | Label manifest reading and validation (missing/corrupt files, typos, duplicates), folder-to-manifest conversion, EXIF-aware image loading, transforms/augmentation, PyTorch `Dataset` |
| `projects/ribbon_classifier/src/ribbon_classifier/split.py` | Stratified, grouped train/val/test split; reference-PDF images forced into train |
| `projects/ribbon_classifier/src/ribbon_classifier/reference_pdf.py` | Extracts embedded images + page text from the reference PDF into a CSV for labelling |
| `projects/ribbon_classifier/src/ribbon_classifier/moa.py` | Turns MOA catalogue + design YAML into per-image labels: one count column per assembly operation, plus a single `label` column for `ribbon manifest`; reports unknown phrases, missing or unlisted images, and operations with no example |
| `projects/ribbon_classifier/src/ribbon_classifier/localize.py` | Zero-shot ribbon detection (OWLv2) and cropping, single image or whole manifest |
| `projects/ribbon_classifier/src/ribbon_classifier/model.py` | Builds the timm backbone + new head; saves/loads runs (safetensors + meta.json) |
| `projects/ribbon_classifier/src/ribbon_classifier/probe.py` | Linear-probe baseline: frozen DINOv2 features + logistic regression |
| `projects/ribbon_classifier/src/ribbon_classifier/train.py` | Fine-tuning loop: head-only warm-up, differential learning rates, class weights, mixed precision, early stopping, best-checkpoint saving |
| `projects/ribbon_classifier/src/ribbon_classifier/evaluate.py` | Accuracy, macro-F1, per-class report, confusion-matrix image, list of errors |
| `projects/ribbon_classifier/src/ribbon_classifier/predict.py` | `Predictor` class: load a run once, classify photos (with matching cropping) |
| `projects/ribbon_classifier/tests/` | Offline tests on synthetic images with a tiny random model (no downloads) |
| `projects/ribbon_classifier/tests/conftest.py` | Fixture that builds a throwaway project: synthetic "technique" images in grouped folders + a test config |
| `projects/ribbon_classifier/tests/test_data.py` | Config defaults vs YAML, manifest building, validation messages, grouped splitting, EXIF/alpha loading, reference-PDF extraction, localiser cropping |
| `projects/ribbon_classifier/tests/test_moa.py` | MOA phrase parsing, per-design count aggregation, label choice, and the problems report (on made-up phrases) |
| `projects/ribbon_classifier/tests/test_pipeline.py` | End-to-end CLI run: manifest → split → probe → train → evaluate → predict |

### Project 2: `projects/layer_diff/`

| Path | What it is |
|------|------------|
| `projects/layer_diff/` | The PDF layer difference tool |
| `projects/layer_diff/README.md` | Approach and trade-offs (classical vs deep-feature vs vision-language model), why it's a separate tool, score definitions, usage, production-PDF caveats |
| `projects/layer_diff/pyproject.toml` | Package definition (`layer-diff`), dependency ranges, optional `cloud` extra, and the `layer-diff` console command |
| `projects/layer_diff/configs/` | Configuration files |
| `projects/layer_diff/configs/default.yaml` | Every setting, commented (DPI, colour tolerance, minimum region, merge distance, move search, mode, describer). Equal to the built-in defaults; a test enforces it |
| `projects/layer_diff/data/` | Data for Project 2 |
| `projects/layer_diff/data/README.md` | What goes in `input/`, `samples/`, `output/` |
| `projects/layer_diff/data/input/` | Your production PDFs (contents git-ignored) |
| `projects/layer_diff/data/input/.gitkeep` | Empty marker so git keeps the folder |
| `projects/layer_diff/data/output/` | Default location for reports, one timestamped folder per run (contents git-ignored) |
| `projects/layer_diff/data/output/.gitkeep` | Empty marker so git keeps the folder |
| `projects/layer_diff/data/samples/` | Synthetic layered PDFs for trying the tool (regenerate with `layer-diff make-samples`) |
| `projects/layer_diff/data/samples/label_v1.pdf` | Baseline 4×6 in label with layers Artwork, Dieline, Varnish |
| `projects/layer_diff/data/samples/label_v2.pdf` | v1 with the dieline notch moved 7 mm up and the product text changed (25 mm → 38 mm) |
| `projects/layer_diff/data/samples/label_v3.pdf` | v1 with the header band recoloured (red → navy), the fold line removed, and a varnish patch added |
| `projects/layer_diff/notebooks/` | Jupyter notebooks |
| `projects/layer_diff/notebooks/01_layer_diff_walkthrough.ipynb` | Step-by-step tour: list layers, render, measure, overlay, describe, full multi-file report |
| `projects/layer_diff/src/` | Source root |
| `projects/layer_diff/src/layer_diff/` | The Python package |
| `projects/layer_diff/src/layer_diff/__init__.py` | Package marker, version, usage example |
| `projects/layer_diff/src/layer_diff/__main__.py` | Enables `python -m layer_diff` |
| `projects/layer_diff/src/layer_diff/cli.py` | The `layer-diff` command: `inspect`, `compare`, `make-samples` |
| `projects/layer_diff/src/layer_diff/config.py` | `Settings` dataclass with documented defaults; YAML loading with overrides |
| `projects/layer_diff/src/layer_diff/pdf_layers.py` | Lists layers (OCGs), matches names/keywords/wildcards/regex, isolates layers by rewriting the default visibility config (pikepdf) and renders with pdfium; detects unlayered content; lists spot colours |
| `projects/layer_diff/src/layer_diff/compare.py` | CIELAB colour difference, region grouping, added/removed/changed/moved classification (template matching for moves), SSIM, content-normalised score, pairwise/baseline comparison and N×N matrix |
| `projects/layer_diff/src/layer_diff/describe.py` | Plain-English sentences (positions and sizes in mm, colour names, move direction); optional Claude describer with JSON-schema output and automatic local fallback |
| `projects/layer_diff/src/layer_diff/report.py` | Colour-coded overlay images with numbered boxes; JSON and Markdown report writers |
| `projects/layer_diff/src/layer_diff/pipeline.py` | End-to-end `run()`: inspect files → render → compare → describe → write report |
| `projects/layer_diff/src/layer_diff/samples.py` | Generates the layered sample PDFs (also used by tests and `check_env.py`) |
| `projects/layer_diff/tests/` | Offline tests (about 2 s) |
| `projects/layer_diff/tests/conftest.py` | Fixture that generates the sample PDFs into a temp folder |
| `projects/layer_diff/tests/test_pdf_layers.py` | Layer listing and matching rules, single-layer rendering, missing layers, unlayered content, source file untouched, spot colours |
| `projects/layer_diff/tests/test_compare.py` | Identical/added/removed/recoloured/moved (near and far) regions, noise filtering, page-size padding, pairwise matrix, baseline mode, missing layer |
| `projects/layer_diff/tests/test_describe_and_cli.py` | Colour naming, descriptions, YAML defaults, page parsing, CLI report output and errors, Claude describer request shape and fallback (with a fake client, no network) |

## Created on your machine (not in git)

| Path | Created by | Contents |
|------|------------|----------|
| `.venv\` | `scripts\setup_env.ps1` | The virtual environment: Python 3.14 + every installed package (about 1.5 GB CPU / 4 GB CUDA). Disposable |
| `%USERPROFILE%\.cache\huggingface\` | first `ribbon train` / `probe` / `crop` | Downloaded pretrained weights (about 0.8 GB for all four models); move with `HF_HOME` |
| `projects\ribbon_classifier\data\raw\…` | you | The reference PDF and example photos |
| `projects\ribbon_classifier\data\interim\reference_pdf\` | `ribbon extract-reference` | Extracted images + `reference_candidates.csv` |
| `projects\ribbon_classifier\data\labels.csv` | `ribbon manifest` | All labelled images |
| `projects\ribbon_classifier\data\splits.csv` | `ribbon split` | + train/val/test column |
| `projects\ribbon_classifier\data\processed\crops\`, `data\splits_cropped.csv` | `ribbon crop` | Ribbon crops and the manifest pointing at them |
| `projects\ribbon_classifier\models\<timestamp>_<backbone>\` | `ribbon train` / `ribbon evaluate` | `model.safetensors`, `meta.json`, `history.csv`, `eval_*.json`, `confusion_*.png`, `errors_*.csv` |
| `projects\ribbon_classifier\models\probe_<timestamp>\` | `ribbon probe` | `probe_report.json` |
| `projects\layer_diff\data\input\*.pdf` | you | Production PDFs to compare |
| `projects\layer_diff\data\output\<timestamp>\` | `layer-diff compare` | `report.md`, `report.json`, `images\<layer>\` renders and overlays |
| `projects\*\src\*.egg-info\` | editable install | Package metadata for pip |
| `__pycache__\`, `.pytest_cache\`, `.ruff_cache\`, `.ipynb_checkpoints\` | Python / tools | Caches, safe to delete |
