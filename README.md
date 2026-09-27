# MachineLearningTrial: two computer-vision tools

| | Project | What it does | Core technique |
|-|---------|--------------|----------------|
| 1 | [**Ribbon fold/cut classifier**](projects/ribbon_classifier/README.md) | Takes a photo, finds the ribbon, says which fold or cut technique was used (with a confidence) | Deep learning: zero-shot detection (OWLv2) + transfer learning (fine-tuned ConvNeXt) |
| 2 | [**PDF layer difference tool**](projects/layer_diff/README.md) | Compares a named layer (e.g. `Dieline`) across 2..N production PDFs, describes each difference in plain English, and scores how different they are (%) | Classical computer vision on vector renders (CIELAB colour difference, region analysis, SSIM); optional Claude describer |

## Why two separate tools

They answer different kinds of question:

* **Project 1** *learns* from labelled examples. It needs a training loop, labelled data,
  PyTorch and ideally a GPU, and it changes whenever the data grows.
* **Project 2** *measures*. The inputs are vector PDFs, so each layer can be rendered into
  pixel-perfect, perfectly aligned images, where any differing pixel is a real change. No
  training is needed, results are exact and repeatable, and it runs in seconds on any CPU
  without PyTorch.

Separate packages keep their dependencies, data and release cycles independent. Project 2 can
be installed on a prepress workstation without the 2.5 GB deep-learning stack. They share one
repository, environment, tooling and documentation. They do not share a "common" library,
because the overlap is a few lines of image handling. Premature shared libraries couple things
that change for different reasons.

## Quick start (Windows 11)

Full instructions, with the reasoning behind each step: [docs/02_INSTALLATION.md](docs/02_INSTALLATION.md).

```powershell
# One-time prerequisites: Git, the Python install manager, then Python 3.14
winget install --id Git.Git -e
winget install --id Python.PythonInstallManager -e
py install 3.14

# Clone, then create .venv and install everything (auto-detects an NVIDIA GPU)
git clone https://github.com/cblevins-AG/MachineLearningTrial.git
cd MachineLearningTrial
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once: allow local scripts
.\scripts\setup_env.ps1

# Every new terminal
.\.venv\Scripts\Activate.ps1

# Check the environment and run the tests (about 15 s, offline)
python scripts\check_env.py
pytest

# Try Project 2 immediately on the bundled samples
layer-diff compare projects\layer_diff\data\samples\label_v1.pdf projects\layer_diff\data\samples\label_v2.pdf -l Dieline
```

Project 1 needs your data first: see [docs/01_GETTING_STARTED.md](docs/01_GETTING_STARTED.md), section 7.

## Documentation

| Doc | Read it when |
|-----|--------------|
| [01 Getting started](docs/01_GETTING_STARTED.md) | First. Python/ML for an experienced C#/.NET developer; how training works end to end; turning the reference PDF + examples into a labelled dataset |
| [02 Installation](docs/02_INSTALLATION.md) | Setting up a machine: what to install, from where, and why |
| [03 Learning resources](docs/03_LEARNING_RESOURCES.md) | Going deeper: tiered courses, books and papers |
| [04 Troubleshooting](docs/04_TROUBLESHOOTING.md) | Something breaks: GPU/drivers, dependency conflicts, over-fitting, data loading, PDFs |
| [05 Repository inventory](docs/05_REPO_INVENTORY.md) | Finding your way around: every file and folder explained |

## Assumptions made (tell me if any are wrong)

* Development on **Windows 11 (native, PowerShell)** with **Python 3.14**.
* **GPU unknown**: the setup script detects an NVIDIA GPU and picks the matching PyTorch build
  (CUDA 13.0, CUDA 12.6 or CPU). Defaults are CPU-friendly (ConvNeXt-Nano backbone), and Google
  Colab is documented for free GPU time.
* Project 1: **one technique label per photo**; the 100-page reference document is a **PDF**.
* Project 2: "layers" are PDF **optional content groups** (Acrobat/Illustrator layers),
  selected by name or keyword. Descriptions are generated **locally**; the Claude API is an
  opt-in extra.

## Repository layout (short version; full detail in [docs/05](docs/05_REPO_INVENTORY.md))

```
docs/                     the five guides
projects/
  ribbon_classifier/      Project 1: configs/ data/ models/ notebooks/ src/ tests/
  layer_diff/             Project 2: configs/ data/ notebooks/ src/ tests/
requirements/             pinned dependencies per project + full lock file
scripts/                  setup_env.ps1 (Windows setup), check_env.py (health check), check_inventory.py
requirements.txt          one-stop install (used by setup_env.ps1)
environment.yml           optional conda alternative
```
