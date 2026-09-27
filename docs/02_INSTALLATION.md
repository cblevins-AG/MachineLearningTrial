# 02 · Installation (Windows 11)

Everything to download and install, where to get it (official sources only), and why each
piece is needed. Plan for 30–45 minutes, mostly downloads: about 4 GB with an NVIDIA GPU, about
1.5 GB without, plus about 0.8 GB of pretrained models on first use.

> **Verified 27 September 2026.** Versions and wheel availability were checked against PyPI's
> own metadata. Every pinned package resolves to pre-built Windows wheels for Python 3.14, so
> nothing needs a C/C++ compiler. Download pages and commands were checked against official
> release notes and search listings for python.org, pytorch.org, nvidia.com, git-scm.com,
> code.visualstudio.com and learn.microsoft.com. Pages change: if a screen looks different from
> what is described here, trust the official page, and the *why* below still applies.

## Summary

| # | What | Required? | Why | Official source |
|---|------|-----------|-----|-----------------|
| 1 | Git for Windows | Yes | Clone this repo, commit, push | [git-scm.com/install/windows](https://git-scm.com/install/windows) |
| 2 | Python install manager + Python 3.14 | Yes | The language runtime; the `py` command manages versions | [python.org/downloads](https://www.python.org/downloads/) |
| 3 | Microsoft Visual C++ Redistributable (x64) | Yes (often already present) | PyTorch's DLLs are built with MSVC and need its runtime | [learn.microsoft.com](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) |
| 4 | Long-path support | Recommended | Some Python packages have deeply nested files beyond Windows' 260-character limit | Windows setting (below) |
| 5 | NVIDIA driver | Only with an NVIDIA GPU | Lets PyTorch use the GPU. **Do not install the CUDA Toolkit.** | [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/) or [NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/) |
| 6 | Visual Studio Code + extensions | Recommended | Best Python + notebook editor; the repo ships settings for it | [code.visualstudio.com](https://code.visualstudio.com/) |
| 7 | This repo's environment (`setup_env.ps1`) | Yes | Creates `.venv` with every package pinned, plus the right PyTorch build | this repo |
| — | Anthropic API key | Optional | Only for `layer-diff --describer claude` | [platform.claude.com](https://platform.claude.com/settings/keys) |
| — | Google Colab | Optional | Free cloud GPU if you have none | [colab.research.google.com](https://colab.research.google.com/) |
| — | Label Studio | Optional | Drawing bounding boxes if zero-shot cropping isn't good enough | [labelstud.io](https://labelstud.io/) |

`winget` (Windows Package Manager) ships with Windows 11. Where a `winget` command is given, it
installs the same official package as the website.

## 1. Git for Windows

**Why:** version control for code, and required to clone this repository. It also provides the
git filter that strips notebook outputs.

```powershell
winget install --id Git.Git -e --source winget
```

Or download the 64-bit installer from [git-scm.com/install/windows](https://git-scm.com/install/windows);
the defaults are fine. Then, once:

```powershell
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
git config --global core.longpaths true      # see step 4
```

Line endings are handled by the repo's `.gitattributes`, so leave `core.autocrlf` at the
installer default.

## 2. Python 3.14 via the Python install manager

**Why Python 3.14:** it is the newest version that *every* package here ships pre-built Windows
wheels for (PyTorch 2.14, NumPy 2.5, pikepdf 10, …), and it receives bug fixes until October
2027. 3.13 moves to security-only fixes around October 2026, after which python.org stops
publishing Windows installers for it. 3.15 is not supported by PyTorch yet.

**Why the install manager:** since Python 3.14, python.org's recommended Windows installer is
the *Python install manager* (`py`), which installs, updates and switches between Python
versions, much as the .NET SDK does for SDK versions. The traditional per-version installer is
deprecated and will not be produced from Python 3.16 onward.

```powershell
winget install --id Python.PythonInstallManager -e
```

Alternatives: the "Python install manager" download on [python.org/downloads](https://www.python.org/downloads/),
or the Microsoft Store. Then open a **new** terminal:

```powershell
py install 3.14        # download and install the latest 3.14.x (3.14.7 as of this writing)
py list                # show installed runtimes
py -3.14 --version     # run a specific version: Python 3.14.7
```

If you already have Pythons installed with the old installer, they keep working side by side.
The setup script always asks for 3.14 explicitly (`py -3.14 -m venv .venv`).

Background reading: [Using Python on Windows](https://docs.python.org/3/using/windows.html) (official docs).

## 3. Microsoft Visual C++ Redistributable

**Why:** PyTorch's Windows binaries are compiled with Microsoft's C++ compiler and load its
runtime DLLs. Most machines have it already (Visual Studio and many apps install it). If it is
missing, `import torch` fails with `OSError: [WinError 126] … c10.dll` or `fbgemm.dll`.
Installing the latest version is harmless:

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 -e
```

Or download [aka.ms/vc14/vc_redist.x64.exe](https://aka.ms/vc14/vc_redist.x64.exe), Microsoft's
permanent link to the latest supported version, listed on
[Latest supported Visual C++ Redistributable downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

## 4. Enable long paths (recommended)

**Why:** Windows limits paths to 260 characters unless long paths are enabled. Some packages
(notably Jupyter's) contain deeply nested files; combined with a repo under a long folder
path, installs can fail with a confusing "No such file or directory". Run once in an
**administrator** PowerShell, then sign out and in:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

(Microsoft's explanation: [Maximum path length limitation](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation).)
Also consider cloning to a short path such as `C:\src\MachineLearningTrial`.

## 5. GPU: find out what you have, then the driver

You weren't sure whether a GPU is available, so here is how to check. The setup script also
detects it automatically.

1. **Task Manager → Performance.** Any "GPU" entries show the model name.
2. In PowerShell: `nvidia-smi`. If this prints a table, you have an NVIDIA GPU with a working
   driver. The table's "CUDA Version" is the **highest CUDA version the driver supports**, not
   something installed.

Which PyTorch build to use (the setup script applies this table):

| Your hardware | PyTorch build | Why |
|---------------|---------------|-----|
| NVIDIA RTX 20xx / 30xx / 40xx / 50xx, GTX 16xx, or a recent workstation card (compute capability ≥ 7.5) **and** driver 580 or newer | `cu130` (CUDA 13.0) | Current default CUDA for PyTorch 2.14 |
| NVIDIA GTX 10xx / 9xx (Pascal/Maxwell), Titan V/V100 (Volta), **or** a driver older than 580 | `cu126` (CUDA 12.6) | CUDA 13 dropped these older architectures; the 580 driver branch is the last to support them |
| No NVIDIA GPU (Intel/AMD integrated or discrete) | `cpu` | Works everywhere. Project 2 never needs a GPU; for Project 1 the default settings are CPU-friendly |

**Driver (NVIDIA only):** install the latest Game Ready or Studio driver through the
[NVIDIA App](https://www.nvidia.com/en-us/software/nvidia-app/) or from
[nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/). For workstation cards, use the
RTX/Quadro driver from the same page.

**Do not install the CUDA Toolkit.** PyTorch's wheels bundle the CUDA libraries they need;
only the *driver* is required. A separately installed toolkit is unnecessary and a common
source of confusion.

**No GPU?** CPU training works for this dataset size (measured: about 1.2 minutes per epoch per
1,000 images with the default ConvNeXt-Nano on a 4-core CPU). For bigger experiments, use
[Google Colab](https://colab.research.google.com/) (free, time-limited GPU sessions): upload the
repo and your data to Google Drive, `pip install -r requirements.txt`, and run the same
`ribbon` commands. PyTorch on Linux already includes CUDA, so no extra step is needed there.

## 6. Visual Studio Code and extensions

**Why VS Code rather than Visual Studio:** Visual Studio has a Python workload, but the Python
ecosystem's tooling (Pylance, Jupyter notebooks, Ruff) is built for and best in VS Code. The
repo includes `.vscode/settings.json` (uses `.venv`, runs pytest, formats on save) and
`.vscode/extensions.json` (VS Code offers to install the recommended extensions when you open
the folder).

```powershell
winget install --id Microsoft.VisualStudioCode -e
```

Or the "User Installer" from [code.visualstudio.com/download](https://code.visualstudio.com/download).
Recommended extensions (all from the official Marketplace):

| Extension | ID | Why |
|-----------|----|-----|
| Python | `ms-python.python` | Run/debug Python, pick the interpreter, test explorer |
| Pylance | `ms-python.vscode-pylance` | IntelliSense and type checking (Roslyn-like) |
| Jupyter | `ms-toolsai.jupyter` | Open and run `.ipynb` notebooks in the editor |
| Ruff | `charliermarsh.ruff` | Lint and format on save |
| YAML | `redhat.vscode-yaml` | Editing the config files |

## 7. Clone the repo and create the environment

```powershell
cd C:\src                                   # any short path
git clone https://github.com/cblevins-AG/MachineLearningTrial.git
cd MachineLearningTrial
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned    # once per user: allow local .ps1 scripts
.\scripts\setup_env.ps1                     # or: -Torch cpu | cu130 | cu126, and -Recreate
```

**Why the execution policy change:** Windows blocks PowerShell scripts by default.
`RemoteSigned` for your user allows scripts you have on disk (including `.venv`'s `Activate.ps1`)
while still requiring downloaded scripts to be signed.

What `setup_env.ps1` does, and the manual equivalent if you prefer to type it yourself:

| Step | Manual command | Why |
|------|----------------|-----|
| Create `.venv` with Python 3.14 | `py -3.14 -m venv .venv` | A private, disposable environment for this repo |
| Upgrade pip | `.venv\Scripts\python -m pip install --upgrade pip` | Newer pip resolves dependencies better |
| Install PyTorch for your hardware | `.venv\Scripts\python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130` (or `cu126`, `cpu`) | **On Windows, PyPI only has the CPU build of PyTorch.** GPU builds come from PyTorch's own index |
| Install everything else | `.venv\Scripts\python -m pip install -r requirements.txt` | Both projects (editable) + pinned dependencies + dev tools. Must run from the repo root |
| Notebook git filter | `.venv\Scripts\nbstripout --install` | Strips notebook outputs on commit: small diffs, no accidental data in git |
| Health check | `.venv\Scripts\python scripts\check_env.py` | Confirms imports, the PyTorch build, GPU use and PDF rendering |

The PyTorch command is the one [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)
generates for Windows + Pip + your CUDA version; it's worth opening that page once to confirm.

**Conda users:** `environment.yml` is provided as an alternative (`conda env create -f environment.yml`).
PyTorch stopped publishing conda packages after version 2.5, so conda only provides Python
there, and pip installs everything else. For a GPU build, run the `--force-reinstall` command in
that file's header afterwards.

**uv users:** [uv](https://docs.astral.sh/uv/) is a much faster, drop-in pip replacement and is
installed as a dev tool (it regenerates `requirements/lock.txt`). `uv pip install -r requirements.txt`
works in the activated venv. pip remains the default because every tutorial you'll read uses it.

## 8. Verify

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\check_env.py
pytest
```

`check_env.py` should end with `Result: ready`. With an NVIDIA GPU it should report
`GPU ready: <name>`; with the CPU build it notes that training runs on the CPU. `pytest` should
report all tests passed. In VS Code: **Ctrl+Shift+P → Python: Select Interpreter →
`.venv\Scripts\python.exe`** (usually picked automatically).

## 9. First-use downloads (automatic)

Pretrained weights download from the [Hugging Face Hub](https://huggingface.co/) the first time
they're used, and are cached in `%USERPROFILE%\.cache\huggingface` (set the `HF_HOME` environment
variable to move the cache, e.g. to a bigger drive):

| Model | Used by | Size (approx.) |
|-------|---------|----------------|
| [`timm/convnext_nano.in12k_ft_in1k`](https://huggingface.co/timm/convnext_nano.in12k_ft_in1k) | `ribbon train` (default backbone) | 60 MB |
| [`timm/convnext_tiny.fb_in22k_ft_in1k`](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k) | `ribbon train` (GPU option) | 110 MB |
| [`timm/vit_small_patch14_dinov2.lvd142m`](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | `ribbon probe` | 90 MB |
| [`google/owlv2-base-patch16-ensemble`](https://huggingface.co/google/owlv2-base-patch16-ensemble) | `ribbon crop`, `ribbon predict` | 620 MB |

On Windows the Hub may warn that it cannot create symlinks in its cache. That is harmless (it
copies files instead). Enabling **Developer Mode** in Windows settings removes the warning and
saves disk space.

## 10. Optional extras

**Anthropic API key (for `layer-diff --describer claude`).** Create a key in the
[Claude Console](https://platform.claude.com/settings/keys)
([guide](https://platform.claude.com/docs/en/get-api-key)). The key is shown once, so store it in
a password manager. Then make it available to Python without putting it in any file in the repo:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."                            # this terminal only
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")   # persistent; open a new terminal
```

Using it sends layer images to Anthropic's API and is billed per request. The default local
describer sends nothing anywhere.

**Label Studio** ([labelstud.io](https://labelstud.io/)): if zero-shot cropping isn't accurate
enough, draw bounding boxes here to train a detector. Install it in its **own** venv
(`py -3.14 -m venv C:\tools\labelstudio-venv`, then `pip install label-studio` inside it),
because its pinned dependencies would conflict with this repo's.

## Uninstall / reset

* Environment only: delete `.venv\` and re-run `.\scripts\setup_env.ps1`.
* Cached models: delete `%USERPROFILE%\.cache\huggingface`.
* Python: `py uninstall 3.14`; then uninstall "Python install manager" from Windows Settings → Apps.
