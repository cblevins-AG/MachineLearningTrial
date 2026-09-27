"""Print a health report for the Python environment. Run: python scripts/check_env.py

Checks the interpreter, every pinned package, which PyTorch build is installed
and whether it can use a GPU (with a tiny calculation on it), and that PDF
layer rendering works. Exit code 1 if something required is missing.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import platform
import sys
import tempfile
from pathlib import Path

REQUIRED = [
    # (import name, distribution name)
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("PIL", "pillow"),
    ("yaml", "pyyaml"),
    ("matplotlib", "matplotlib"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("timm", "timm"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("sklearn", "scikit-learn"),
    ("pillow_heif", "pillow-heif"),
    ("pypdfium2", "pypdfium2"),
    ("pikepdf", "pikepdf"),
    ("cv2", "opencv-python-headless"),
    ("skimage", "scikit-image"),
    ("ribbon_classifier", "ribbon-classifier"),
    ("layer_diff", "layer-diff"),
]
OPTIONAL = [("anthropic", "anthropic"), ("pytest", "pytest"), ("ruff", "ruff"), ("jupyterlab", "jupyterlab")]


def ok(text: str) -> None:
    print(f"  [ok]   {text}")


def warn(text: str) -> None:
    print(f"  [warn] {text}")


def fail(text: str) -> None:
    print(f"  [FAIL] {text}")


def check_packages() -> bool:
    print("\nPackages")
    all_ok = True
    for module, dist in REQUIRED + OPTIONAL:
        try:
            importlib.import_module(module)
            ok(f"{dist} {metadata.version(dist)}")
        except Exception as error:  # ImportError, or a DLL load failure on Windows
            if (module, dist) in OPTIONAL:
                warn(f"{dist} not importable ({type(error).__name__}) - optional")
            else:
                fail(f"{dist}: {type(error).__name__}: {error}")
                all_ok = False
    return all_ok


def check_torch() -> None:
    print("\nPyTorch")
    import torch

    build = torch.version.cuda
    ok(f"torch {torch.__version__} ({'CUDA ' + build + ' build' if build else 'CPU-only build'})")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 2**30
        x = torch.randn(512, 512, device="cuda")
        (x @ x).sum().item()  # forces a real kernel launch
        ok(f"GPU ready: {name}, compute capability {major}.{minor}, {memory:.1f} GB")
    elif build:
        warn("CUDA build installed but no usable GPU: check the NVIDIA driver (docs/04_TROUBLESHOOTING.md)")
    elif torch.backends.mps.is_available():
        ok("Apple GPU (MPS) available")
    else:
        warn("No GPU in use: training runs on the CPU (fine for the probe and small runs; see docs/02_INSTALLATION.md)")


def check_pdf_layers() -> None:
    print("\nPDF layer rendering")
    from layer_diff.pdf_layers import render_layer
    from layer_diff.samples import make_samples

    with tempfile.TemporaryDirectory() as folder:
        v1 = make_samples(folder)[0]
        dieline = render_layer(v1, "Dieline", dpi=50).pages[0]
        artwork = render_layer(v1, "Artwork", dpi=50).pages[0]
    # The dieline layer is magenta line work; the red header band must not appear in it.
    red_in_dieline = ((dieline[..., 0] > 180) & (dieline[..., 1] < 60) & (dieline[..., 2] < 80)).sum()
    red_in_artwork = ((artwork[..., 0] > 180) & (artwork[..., 1] < 60) & (artwork[..., 2] < 80)).sum()
    if red_in_artwork > 100 and red_in_dieline == 0:
        ok("single-layer rendering works (pikepdf + pdfium)")
    else:
        fail("layer isolation did not behave as expected - please report this")


def main() -> int:
    print("Environment check")
    print(f"  Python {platform.python_version()} ({platform.architecture()[0]}) at {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    (ok if in_venv else warn)("running inside a virtual environment" if in_venv else "NOT inside a virtual environment")
    if sys.version_info < (3, 12):  # noqa: UP036 - this script may be run by the wrong Python on purpose
        fail("Python 3.12+ is required (3.14 recommended)")
        return 1

    packages_ok = check_packages()
    if packages_ok:
        check_torch()
        check_pdf_layers()

    print("\nOther")
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ok(f"pretrained models download to {hf_home} (set HF_HOME to move it)")
    key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    (ok if key_set else warn)(
        "ANTHROPIC_API_KEY is set (layer-diff --describer claude available)"
        if key_set
        else "ANTHROPIC_API_KEY not set - only needed for the optional Claude describer"
    )
    print("\nResult:", "ready" if packages_ok else "problems found - see docs/04_TROUBLESHOOTING.md")
    return 0 if packages_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
