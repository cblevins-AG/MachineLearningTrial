# 04 · Troubleshooting

Problems are grouped by where they show up. Each entry gives the **symptom** (often the exact
error text), the **cause**, and the **fix**. Start with `python scripts\check_env.py`: it
diagnoses most environment problems in one go.

**Contents:** [1. Python and the environment](#1-python-and-the-environment) ·
[2. Installing packages](#2-installing-packages-and-dependency-conflicts) ·
[3. GPU and drivers](#3-gpu-drivers-and-pytorch-builds) ·
[4. Loading data](#4-loading-images-and-label-files) ·
[5. Training results](#5-training-results-over-fitting-and-friends) ·
[6. Cropping and predicting](#6-cropping-model-downloads-and-prediction) ·
[7. PDF layer comparison](#7-pdf-layer-comparison-project-2) ·
[8. Notebooks and git](#8-notebooks-editors-and-git)

---

## 1. Python and the environment

**`python` opens the Microsoft Store, or prints "Python was not found".**
*Cause:* Windows' "App execution alias" for `python.exe` points at the Store when no real
Python is on PATH. *Fix:* use `py` (the install manager) outside the venv. Inside an activated
venv, `python` always works. Optionally turn the aliases off in **Settings → Apps → Advanced
app settings → App execution aliases**, or run `py install --configure`.

**`py -3.14` reports that no matching Python runtime is installed.**
*Fix:* `py install 3.14`, then `py list` to confirm.

**`.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`**
*Cause:* PowerShell's default execution policy. *Fix (once):*
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. The per-user scope needs no admin rights.

**`ModuleNotFoundError: No module named 'torch'` (or `layer_diff`, `ribbon_classifier`) although you installed it.**
*Cause:* a different Python is running: the venv isn't activated, or VS Code or a notebook is
using another interpreter or kernel. *Check:* `python -c "import sys; print(sys.executable)"`
must print a path ending in `MachineLearningTrial\.venv\Scripts\python.exe`. *Fix:* activate
the venv; in VS Code run **Python: Select Interpreter** and choose `.venv`; in a notebook pick
the `.venv` kernel (top right) and restart the kernel.

**`ribbon` or `layer-diff` is "not recognized as the name of a cmdlet".**
*Cause:* the venv isn't active (the commands live in `.venv\Scripts`). *Fix:* activate it, or
run `python -m ribbon_classifier …` / `python -m layer_diff …`.

**The environment is in a weird state after experiments.**
*Fix:* it's disposable. `.\scripts\setup_env.ps1 -Recreate` rebuilds it from the pinned files
in a few minutes.

## 2. Installing packages and dependency conflicts

**`ERROR: No matching distribution found for torch==2.14.0` (or numpy, pikepdf, …).**
*Cause:* an unsupported Python: too new (3.15), too old (below 3.12 for NumPy 2.5), or 32-bit.
*Check:* `python -c "import sys, platform; print(sys.version, platform.architecture())"`.
*Fix:* rebuild the venv with 64-bit Python 3.14: `.\scripts\setup_env.ps1 -Recreate`.

**`error: Microsoft Visual C++ 14.0 or greater is required` / "Building wheel for … failed".**
*Cause:* pip found no pre-built wheel for your Python and is trying to compile from source.
Every pin in this repo has Windows wheels for Python 3.14, so this means a different Python
version, or a package added without checking. *Fix:* use Python 3.14. Don't install Visual
Studio Build Tools to work around it.

**`OSError: [Errno 2] No such file or directory` during install, with a very long path.**
*Cause:* Windows' 260-character path limit. *Fix:* enable long paths
([02_INSTALLATION.md](02_INSTALLATION.md), step 4) and/or clone to a short folder like `C:\src\`.

**`ERROR: ResolutionImpossible` / "Cannot install X and Y because these package versions have conflicting dependencies".**
*Cause:* a newly requested package needs versions that contradict the pins. *Fix:*
1. Don't fight it with `--force` or `--no-deps`.
2. Find a version of the new package compatible with the pins (pypi.org → "Release history",
   check its requirements), or bump the conflicting pin deliberately.
3. Update `requirements/*.txt`, regenerate `requirements/lock.txt` (command at its top),
   reinstall, and run `pytest`.
4. Tools with heavy, conflicting dependency trees (Label Studio) get their **own** venv.

**`pip check` reports broken requirements after a manual `pip install`.**
*Fix:* `python -m pip install -r requirements.txt` restores the pinned set; or `-Recreate`.

**The GPU build of PyTorch got replaced by the CPU build.**
*Symptom:* `python -c "import torch; print(torch.__version__)"` prints `2.14.0+cpu` (or no
`+cu…` suffix) and `check_env.py` says "CPU-only build". *Cause:* something ran
`pip install -U torch` or installed a package that pulled torch from PyPI, which only has CPU
builds for Windows. *Fix:*
`python -m pip install --force-reinstall --no-deps torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130`
(or `cu126`).

**SSL / certificate errors from pip or model downloads on a corporate network.**
*Cause:* a proxy that inspects TLS traffic. *Fix:* recent pip uses the Windows certificate
store, so it usually works once IT has deployed their root certificate. Otherwise ask IT for the
root certificate file and set `PIP_CERT` and `SSL_CERT_FILE` to its path. Set `HTTPS_PROXY` if
a proxy must be named explicitly. Never disable certificate verification.

## 3. GPU, drivers and PyTorch builds

**`torch.cuda.is_available()` is `False`** (check_env: "CUDA build installed but no usable GPU"). In order of likelihood:
1. **CPU build installed:** see "GPU build got replaced" above.
2. **Driver too old for the build:** `cu130` needs driver **580 or newer**. `nvidia-smi`'s
   top-right "CUDA Version" must be ≥ 13.0. *Fix:* update the driver, or use `-Torch cu126`.
3. **No NVIDIA driver / no NVIDIA GPU:** `nvidia-smi` isn't found or errors. On laptops with
   hybrid graphics, set Windows **Settings → Display → Graphics** to "High performance" for
   `python.exe`.
4. **Remote desktop / virtual machine** without GPU pass-through.

**`CUDA error: no kernel image is available for execution on the device`.**
*Cause:* the GPU is older than the build supports. CUDA 13 dropped Maxwell, Pascal and Volta
(GTX 9xx/10xx, Titan V/V100). *Fix:* `.\scripts\setup_env.ps1 -Torch cu126 -Recreate`.

**`nvidia-smi` shows "CUDA Version: 12.x". Do I need to install CUDA 13?**
No. That number is the newest CUDA version the *driver* supports; PyTorch brings its own CUDA
libraries. Update the *driver* if you want the `cu130` build. Never install the CUDA Toolkit
for this repo.

**`torch.OutOfMemoryError: CUDA out of memory`.**
*Fix, in order:* lower `train.batch_size` (32 → 16 → 8); keep `train.amp: true` (half-precision
activations use about half the memory); close other GPU-heavy apps (browsers, games); lower
`model.image_size` (224 → 192); use a smaller backbone (`convnext_nano…` or `efficientnet_b0…`).

**`OSError: [WinError 126] The specified module could not be found. Error loading "…\torch\lib\c10.dll"` (or `fbgemm.dll`).**
*Cause:* missing Microsoft Visual C++ Redistributable. *Fix:* install it
([02_INSTALLATION.md](02_INSTALLATION.md), step 3) and open a new terminal.

**Training on the CPU is too slow.**
Run `ribbon probe` first (minutes) to validate the data before any long run. Keep the
CPU-friendly default backbone (ConvNeXt-Nano: measured about 1.2 min/epoch per 1,000 images on 4
cores) or switch to `efficientnet_b0.ra4_e3600_r224_in1k` (about 0.9 min). Set
`train.num_workers: 4` so image decoding runs in parallel (see section 4). Lower `train.epochs`
for quick experiments. Or use Google Colab's free GPU
([02_INSTALLATION.md](02_INSTALLATION.md), step 5).

**Apple Silicon: `NotImplementedError: The operator … is not currently implemented for the MPS device`.**
Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported operations fall back to the CPU, or pass
`--device cpu`.

## 4. Loading images and label files

**The DataLoader hangs, or crashes with `RuntimeError: … An attempt has been made to start a new process before the current process has finished its bootstrapping phase`.**
*Cause:* on Windows, `num_workers > 0` starts worker *processes* by re-importing your script.
That needs the `if __name__ == "__main__":` guard, and doesn't work well inside notebooks.
*Fix:* keep `train.num_workers: 0` (the default) in notebooks. The `ribbon` commands are safe
with `num_workers: 4`.

**`PIL.UnidentifiedImageError: cannot identify image file …`.**
*Causes:* a corrupt or partially copied file; a non-image file in an image folder (`Thumbs.db`,
`.DS_Store`, a PDF); a format Pillow can't read. HEIC is supported through `pillow-heif`.
`ribbon split` opens every image and lists unreadable ones. *Fix:* delete or re-export those
files, or remove their rows.

**Photos appear rotated 90° in notebooks or crops.**
*Cause:* phones store orientation as EXIF metadata. This repo's `load_image` applies it
(`ImageOps.exif_transpose`). If you open images with plain `Image.open` in your own code,
apply it too.

**`DecompressionBombWarning` / `DecompressionBombError` on huge scans.**
Pillow warns above about 89 megapixels and refuses images above about 179 megapixels, as a safety measure against malicious files. Downscale the scans
(nothing here needs more than about 2000 px on the long side), or, for trusted files only, raise
`PIL.Image.MAX_IMAGE_PIXELS`.

**"Missing file" reported by `ribbon split` for files that exist.**
*Causes:* `image_path` is not relative to `projects/ribbon_classifier/data/` (it must look like
`raw/examples/dovetail_cut/IMG_0001.jpg`); a typo or case difference in folder names; the path
still contains a drive letter. Backslashes are converted to forward slashes automatically.

**Excel mangled the CSV (accents became `Ã©`, leading zeros vanished, columns merged).**
*Fix:* save with **File → Save As → CSV UTF-8 (Comma delimited)**. `reference_candidates.csv`
is written with a UTF-8 byte-order mark so Excel opens it correctly. Don't let Excel convert
paths or IDs into numbers or dates: format those columns as Text before editing.

**"Labels that look like the same class: ['Box Pleat', 'box_pleat']".**
Pick one spelling (the one in your label spec), fix the CSV or folder name, and re-run
`ribbon manifest` and `ribbon split`. The label string *is* the class; any difference makes a
new class.

**`ValueError: Only N group(s) available but need at least M …` from `ribbon split`.**
There are too few independent groups to carve out validation and test sets. Add images, or
temporarily lower `split.val_fraction` / `split.test_fraction`. With very little data, prefer
more data over clever splitting.

## 5. Training results: over-fitting and friends

**Over-fitting (the classic small-dataset problem).**
*Symptoms:* training accuracy approaches 100% while validation accuracy stalls or drops; a large
gap between `train_acc` and `val_acc` in `history.csv`; early stopping triggers after a few
epochs. *Fixes, cheapest first:*
1. **More or better data**: more images for the confused classes; fix label errors
   (`errors_*.csv`).
2. **Crop to the ribbon** (`ribbon crop`) so the model can't memorise backgrounds.
3. **Stronger augmentation**: `augment.min_crop_scale` 0.6 → 0.4, `rotation_degrees` 15 → 30,
   `color_jitter` 0.2 → 0.3, but only transformations that keep the label true.
4. **Freeze more / train less**: more `head_only_epochs`; lower `backbone_lr` (1e-4 → 3e-5); or
   rely on the linear probe if it is nearly as good.
5. **More regularisation**: `model.drop_rate` 0.2 → 0.4; `train.weight_decay` 0.05 → 0.1.
6. **Smaller backbone**: fewer parameters to memorise with.

**Validation or test accuracy is suspiciously high (about 100%) straight away.**
*Suspect leakage first:* near-duplicate photos (burst shots, crops of the same image, the same
ribbon photographed twice) split across train and test; filenames or backgrounds that reveal the
class; reference-document images that also appear among the examples. *Fix:* use groups
(sub-folders per physical ribbon); check the "groups found in more than one split" cell in
notebook 01 (must be 0); look at what differs between classes besides the technique.

**Great validation scores, but real-world photos fail.**
*Cause:* **domain shift**. Training images (reference-document figures, one photographer, one
table) differ from real use (other lighting, backgrounds, phones). *Fix:* make the test set look
like real use (that is why reference images are train-only); collect examples from the real
setting; crop to the ribbon; use more lighting/colour augmentation.

**Loss is `nan` or explodes.**
*Causes:* learning rate too high; rarely, half-precision overflow. *Fix:* lower `head_lr` and
`backbone_lr` by 3–10×; set `train.amp: false` to test; check that no image fails to load
(corrupt files can yield garbage).

**Loss barely moves; accuracy stays at chance (1 ÷ number of classes).**
*Causes:* labels shuffled relative to images (check a few rows by eye), a learning rate far too
low, or a data bug. *Debug the Karpathy way* (see [03_LEARNING_RESOURCES.md](03_LEARNING_RESOURCES.md)):
train on a tiny subset (about 20 images) and confirm the model can reach about 100% training
accuracy. If it can't memorise 20 images, something in the pipeline is broken.

**One class is never predicted (recall 0 in the report) / accuracy is fine but macro-F1 is poor.**
*Cause:* class imbalance, or a class that looks like another. Class weights are already
applied. *Fix:* more images of that class; check the confusion matrix for which class absorbs
it; decide whether the two should be merged.

**Scores change noticeably between identical runs.**
Expected with small validation sets: one or two images flipping changes the score by several
percent. Seeds are fixed (`train.seed`), but GPU arithmetic isn't bit-exact. Compare
configurations by the probe or by several seeds, not a single run.

## 6. Cropping, model downloads and prediction

**Model downloads fail (`OSError: We couldn't connect to 'https://huggingface.co'`, timeouts, HTTP 429).**
*Causes:* offline, a proxy or firewall (see the SSL entry in section 2), or anonymous rate
limits. *Fix:* retry; create a free Hugging Face token and set `HF_TOKEN`; once models are
cached, `HF_HUB_OFFLINE=1` forces cache-only use. The Hub warns about symlinks on Windows
(harmless; enable Developer Mode to silence it).

**`ribbon crop` finds no ribbon (`crop_found` False) in many photos, or crops the wrong object.**
*Fix:* sort `splits_cropped.csv` by `crop_score` and look at the worst crops. Try other
wording in `localizer.prompts` ("a satin ribbon", "a ribbon end", "a folded ribbon"); lower
`score_threshold` (0.10 → 0.05) if it misses, raise it if it grabs wrong objects. Increase
`margin` if crops cut off the ends (which is where cuts are!). If it's still poor, label about
200 boxes in Label Studio and train a detector ([projects/ribbon_classifier/README.md](../projects/ribbon_classifier/README.md)).
Images with no detection are kept whole, so training still works.

**Predictions look random on new photos although evaluation was fine.**
*Cause:* a train/inference mismatch. A model trained on crops (`trained_on_crops: true` in
`meta.json`) must see crops: don't pass `--no-localize`. Also check that the photos resemble
the training photos (section 5, domain shift).

**`RuntimeError: Error(s) in loading state_dict … size mismatch` when loading a run.**
*Cause:* `model.safetensors` and `meta.json` come from different runs, or were edited. Copy a
run folder whole.

## 7. PDF layer comparison (Project 2)

**"No layer matching 'X' exists in file.pdf".**
Run `layer-diff inspect file.pdf` to see the real layer names. Common causes:
* **Different naming across files** ("Dieline", "Die Line", "DIELINE_v2"): use a shorter
  keyword (`die`), a wildcard (`Die*`), or a regular expression (`re:^die\s?line`). The report
  lists which layers matched in each file.
* **Flattened export**: PDF/X-1a and PDF/X-3 can't contain layers; "flatten layers" on export
  removes them. Re-export as PDF/X-4 or plain PDF with "Create Acrobat Layers" (Illustrator)
  enabled.
* **It's a spot colour, not a layer**: `inspect` also lists spot colours (e.g. `CutContour`).
  This tool isolates layers; spot-colour isolation would be an extension.

**Every layer render also shows other artwork; `inspect` warns "content outside every layer".**
Content that isn't assigned to any layer is always drawn. If it is identical in all files it
cancels out of the differences, but it dilutes the score (it counts as ink). Fix it at the
source: move that artwork onto a layer in the design file.

**Tiny differences everywhere / high scores for files that "look the same".**
*Causes:* a re-export shifted everything by a fraction of a point; different colour conversion
(RGB vs CMYK, a different ICC profile) changed colours slightly; different fonts were
substituted. *Fix:* raise `--tolerance` (10 → 15) for colour noise; raise `min_region_mm2` for
specks; the `moved` detection already groups shifted elements. Look at the overlay to see which
kind of noise it is.

**A small but real change is missed.**
Lower `--tolerance` (e.g. 5) for subtle colour changes; raise `--dpi` to 300 for small text or
hairlines; lower `min_region_mm2` (0.5 → 0.1).

**Slow or out of memory on large pages.**
Memory grows with the square of DPI: an A3 page at 300 DPI is about 17 million pixels per
render, and several arrays are held per comparison. Use 150 DPI (the default) for overview runs,
`--pages` to limit pages, and 300 DPI only where needed.

**`error: a PDF is password-protected`.**
Remove the security in Acrobat (File → Properties → Security) and save a copy.

**`--describer claude` falls back to the local text, with "(Claude describer unavailable: …)".**
The reason is in the message: `AuthenticationError` means `ANTHROPIC_API_KEY` is missing or
invalid; `APIConnectionError` means network or proxy trouble; `RateLimitError` means wait and
retry. The comparison and scores never depend on the describer.

## 8. Notebooks, editors and git

**A notebook cell fails with `NameError` for something defined earlier.**
Cells share state and run in the order *you* run them. Use **Restart Kernel and Run All**
(VS Code: "Restart", then "Run All") to check the notebook works top to bottom.

**Code changes in `src/` don't show up in an open notebook.**
The module was imported before you edited it. Restart the kernel, or add
`%load_ext autoreload` and `%autoreload 2` in the first cell.

**`git push` rejected: "File … exceeds GitHub's file size limit of 100.00 MB".**
A data or model file slipped past `.gitignore` (e.g. saved outside the ignored folders).
*Fix:* `git rm --cached <file>`, move it into an ignored folder, commit, and push. If it is in
earlier commits, it must be removed from history. Ask before rewriting shared history. Use
Git LFS or DVC for large files ([01_GETTING_STARTED.md](01_GETTING_STARTED.md), section 8).

**Every notebook shows as changed after merely running it.**
`nbstripout` isn't installed as a git filter in this clone: run
`.venv\Scripts\nbstripout --install` (the setup script normally does this).

**Line-ending warnings (`LF will be replaced by CRLF`).**
Harmless: `.gitattributes` normalises line endings. PowerShell scripts are kept as CRLF on
purpose.
