# Project 2: PDF layer difference tool

Given 2 to N production PDFs and one or more layer names or keywords (e.g. `Dieline`, `varnish`,
`Cut*`), render that layer from every file, find what differs, describe each difference in
plain English, and give a **difference score** as a percentage.

```
PDFs ──► isolate layer "Dieline" (pikepdf) ──► render at 150 DPI (pdfium) ──► per-pixel CIELAB colour difference
     ──► group into regions ──► classify: added / removed / changed / moved ──► sentences + score + overlay images
```

Example output on the included samples (`layer-diff compare data\samples\label_v1.pdf data\samples\label_v2.pdf -l Dieline`):

```
'Dieline': label_v1.pdf vs label_v2.pdf - 4.5% of the layer's artwork differs (1 difference: 1 moved; SSIM 0.997).
  - #1 Moved: an element about 7.5 x 14.4 mm at the middle-left of the page (x 4.4 mm, y 65.5 mm from the top-left corner) shifted 7.1 mm up.
```

## Recommended approach and why

Your inputs are **vector PDFs**, not photos. That changes the problem completely. Rendering the
same layer from each file at the same resolution produces images that line up pixel for pixel:
there is no camera angle, lighting or alignment to correct. Any pixel that differs is a real
change. Classical image comparison is therefore exact, fast and explainable here, and machine
learning is not needed to *find* differences. A language model can still help to *phrase* them.

| Approach | Strengths | Weaknesses | Role |
|----------|-----------|------------|------|
| **Classical pixel comparison of rendered layers** (CIELAB colour difference, connected regions, SSIM) | Exact, deterministic (same input → same answer), fast (seconds), offline, no training data, every number can be traced to pixels | Literal: flags *any* visual change, including export noise. Tolerance and minimum-area settings handle that. No idea what an element *is*. | **Default**: finds, measures and scores everything |
| Deep-feature similarity (DINOv2/CLIP embeddings, LPIPS) | Robust to small shifts and noise; good at "how alike are these overall" for *photos* | Cannot say *where* or *what* changed precisely; the score is not "% changed"; slower | Not needed for aligned vector renders. Worth revisiting if inputs ever become photos or scans. |
| Vision-language model (Claude) | Rich, domain-aware wording ("the left notch is deeper") | Can miss or invent small differences, costs money per call, sends artwork off-site, not repeatable, so it can't be the source of a score | **Optional describer** (`--describer claude`), given the *measured* regions so it describes rather than detects |
| Vector-level diff (compare drawing operators / text objects) | Exact text changes ("25 mm" → "38 mm") | The same picture can be encoded many ways, and a re-export changes the operators without changing the look, so there are many false alarms | Possible future add-on for text layers |

### Why this is a separate tool from Project 1

* **Different kind of problem.** Project 1 learns from labelled examples (supervised ML).
  Project 2 measures; there is nothing to train.
* **Different dependencies.** Project 2 does not need PyTorch (about 2.5 GB with CUDA). It can be
  installed on a prepress workstation or a server with `pip install -e projects/layer_diff` and
  nothing else.
* **Different data and lifecycle.** Project 1 has labelled photos, retraining and model versions.
  Project 2 has input PDFs and reports.
* **Little shared code.** The overlap is a few lines of image handling. A shared "common"
  library for that would couple two projects that change for different reasons. If real
  duplication appears later, extract it then.

They live in one repository because they share the environment, tooling, docs and PDF know-how.

## How the numbers are defined

* **Difference score (per pair)** = changed pixels ÷ pixels that have ink in *either* file × 100.
  It measures *how much of the layer's artwork changed*, independent of page size: a dieline that
  moved entirely scores near 100% even if it covers 2% of the page. Pages of multi-page files are
  pooled (pixel-weighted).
* **Overall score (N files)** = mean of the pair scores. `pairwise` mode (default) compares every
  pair, N×(N−1)/2 comparisons. `--mode baseline` compares each file against the first, useful
  when one file is the approved master. The full N×N matrix is always in the report.
* **SSIM** (structural similarity, 1.0 = identical) is reported alongside as a second opinion.
  It is less sensitive to 1-pixel shifts and more sensitive to texture changes.
* A pixel is "changed" when its colour difference is above `color_tolerance` (CIE76 ΔE; default
  10, well above rendering noise and below any change a person would notice). Specks smaller
  than `min_region_mm2` are ignored, and changes closer than `merge_distance_mm` merge into one
  region.

## Usage

```powershell
# What layers, pages and spot colours do these files have? Any content outside layers?
layer-diff inspect data\input\*.pdf

# Compare one or more layers across 2..N files (layer names are case-insensitive keywords)
layer-diff compare data\input\job_v1.pdf data\input\job_v2.pdf data\input\job_v3.pdf --layer Dieline --layer "White Ink"

# Options
#   --mode baseline            each file vs the first instead of all pairs
#   --pages 1,3-4              only some pages
#   --dpi 300                  finer detail (4x slower)
#   --tolerance 5              stricter colour matching
#   --describer claude         richer wording via the Claude API (needs ANTHROPIC_API_KEY)
#   --config my_settings.yaml  all settings at once (copy configs\default.yaml)
#   --out <folder>             default: data\output\<timestamp>\
```

Layer matching: an exact name wins (`Cut` will not also pick up `Cut Contour`); otherwise any
layer *containing* the keyword; `Cut*` uses wildcards; `re:^die\s?line` is a regular expression.
The report lists which layer names matched in each file.

Output folder: `report.md` (human-readable, with overlay images), `report.json` (everything,
machine-readable), and `images/<layer>/`, containing each file's layer render plus a
colour-coded overlay per pair. In the overlays, red means only in the first file, blue means
only in the second, amber means in both but different, and a purple box marks a move. Boxes
are numbered to match the text.

Try it now on the synthetic samples:

```powershell
layer-diff compare data\samples\label_v1.pdf data\samples\label_v2.pdf data\samples\label_v3.pdf -l Dieline -l Artwork -l Varnish
```

or step through `notebooks/01_layer_diff_walkthrough.ipynb`.

## Things to know about production PDFs

* **Content outside every layer** is always drawn, so it would appear in every layer's render.
  `inspect` and `compare` warn about it. If identical across files it cancels out of the diff,
  but it still inflates the score's denominator.
* **"Layer" vs spot colour.** A dieline is sometimes a *spot colour* (e.g. `CutContour`) rather
  than, or as well as, a layer. `inspect` lists both. This tool isolates layers (optional content
  groups); isolating a spot-colour separation is a possible extension.
* **PDF/X-1a and PDF/X-3 files cannot contain layers** (the standards forbid optional content).
  PDF/X-4 can. If a file shows no layers, check how it was exported.
* Rendering uses the page's CropBox, as viewers do. Pages of different sizes are aligned at the
  top-left and reported.
* pdfium does not simulate **overprint**. A layer relying on overprint renders as a knockout,
  identically in every file, so comparisons stay valid.
* Missing fonts are substituted the same way for every file, so text still compares correctly.
* Password-protected files must be unlocked first.

## Layout

| Path | Contents |
|------|----------|
| `configs/default.yaml` | All settings with explanations (equal to the built-in defaults) |
| `data/` | `input/` your PDFs (git-ignored), `samples/` synthetic test PDFs, `output/` reports (git-ignored) |
| `notebooks/01_layer_diff_walkthrough.ipynb` | Step-by-step tour of the pipeline |
| `src/layer_diff/` | `pdf_layers.py` (find/isolate/render layers), `compare.py` (measure), `describe.py` (English), `report.py` (overlays, JSON/Markdown), `pipeline.py` (end-to-end), `cli.py`, `samples.py` |
| `tests/` | `pytest projects/layer_diff` (about 2 s, fully offline) |
