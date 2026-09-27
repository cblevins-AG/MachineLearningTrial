# Project 1: Ribbon fold/cut classifier

Given a photo, find the ribbon in it and say which fold or cut technique was used, with a
confidence score.

```
photo ──► 1. localise: zero-shot detector (OWLv2) ──► crop ──► 2. classify: fine-tuned ConvNeXt ──► "dovetail_cut" (0.93)
```

## Recommended approach and why

### Classification: transfer learning, not training from scratch

| Option | Labelled images needed | Compute | Realistic result with a modest dataset | Role here |
|--------|-----------------------|---------|----------------------------------------|-----------|
| Train a CNN **from scratch** | thousands per class | GPU, hours to days | Memorises the training set and generalises poorly. It has to learn what an edge is before it can learn what a fold is. | Not recommended |
| **Linear probe**: frozen DINOv2 features + logistic regression | ~20+ per class | CPU, minutes | A solid baseline; often surprisingly close to fine-tuning | **Step 1**: sanity-check the labels and get a number to beat |
| **Fine-tune** a pretrained ConvNeXt (transfer learning) | ~50–100+ per class | CPU: about an hour for 1,000 images; GPU: minutes | Usually the best accuracy for fine-grained visual differences | **Default model** |
| Ask a vision-language model (e.g. Claude) with a few examples | 0–5 per class | API cost per photo | Good at coarse categories; less dependable for subtle fold geometry; needs a network connection | Handy for *pre-labelling* images for a person to confirm; not the production classifier |

Why transfer learning works: a backbone pretrained on millions of images has already learned
general visual features (edges → textures → shapes → parts). We keep those, replace the final
layer with one sized for your classes, and train mostly that layer, nudging the rest only
slightly. A few hundred labelled photos can then be enough. Training starts with a few
"head-only" epochs, with the backbone frozen so a randomly initialised head cannot damage good
pretrained weights, and then fine-tunes everything at a 10× lower learning rate for the
backbone.

**Why ConvNeXt-Nano by default:** it is a modern convolutional network that is strong on fine
texture and shape and stable to fine-tune. At 15M parameters it trains at about **1.2 min per
epoch per 1,000 images on a 4-core CPU** (measured). Since it is not yet known whether a GPU is
available, the default is CPU-friendly. With an NVIDIA GPU, switch `model.backbone` in
`configs/default.yaml` to `convnext_tiny.fb_in22k_ft_in1k`: 28M parameters, usually slightly
more accurate, about 2.8 min/epoch/1,000 images on the same CPU but only seconds on a GPU.

**Why DINOv2 for the probe:** its features were learned without labels (self-supervised) and are
unusually good when kept frozen, which is exactly what a probe needs.

### Localisation: zero-shot detection first

| Option | Labels needed | Notes | Role |
|--------|---------------|-------|------|
| No localisation (classify the whole photo) | none | Fine only if the ribbon fills the frame. Otherwise the model can learn the background ("blue tablecloth = box pleat"). | Fallback when nothing is detected |
| **Zero-shot detector (OWLv2)**, prompted with "a photo of a ribbon" | none | Understands text prompts, so no training. 154M parameters (~0.6 GB download); roughly 2–5 s per photo on a 4-core CPU, well under a second on a GPU. | **Default** |
| Fine-tuned detector (e.g. RT-DETR in `transformers`, or torchvision Faster R-CNN) | ~200+ drawn boxes (Label Studio) | Best boxes on your specific photos | Upgrade if OWLv2 crops are poor |
| Segmentation (SAM 2) from the detected box | none | Pixel mask removes the background entirely; heavier | Upgrade if backgrounds still leak into predictions |

Licence note: the popular Ultralytics YOLO package is AGPL-3.0, which obliges you to publish
source code if you distribute it or offer it as a network service. Everything used here is
Apache-2.0, BSD or MIT.

### Assumptions (change these if they are wrong)

* **One technique per image** (single-label classification). If a single ribbon can show a fold
  *and* a cut, switch to two outputs (fold type, cut type). Ask, and this is a small change.
* The reference document is a **PDF**. For Word files, save as PDF first.
* Photos are JPEG, PNG or HEIC (iPhone). Rotation stored in EXIF is applied automatically.
* The **test set should look like real use**: reference-document images always go to training
  (`split.train_only_sources`), never to validation or test.
* Horizontal flip is used as augmentation, which assumes no technique has a mirror-image variant
  with a different name. If one does, set `augment.horizontal_flip: false`.

## Workflow (run from `projects\ribbon_classifier` with the venv active)

The full walkthrough with explanations is in [docs/01_GETTING_STARTED.md](../../docs/01_GETTING_STARTED.md), section 7.

```powershell
# 0. Put the reference PDF in data\raw\reference_pdf\ and labelled photos in data\raw\examples\<label>\...

# 1. Pull images + page text out of the reference PDF, then label the CSV in Excel
ribbon extract-reference data\raw\reference_pdf\reference.pdf

# 2. Combine labelled folders + the finished reference CSV into data\labels.csv
ribbon manifest --folders data\raw\examples --csv data\interim\reference_pdf\reference_candidates.csv

# 3. Validate (missing/corrupt files, duplicates, label typos) and split into train/val/test
ribbon split

# 4. Crop every image to the ribbon (review crop_found / crop_score in the CSV afterwards)
ribbon crop

# 5. Baseline in minutes (CPU is fine)
ribbon probe

# 6. Fine-tune (writes models\<timestamp>_<backbone>\)
ribbon train

# 7. Score on the untouched test split: accuracy, macro-F1, confusion matrix, list of errors
ribbon evaluate models\<run folder>

# 8. Use it
ribbon predict models\<run folder> C:\path\to\new_photo.jpg
```

Relative paths in `configs/default.yaml` are resolved against this folder, so `ribbon` commands
work from any directory. Pass `--config my.yaml` to use a variant (copy `default.yaml` into
`configs/` and edit), and `--device cpu|cuda|mps` to override device selection.

## What "good" looks like

* **Probe first.** If the probe's validation macro-F1 is low (say below 0.5 with 5+ classes),
  look at the data before training: `notebooks/01_explore_dataset.ipynb`.
* **Fine-tuning should beat the probe.** If it doesn't, you are probably over-fitting (too few
  images per class); see [docs/04_TROUBLESHOOTING.md](../../docs/04_TROUBLESHOOTING.md).
* **Read `errors_test.csv`.** The most confident mistakes are usually label errors or ambiguous
  photos. Fixing ten labels often beats any hyper-parameter change.
* Report **macro-F1** alongside accuracy: it does not let a common class hide a rare one that is
  always wrong.

## Layout

| Path | Contents |
|------|----------|
| `configs/default.yaml` | Every tunable setting, commented |
| `data/` | Raw → interim → processed data and the label CSVs (git-ignored; see `data/README.md`) |
| `models/` | One folder per training run (git-ignored; see `models/README.md`) |
| `notebooks/` | `01_explore_dataset.ipynb` (before training), `02_review_model.ipynb` (after) |
| `src/ribbon_classifier/` | The package: one module per pipeline step, plus `cli.py` |
| `tests/` | Offline tests on synthetic images (`pytest projects/ribbon_classifier`) |

## Deploying later (.NET)

A trained model can be exported to ONNX (`torch.onnx.export`) and run from C# with
`Microsoft.ML.OnnxRuntime`, e.g. behind an ASP.NET Core API that a Blazor app calls. Pre- and
post-processing (resize, normalise with the `mean`/`std` in `meta.json`, softmax, class names)
must then be reimplemented in C#. Worth doing once the model is good; not before.
