# Trained models

One folder per training run, created by `ribbon train` (git-ignored, except this README).

| File | Contents |
|------|----------|
| `model.safetensors` | The trained weights. safetensors holds numbers only, so loading it cannot run code (unlike pickle-based `.pt` files). |
| `meta.json` | Everything needed to use the weights: backbone name, class names in output order, image size, normalisation, the full config, library versions and validation scores. |
| `history.csv` | Loss / accuracy / macro-F1 per epoch; plot it in `notebooks/02_review_model.ipynb`. |
| `eval_<split>.json`, `confusion_<split>.png`, `errors_<split>.csv` | Written by `ribbon evaluate <run folder>`. |

`probe_<timestamp>/probe_report.json` folders come from `ribbon probe` (baseline scores only; no weights).

To keep or share a model, copy its whole run folder: the weights are useless without `meta.json`.
