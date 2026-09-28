# Ribbon classifier data

Everything in this folder except the README files is **git-ignored**: images are large,
may be proprietary, and can be re-created. Share them via a file share, Git LFS or DVC
(see docs/01_GETTING_STARTED.md, "Storing data and models").

| Path | What goes here | Created by |
|------|----------------|------------|
| `raw/reference_pdf/` | The 100-page reference document (PDF). Never edited. | you |
| `raw/examples/` | The labelled example photos, one sub-folder per label; optional nested sub-folders group photos of the same physical ribbon (`raw/examples/dovetail_cut/roll_17/IMG_0001.jpg`). Never edited. | you |
| `reference/` | `moa_catalog.yaml` + `moa_designs.yaml`: a reference deck of assembly steps (MOAs) typed up per example design. See `reference/README.md`. | you |
| `interim/moa/` | `moa_labels.csv` (one row per image: `label` + a count per operation), `moa_designs.csv`, `moa_operation_summary.csv`. | `ribbon moa-labels` |
| `interim/reference_pdf/` | Images extracted from the reference PDF + `reference_candidates.csv` to label in Excel. | `ribbon extract-reference` |
| `processed/crops/` | The same images cropped to the ribbon. | `ribbon crop` |
| `labels.csv` | Every labelled image: `image_path,label,group,source`. | `ribbon manifest` |
| `splits.csv` | `labels.csv` + a `split` column (train/val/test). | `ribbon split` |
| `splits_cropped.csv` | `splits.csv` pointing at the crops, plus `crop_found`/`crop_score`. | `ribbon crop` |

`image_path` values are relative to this folder and use forward slashes, so the CSVs work on
Windows, macOS and Linux alike. Treat `raw/` as read-only: every later file can be regenerated
from it with the commands above.
