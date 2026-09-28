# Reference decks that list assembly steps (MOAs)

Some reference documents don't name one technique per picture. Instead, each example design
comes with an ordered list of **methods of assembly** (MOAs), such as "cut into 2 pieces →
insert through 1x → tie 1x → wrap around element 2x". For such a deck, type the information
up once in two YAML files here and let `ribbon moa-labels` label every extracted image from
them. These files are **git-ignored** like everything else in `data/`: the deck may be
confidential.

## `moa_catalog.yaml`: the vocabulary

```yaml
operations:
  - id: insert_thru            # snake_case label, becomes a column in moa_labels.csv
    family: insert             # grouping, for reports
    counted: true              # the phrase carries a number ...
    max_count: 7               # ... and the deck's top bucket means "7 or more"
    patterns: ['^insert thru (\d+)\+? ?x']   # case-insensitive; group 1 = the count
    definition: ...
    pages: [2, 26]
legacy_methods:                # optional: an older one-phrase attachment method per design
  - id: weave_thru_slits
    patterns: ['^weave thru slits']
```

Phrases are lower-cased, curly quotes straightened, spaces collapsed and a trailing full stop
dropped before matching, so typos and punctuation in the deck don't need their own pattern.

## `moa_designs.yaml`: the labelled examples

```yaml
source:
  images_dir: interim/moa_pdf          # where `ribbon extract-reference --out` wrote the images
excluded_images:                       # extracted images that are not training data, and why
  page_002_img_01.png: logo            # files identical to this one are excluded too
designs:
  - page: 8
    section: Wrap around
    material: "1234567"                # the product/material number shown on the page
    ribbons:                           # one entry per separately assembled ribbon
      - legacy: "Weave thru slits"
        moas: ["Cut into 2 Pieces", "Insert Thru 1x", "Tie 1x Knot"]
    images: [page_008_img_02.png, {file: page_005_img_03.png, note: "definitions page", same_as: page_008_img_02.png}]
    notes: ...
    questions: [...]                   # open points to confirm with the deck's owners
illustrations: [...]                   # same shape; single-operation drawings, no legacy method
```

Copy MOA phrases **exactly as printed**; normalisation happens in code, and an unknown phrase is
reported rather than guessed.

## Output (`data/interim/moa/`)

| File | Contents |
|------|----------|
| `moa_labels.csv` | One row per image: `image_path,label,group,source`, then design metadata, then **one count column per operation** (summed over the design's ribbons, 0 when absent). Feed it to `ribbon manifest --csv`. |
| `moa_designs.csv` | One row per design: the per-ribbon step sequence, the counts, notes and open questions. |
| `moa_operation_summary.csv` | Per operation: how many designs show it, and which counts have (and have no) examples. |

`label` is the design's single legacy method by default (`--label-by legacy`). It is left empty,
and so skipped by `ribbon manifest`, when a design's ribbons have different legacy methods. Use
`--label-by section` for the deck's section headings. The count columns are multi-label targets
for a model with one output per operation; the current classifier trains on `label` only.
