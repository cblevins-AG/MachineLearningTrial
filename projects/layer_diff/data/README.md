# Layer diff data

| Path | What goes here | In git? |
|------|----------------|---------|
| `input/` | Your production PDFs to compare. | No (git-ignored) |
| `samples/` | Three small synthetic layered PDFs (`label_v1..v3.pdf`) for trying the tool and for tests. Regenerate with `layer-diff make-samples`. | Yes |
| `output/` | One folder per `layer-diff compare` run (`report.md`, `report.json`, `images/`). Default location when `--out` is not given. | No (git-ignored) |
