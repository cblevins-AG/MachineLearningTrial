"""Check that docs/05_REPO_INVENTORY.md mentions every file and folder in the repo.

Run from anywhere:  python scripts/check_inventory.py
Lists every path that git tracks (or would track: new, non-ignored files), plus each
folder containing them, and reports any that the inventory does not mention as
`path` (files) or `path/` (folders). Exit code 1 if something is missing, so it can
run in CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "docs" / "05_REPO_INVENTORY.md"


def repo_files() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sorted({line for line in output.splitlines() if line and (REPO / line).exists()})


def main() -> int:
    text = INVENTORY.read_text(encoding="utf-8")
    files = repo_files()
    folders = sorted({str(parent.as_posix()) for f in files for parent in Path(f).parents if str(parent) != "."})
    missing = [f for f in files if f"`{f}`" not in text] + [d for d in folders if f"`{d}/`" not in text]
    if missing:
        print(f"{INVENTORY.name} does not mention {len(missing)} path(s):")
        for path in missing:
            print(f"  {path}")
        return 1
    print(f"Inventory complete: {len(files)} files and {len(folders)} folders documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
