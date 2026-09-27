"""Assign each image to train / val / test without leaking groups across splits.

* **train** - what the model learns from.
* **val**   - checked after every epoch to pick the best checkpoint and to stop
              early. Because decisions are made on it, it slowly stops being
              an unbiased judge.
* **test**  - looked at once, at the end, for the number you report.

Splitting is *stratified* (every class keeps roughly the same share in each
split) and *grouped* (all images in a group land in the same split).
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def _take_fold(df: pd.DataFrame, fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split off ~fraction of *df* (by group, stratified by label). Returns (rest, taken)."""
    n_splits = max(2, round(1 / fraction))
    if df["group"].nunique() < n_splits:
        raise ValueError(
            f"Only {df['group'].nunique()} group(s) available but need at least {n_splits} to split off "
            f"{fraction:.0%}. Add more images or lower val_fraction/test_fraction in the config."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():  # sklearn warns when a class has fewer members than n_splits
        warnings.simplefilter("ignore", UserWarning)
        rest_idx, taken_idx = next(splitter.split(df, df["label"], df["group"]))
    return df.iloc[rest_idx], df.iloc[taken_idx]


def assign_splits(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    train_only_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of *df* with a ``split`` column (train/val/test)."""
    train_only_sources = train_only_sources or []
    # Whole groups touching a train-only source go to train, so a duplicate of a
    # reference image can never end up in val/test.
    forced_groups = set(df.loc[df["source"].isin(train_only_sources), "group"])
    forced = df[df["group"].isin(forced_groups)]
    pool = df[~df["group"].isin(forced_groups)]

    rest, test = _take_fold(pool, test_fraction, seed)
    train, val = _take_fold(rest, val_fraction / (1 - test_fraction), seed + 1)

    out = pd.concat(
        [forced.assign(split="train"), train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    )
    return out.sort_values(["split", "label", "image_path"]).reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Images per class per split - print it and look for empty cells."""
    table = df.pivot_table(index="label", columns="split", values="image_path", aggfunc="count", fill_value=0)
    return table.reindex(columns=[c for c in ("train", "val", "test") if c in table.columns])
