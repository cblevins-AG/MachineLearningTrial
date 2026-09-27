"""Shared pytest fixtures (pytest discovers this file automatically)."""

from pathlib import Path

import pytest

from layer_diff.samples import make_samples


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    """label_v1/v2/v3.pdf generated once per test session into a temp folder."""
    return make_samples(tmp_path_factory.mktemp("samples"))
