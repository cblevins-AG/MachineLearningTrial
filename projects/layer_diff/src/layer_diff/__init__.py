"""layer_diff: compare named layers across production PDFs.

Typical use is the ``layer-diff`` command (see ``layer_diff.cli``). From Python::

    from layer_diff.config import Settings
    from layer_diff.pipeline import run

    report = run(["v1.pdf", "v2.pdf"], ["Dieline"], Settings(), "data/output/run1")
    print(report["layers"][0]["overall_score_pct"])
"""

__version__ = "0.1.0"
