"""Allows ``python -m layer_diff ...`` as an alternative to the ``layer-diff`` command."""

from .cli import main

raise SystemExit(main())
