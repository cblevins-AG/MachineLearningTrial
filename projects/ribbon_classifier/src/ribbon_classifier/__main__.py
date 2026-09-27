"""Allows ``python -m ribbon_classifier ...`` as an alternative to the ``ribbon`` command."""

from .cli import main

raise SystemExit(main())
