"""Module entry point for ``python -m vehicle_inspection``."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
