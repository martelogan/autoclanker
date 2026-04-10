from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autoclanker")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"


__all__ = ["__version__"]
