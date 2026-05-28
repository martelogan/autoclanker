from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("autoclanker")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

from autoclanker.clankergraph import (
    CLANKERGRAPH_SCHEMA_VERSION,
    belief_input_from_clankergraph,
    load_clankergraph_document,
    summarize_clankergraph_document,
    validate_clankergraph_document,
)

__all__ = [
    "CLANKERGRAPH_SCHEMA_VERSION",
    "__version__",
    "belief_input_from_clankergraph",
    "load_clankergraph_document",
    "summarize_clankergraph_document",
    "validate_clankergraph_document",
]
