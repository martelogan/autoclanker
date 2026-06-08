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
from autoclanker.issue_seeder import (
    ISSUE_SEED_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA_VERSION,
    IssueSeedArtifact,
    IssueSeedBundle,
    IssueSeedInput,
    build_issue_seed_bundle,
    load_issue_seed_input,
)

__all__ = [
    "CLANKERGRAPH_SCHEMA_VERSION",
    "ISSUE_SEED_SCHEMA_VERSION",
    "RUN_CONTRACT_SCHEMA_VERSION",
    "IssueSeedArtifact",
    "IssueSeedBundle",
    "IssueSeedInput",
    "__version__",
    "belief_input_from_clankergraph",
    "build_issue_seed_bundle",
    "load_clankergraph_document",
    "load_issue_seed_input",
    "summarize_clankergraph_document",
    "validate_clankergraph_document",
]
