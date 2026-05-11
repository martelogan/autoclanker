from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    _installed_version = version("autoclanker")
except PackageNotFoundError:  # pragma: no cover
    _installed_version = "0.1.0"

BIGBETS_VERSION = _installed_version

BIGBETS_GENERATOR_NAME = "bigbets"
BIGBETS_REGISTRY_SCHEMA_VERSION = "bigbets.registry.v1"
BIGBETS_ARTIFACT_SCHEMA_VERSION = "bigbets.artifacts.v2"
BIGBETS_SITE_SCHEMA_VERSION = "bigbets.site.v2"


def generator_metadata() -> dict[str, str]:
    return {
        "name": BIGBETS_GENERATOR_NAME,
        "version": BIGBETS_VERSION,
    }


__all__ = [
    "BIGBETS_ARTIFACT_SCHEMA_VERSION",
    "BIGBETS_GENERATOR_NAME",
    "BIGBETS_REGISTRY_SCHEMA_VERSION",
    "BIGBETS_SITE_SCHEMA_VERSION",
    "BIGBETS_VERSION",
    "generator_metadata",
]
