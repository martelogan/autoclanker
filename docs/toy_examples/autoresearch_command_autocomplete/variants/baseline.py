"""Baseline variant for the autoresearch-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "baseline"
VARIANT_SUMMARY = "Use the app defaults as the starting point."
APPROACH_FIT = (
    "This is the reference point for a mostly additive search space where "
    "simple hill-climbing is enough."
)
WHAT_CHANGED = ["No changes. This keeps the default autocomplete profile."]
OVERRIDES: dict[str, object] = {}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
