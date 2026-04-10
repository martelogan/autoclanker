"""Baseline variant for the cEvolve-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "baseline"
VARIANT_SUMMARY = "Keep the default sorting strategy."
APPROACH_FIT = (
    "This is the reference point before any interaction-aware search kicks in."
)
WHAT_CHANGED = ["No changes. This keeps the default quicksort strategy."]
OVERRIDES: dict[str, object] = {}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
