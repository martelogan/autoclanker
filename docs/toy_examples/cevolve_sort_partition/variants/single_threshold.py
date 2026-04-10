"""Single-change threshold variant for the cEvolve-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "single_threshold"
VARIANT_SUMMARY = "Only improve the insertion threshold."
APPROACH_FIT = (
    "A greedy local edit helps a bit, but it does not unlock the best combined "
    "behavior."
)
WHAT_CHANGED = ["Raise INSERTION_THRESHOLD from 16 to 32."]
OVERRIDES = {"insertion_threshold": 32}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
