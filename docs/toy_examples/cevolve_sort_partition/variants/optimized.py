"""Interaction-winning variant for the cEvolve-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "optimized"
VARIANT_SUMMARY = "Combine the compatible threshold, partition, and iterative edits."
APPROACH_FIT = (
    "This is why evolutionary combination search fits: the strongest payoff only "
    "appears when the good changes arrive together."
)
WHAT_CHANGED = [
    "Raise INSERTION_THRESHOLD from 16 to 32.",
    'Switch PARTITION_SCHEME from "lomuto" to "hoare".',
    "Enable USE_ITERATIVE.",
]
OVERRIDES = {
    "insertion_threshold": 32,
    "partition_scheme": "hoare",
    "use_iterative": True,
}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
