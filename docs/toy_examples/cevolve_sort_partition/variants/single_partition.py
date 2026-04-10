"""Single-change partition variant for the cEvolve-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "single_partition"
VARIANT_SUMMARY = "Only switch the partition scheme."
APPROACH_FIT = (
    "This local change also helps a bit, but still misses the real interaction win."
)
WHAT_CHANGED = ['Switch PARTITION_SCHEME from "lomuto" to "hoare".']
OVERRIDES = {"partition_scheme": "hoare"}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
