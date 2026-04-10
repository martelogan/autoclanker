"""OOM-prone branch for the Bayes-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "risky_oom"
VARIANT_SUMMARY = "Push width and batch together into the risky branch."
APPROACH_FIT = (
    "This is the feasibility cautionary example that the Bayes demo is meant "
    "to penalize."
)
WHAT_CHANGED = [
    'Increase HIDDEN_WIDTH from "width_default" to "width_plus_2".',
    'Increase BATCH_SIZE from "batch_default" to "batch_x2".',
]
OVERRIDES = {"hidden_width": "width_plus_2", "batch_size": "batch_x2"}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
