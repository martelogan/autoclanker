"""Best observed local move for the Bayes-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "local_observed_best"
VARIANT_SUMMARY = "Take the best single move that has already been observed."
APPROACH_FIT = (
    "This is what naive 'best seen so far' reasoning would choose before using "
    "structured beliefs."
)
WHAT_CHANGED = ['Increase OPTIM_LR from "lr_default" to "lr_x1_5".']
OVERRIDES = {"optim_lr": "lr_x1_5"}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
