"""Belief-guided unseen pair for the Bayes-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "belief_guided"
VARIANT_SUMMARY = (
    "Add the unseen pair-feature change that pairs well with the good LR move."
)
APPROACH_FIT = (
    "This is the Bayes story: a structured prior can justify trying a better "
    "unseen pair than the best already-observed single edit."
)
WHAT_CHANGED = [
    'Increase OPTIM_LR from "lr_default" to "lr_x1_5".',
    'Switch FEATURE_MODE from "single_only" to "pair_feature".',
]
OVERRIDES = {"optim_lr": "lr_x1_5", "feature_mode": "pair_feature"}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
