"""Optimized variant for the autoresearch-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "optimized"
VARIANT_SUMMARY = "Combine several individually helpful knob changes."
APPROACH_FIT = (
    "Each change helps for its own reason, so greedy tuning and simple priors "
    "work well here."
)
WHAT_CHANGED = [
    "Increase history_depth from 4 to 6 so the helper sees more recent commands.",
    "Reduce micro_batch_tokens from 2048 to 1536 on this toy memory budget.",
    "Lower learning_rate from 0.04 to 0.03.",
    "Add a small warmup_ratio of 0.1.",
]
OVERRIDES = {
    "history_depth": 6,
    "micro_batch_tokens": 1_536,
    "learning_rate": 0.03,
    "warmup_ratio": 0.1,
}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
