"""OOM-prone variant for the autoresearch-style showcase."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast

VARIANT_NAME = "failure_variant"
VARIANT_SUMMARY = "Push the helper into a tempting but over-budget regime."
APPROACH_FIT = (
    "This variant makes the resource constraint visible: some changes look fast "
    "or large, but the combination is not feasible."
)
WHAT_CHANGED = [
    "Increase history_depth from 4 to 6.",
    "Increase rerank_passes from 1 to 2.",
    "Increase micro_batch_tokens from 2048 to 4096.",
    "Increase learning_rate from 0.04 to 0.05.",
]
OVERRIDES = {
    "history_depth": 6,
    "rerank_passes": 2,
    "micro_batch_tokens": 4_096,
    "learning_rate": 0.05,
}


def main() -> int:
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "benchmark.py")
    )
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name(VARIANT_NAME)


if __name__ == "__main__":
    raise SystemExit(main())
