"""Compatibility wrapper for the Bayes-style belief-guided variant."""

from __future__ import annotations

import runpy

from collections.abc import Callable
from pathlib import Path
from typing import cast


def measure() -> dict[str, object]:
    namespace = runpy.run_path(str(Path(__file__).resolve().with_name("benchmark.py")))
    measure_variant = cast(
        Callable[[str], dict[str, object]], namespace["measure_variant"]
    )
    return measure_variant("belief_guided")


def main() -> int:
    namespace = runpy.run_path(str(Path(__file__).resolve().with_name("benchmark.py")))
    run_variant_name = cast(Callable[[str], int], namespace["run_variant_name"])
    return run_variant_name("belief_guided")


if __name__ == "__main__":
    raise SystemExit(main())
