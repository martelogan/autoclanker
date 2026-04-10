"""Benchmark harness for the cEvolve-style showcase."""

from __future__ import annotations

import argparse
import json
import runpy

from dataclasses import replace
from pathlib import Path
from typing import cast

SHOWCASE = "cevolve_sort_partition"
LIVE_EXERCISE = "examples/live_exercises/cevolve_synergy"
REPLAY_COMMAND = (
    "./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py "
    "--showcase cevolve_sort_partition"
)
VARIANT_NAMES = ("baseline", "single_threshold", "single_partition", "optimized")
SHOWCASE_DIR = Path(__file__).resolve().parent
APP_NAMESPACE = runpy.run_path(str(SHOWCASE_DIR / "app.py"))
DEFAULT_STRATEGY = APP_NAMESPACE["DEFAULT_STRATEGY"]
evaluate_strategy = APP_NAMESPACE["evaluate_strategy"]


def _variant_namespace(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    return cast(
        dict[str, object],
        runpy.run_path(str(SHOWCASE_DIR / "variants" / f"{variant_name}.py")),
    )


def measure_variant(variant_name: str) -> dict[str, object]:
    namespace = _variant_namespace(variant_name)
    overrides = cast(dict[str, object], namespace["OVERRIDES"])
    strategy = replace(DEFAULT_STRATEGY, **overrides)
    payload = evaluate_strategy(strategy)
    payload.update(
        {
            "scenario": SHOWCASE,
            "demo_role": "readable_snapshot_only",
            "demo_layout": "app.py + benchmark.py + variants/",
            "variant": namespace["VARIANT_NAME"],
            "variant_summary": namespace["VARIANT_SUMMARY"],
            "approach_fit": namespace["APPROACH_FIT"],
            "what_changed": namespace["WHAT_CHANGED"],
            "what_is_being_optimized": (
                "Lower time_ms for the tiny integer sorter by finding a strong "
                "combination of interacting strategy switches."
            ),
            "manual_app_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/cevolve_sort_partition/app.py "
                f"--variant {variant_name}"
            ),
            "manual_benchmark_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/cevolve_sort_partition/benchmark.py "
                f"--variant {variant_name}"
            ),
            "autoclanker_relationship": (
                "This toy app is only a readable mirror. The replay command runs "
                "the actual autoclanker-backed adapter demo."
            ),
            "llm_note": (
                "The toy app does not call an LLM. It exists to make the "
                "interaction-heavy search landscape visually obvious."
            ),
            "backing_live_exercise": LIVE_EXERCISE,
            "backing_replay_command": REPLAY_COMMAND,
        }
    )
    return payload


def run_variant_name(variant_name: str) -> int:
    print(json.dumps(measure_variant(variant_name), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark one cEvolve-style showcase variant."
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=VARIANT_NAMES,
        help="Which variant to benchmark.",
    )
    args = parser.parse_args(argv)
    return run_variant_name(args.variant)


if __name__ == "__main__":
    raise SystemExit(main())
