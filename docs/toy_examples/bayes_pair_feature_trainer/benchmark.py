"""Benchmark harness for the Bayes-style showcase."""

from __future__ import annotations

import argparse
import json
import runpy

from dataclasses import replace
from pathlib import Path
from typing import cast

SHOWCASE = "bayes_pair_feature_trainer"
LIVE_EXERCISE = "examples/live_exercises/bayes_complex"
REPLAY_COMMAND = (
    "./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py "
    "--showcase bayes_pair_feature_trainer"
)
VARIANT_NAMES = ("baseline", "local_observed_best", "belief_guided", "risky_oom")
SHOWCASE_DIR = Path(__file__).resolve().parent
APP_NAMESPACE = runpy.run_path(str(SHOWCASE_DIR / "app.py"))
DEFAULT_CONFIG = APP_NAMESPACE["DEFAULT_CONFIG"]
evaluate_config = APP_NAMESPACE["evaluate_config"]


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
    config = replace(DEFAULT_CONFIG, **overrides)
    payload = evaluate_config(config)
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
                "Raise utility for the tiny trainer by finding a stronger "
                "combination while avoiding OOM-prone branches."
            ),
            "manual_app_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/bayes_pair_feature_trainer/app.py "
                f"--variant {variant_name}"
            ),
            "manual_benchmark_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/bayes_pair_feature_trainer/benchmark.py "
                f"--variant {variant_name}"
            ),
            "autoclanker_relationship": (
                "This toy app is only a readable mirror. The replay command runs "
                "the actual autoclanker session flow that uses beliefs."
            ),
            "llm_note": (
                "The toy app itself does not call an LLM. The real autoclanker "
                "session flow is where human or LLM-authored beliefs enter."
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
        description="Benchmark one Bayes-style showcase variant."
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
