from __future__ import annotations

import json
import runpy

from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]


def _measure(showcase: str, variant: str) -> dict[str, object]:
    namespace = runpy.run_path(
        str(ROOT / "docs" / "toy_examples" / showcase / "benchmark.py")
    )
    measure_func = namespace.get("measure_variant")
    if not callable(measure_func):
        raise RuntimeError(
            f"{showcase} benchmark.py does not define measure_variant()."
        )
    payload = measure_func(variant)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{showcase} measure_variant() did not return an object.")
    return cast(dict[str, object], payload)


def main() -> int:
    showcases = {
        "autoresearch_command_autocomplete": {
            "about": {
                "goal": (
                    "Lower val_bpb for a tiny command-autocomplete helper by "
                    "tuning mostly independent knobs."
                ),
                "toy_code_role": (
                    "Readable standalone toy app with app.py, benchmark.py, and variants/."
                ),
                "manual_walkthrough": [
                    "./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/app.py --variant baseline",
                    "./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/benchmark.py --variant optimized",
                    "./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase autoresearch_command_autocomplete",
                ],
                "primary_files": {
                    "app": "docs/toy_examples/autoresearch_command_autocomplete/app.py",
                    "benchmark": "docs/toy_examples/autoresearch_command_autocomplete/benchmark.py",
                    "variants_dir": "docs/toy_examples/autoresearch_command_autocomplete/variants",
                },
                "backing_replay_command": (
                    "./bin/dev exec -- python "
                    "scripts/showcase/replay_backing_exercise.py --showcase "
                    "autoresearch_command_autocomplete"
                ),
            },
            "baseline": _measure("autoresearch_command_autocomplete", "baseline"),
            "optimized": _measure("autoresearch_command_autocomplete", "optimized"),
            "failure_variant": _measure(
                "autoresearch_command_autocomplete", "failure_variant"
            ),
        },
        "cevolve_sort_partition": {
            "about": {
                "goal": (
                    "Lower time_ms for a tiny integer sorter by finding a strong "
                    "combination of interacting switches."
                ),
                "toy_code_role": (
                    "Readable standalone toy app with app.py, benchmark.py, and variants/."
                ),
                "manual_walkthrough": [
                    "./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/app.py --variant baseline",
                    "./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/benchmark.py --variant optimized",
                    "./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase cevolve_sort_partition",
                ],
                "primary_files": {
                    "app": "docs/toy_examples/cevolve_sort_partition/app.py",
                    "benchmark": "docs/toy_examples/cevolve_sort_partition/benchmark.py",
                    "variants_dir": "docs/toy_examples/cevolve_sort_partition/variants",
                },
                "backing_replay_command": (
                    "./bin/dev exec -- python "
                    "scripts/showcase/replay_backing_exercise.py --showcase "
                    "cevolve_sort_partition"
                ),
            },
            "baseline": _measure("cevolve_sort_partition", "baseline"),
            "single_threshold": _measure("cevolve_sort_partition", "single_threshold"),
            "single_partition": _measure("cevolve_sort_partition", "single_partition"),
            "optimized": _measure("cevolve_sort_partition", "optimized"),
        },
        "bayes_pair_feature_trainer": {
            "about": {
                "goal": (
                    "Raise utility for a tiny pair-feature trainer by using beliefs to "
                    "promote a better unseen pair while avoiding OOM."
                ),
                "toy_code_role": (
                    "Readable standalone toy app with app.py, benchmark.py, and variants/."
                ),
                "manual_walkthrough": [
                    "./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/app.py --variant baseline",
                    "./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/benchmark.py --variant belief_guided",
                    "./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer",
                ],
                "primary_files": {
                    "app": "docs/toy_examples/bayes_pair_feature_trainer/app.py",
                    "benchmark": "docs/toy_examples/bayes_pair_feature_trainer/benchmark.py",
                    "variants_dir": "docs/toy_examples/bayes_pair_feature_trainer/variants",
                },
                "backing_replay_command": (
                    "./bin/dev exec -- python "
                    "scripts/showcase/replay_backing_exercise.py --showcase "
                    "bayes_pair_feature_trainer"
                ),
            },
            "baseline": _measure("bayes_pair_feature_trainer", "baseline"),
            "local_observed_best": _measure(
                "bayes_pair_feature_trainer", "local_observed_best"
            ),
            "belief_guided": _measure("bayes_pair_feature_trainer", "belief_guided"),
            "risky_oom": _measure("bayes_pair_feature_trainer", "risky_oom"),
        },
    }

    autoresearch = cast(
        dict[str, dict[str, object]], showcases["autoresearch_command_autocomplete"]
    )
    autoresearch["comparison"] = {
        "val_bpb_improvement": round(
            float(autoresearch["baseline"]["val_bpb"])
            - float(autoresearch["optimized"]["val_bpb"]),
            3,
        ),
        "failure_status": str(autoresearch["failure_variant"]["status"]),
    }

    cevolve = cast(dict[str, dict[str, object]], showcases["cevolve_sort_partition"])
    best_single_time_ms = min(
        float(cevolve["single_threshold"]["time_ms"]),
        float(cevolve["single_partition"]["time_ms"]),
    )
    cevolve["comparison"] = {
        "time_ms_improvement": round(
            float(cevolve["baseline"]["time_ms"])
            - float(cevolve["optimized"]["time_ms"]),
            3,
        ),
        "synergy_margin_vs_best_single": round(
            best_single_time_ms - float(cevolve["optimized"]["time_ms"]),
            3,
        ),
    }

    bayes = cast(dict[str, dict[str, object]], showcases["bayes_pair_feature_trainer"])
    bayes["comparison"] = {
        "belief_guided_utility_gain_over_local_best": round(
            float(bayes["belief_guided"]["utility"])
            - float(bayes["local_observed_best"]["utility"]),
            3,
        ),
        "risky_status": str(bayes["risky_oom"]["status"]),
    }

    print(json.dumps(showcases, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
