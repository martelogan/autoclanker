"""Small toy trainer for the Bayes-style showcase.

This version is intentionally closer to a developer-friendly mini-app:

- it trains a tiny score model over a few numeric samples,
- it exposes an obvious "include the pair feature or not" decision,
- and it has an equally obvious risky branch where wider state plus larger
  batches exceed the memory budget.
"""

from __future__ import annotations

import argparse
import json
import runpy

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

TRAIN_SAMPLES = (
    {"length": 2.0, "keywords": 1.0, "target": 1.8},
    {"length": 3.0, "keywords": 0.0, "target": 2.3},
    {"length": 1.0, "keywords": 2.0, "target": 1.9},
    {"length": 4.0, "keywords": 1.0, "target": 3.4},
)

DEMO_QUERY = {"length": 3.5, "keywords": 1.5}


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Tunable knobs for the tiny trainer."""

    optim_lr: str = "lr_default"
    feature_mode: str = "single_only"
    hidden_width: str = "width_default"
    batch_size: str = "batch_default"
    warmup_mode: str = "warmup_default"


DEFAULT_CONFIG = TrainerConfig()
VARIANT_NAMES = ("baseline", "local_observed_best", "belief_guided", "risky_oom")
SHOWCASE_DIR = Path(__file__).resolve().parent


def _load_variant_overrides(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    namespace = runpy.run_path(str(SHOWCASE_DIR / "variants" / f"{variant_name}.py"))
    return cast(dict[str, object], cast(dict[str, object], namespace)["OVERRIDES"])


def _lr_value(config: TrainerConfig) -> float:
    if config.optim_lr == "lr_x1_5":
        return 0.15
    if config.optim_lr == "lr_x2":
        return 0.20
    return 0.10


def _features(sample: dict[str, float], config: TrainerConfig) -> list[float]:
    values = [sample["length"], sample["keywords"], 1.0]
    if config.feature_mode == "pair_feature":
        values.append(sample["length"] * sample["keywords"])
    if config.hidden_width == "width_plus_2":
        values.extend(
            [
                sample["length"] ** 2,
                sample["keywords"] ** 2,
            ]
        )
    return values


def train_demo_model(config: TrainerConfig) -> dict[str, Any]:
    """Run a tiny deterministic fitting loop so the app feels like a real program."""

    feature_count = len(_features(cast(dict[str, float], TRAIN_SAMPLES[0]), config))
    weights = [0.0 for _ in range(feature_count)]
    lr = _lr_value(config)
    if config.warmup_mode == "warmup_short":
        lr *= 0.9

    losses: list[float] = []
    for sample in TRAIN_SAMPLES:
        features = _features(cast(dict[str, float], sample), config)
        prediction = sum(
            weight * value for weight, value in zip(weights, features, strict=True)
        )
        error = float(sample["target"]) - prediction
        losses.append(abs(error))
        for index, value in enumerate(features):
            weights[index] += lr * error * value * 0.05

    query_features = _features(cast(dict[str, float], DEMO_QUERY), config)
    query_prediction = sum(
        weight * value for weight, value in zip(weights, query_features, strict=True)
    )
    return {
        "mean_abs_error": round(sum(losses) / len(losses), 3),
        "query_prediction": round(query_prediction, 3),
        "feature_count": feature_count,
    }


def render_demo(config: TrainerConfig) -> dict[str, Any]:
    """Show a few sample outputs so the toy app is understandable on its own."""

    training_summary = train_demo_model(config)
    return {
        "app_kind": "toy_pairwise_trainer",
        "what_the_app_does": (
            "Fit a tiny numeric score model over a handful of samples, then "
            "predict one held-out query."
        ),
        "sample_training_rows": list(TRAIN_SAMPLES),
        "sample_query": dict(DEMO_QUERY),
        "sample_prediction": training_summary["query_prediction"],
        "training_summary": training_summary,
        "optimization_surface": [
            "optim_lr changes how aggressively the tiny model updates.",
            "feature_mode toggles whether an interaction feature is available.",
            "hidden_width approximates a wider internal representation.",
            "batch_size and warmup_mode affect the cost / stability trade-off.",
        ],
    }


def evaluate_config(config: TrainerConfig) -> dict[str, Any]:
    """Return deterministic utility and failure metrics for one trainer config."""

    delta_perf = 0.0
    peak_vram_mb = 18_000.0
    nondefault_gene_count = 0

    if config.optim_lr == "lr_x1_5":
        delta_perf += 0.35
        nondefault_gene_count += 1
    elif config.optim_lr == "lr_x2":
        delta_perf += 0.18
        nondefault_gene_count += 1

    if config.feature_mode == "pair_feature":
        delta_perf += 0.18
        nondefault_gene_count += 1

    if config.hidden_width == "width_plus_2":
        delta_perf += 0.12
        peak_vram_mb += 3_500.0
        nondefault_gene_count += 1

    if config.batch_size == "batch_x2":
        delta_perf += 0.10
        peak_vram_mb += 3_000.0
        nondefault_gene_count += 1

    if config.warmup_mode == "warmup_short":
        delta_perf += 0.03
        nondefault_gene_count += 1

    # This is the good unseen pair the Bayes demo is meant to promote.
    if config.optim_lr == "lr_x1_5" and config.feature_mode == "pair_feature":
        delta_perf += 0.24

    # This is the risky branch that feasibility beliefs should penalize.
    if config.hidden_width == "width_plus_2" and config.batch_size == "batch_x2":
        delta_perf -= 0.40
        peak_vram_mb += 2_500.0

    if config.optim_lr == "lr_x2" and config.feature_mode == "pair_feature":
        delta_perf -= 0.18
        peak_vram_mb += 500.0

    vram_overage_units = max(peak_vram_mb - 24_000.0, 0.0) / 1_000.0
    utility = delta_perf - (0.1 * nondefault_gene_count) - (0.01 * vram_overage_units)
    status = "valid"

    if config.optim_lr == "lr_x2" and config.feature_mode == "pair_feature":
        status = "runtime_fail"
        utility -= 0.25
    elif peak_vram_mb > 24_800.0:
        status = "oom"
        utility -= 0.3

    return {
        "status": status,
        "delta_perf": round(delta_perf, 3),
        "utility": round(utility, 3),
        "peak_vram_mb": round(peak_vram_mb, 3),
        "nondefault_gene_count": nondefault_gene_count,
        "config": asdict(config),
    }


def _config_for_variant(variant_name: str) -> TrainerConfig:
    return TrainerConfig(**_load_variant_overrides(variant_name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the tiny trainer app directly. This shows the toy program, not "
            "the real autoclanker-backed session flow."
        )
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=VARIANT_NAMES,
        help="Which code variant to render.",
    )
    args = parser.parse_args(argv)
    config = _config_for_variant(args.variant)
    payload = render_demo(config)
    payload.update(
        {
            "variant": args.variant,
            "config": asdict(config),
            "benchmark_preview": evaluate_config(config),
            "next_step_benchmark_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/bayes_pair_feature_trainer/benchmark.py "
                f"--variant {args.variant}"
            ),
            "next_step_replay_command": (
                "./bin/dev exec -- python "
                "scripts/showcase/replay_backing_exercise.py --showcase "
                "bayes_pair_feature_trainer"
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
