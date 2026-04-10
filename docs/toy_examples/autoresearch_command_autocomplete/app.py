"""Small command-history autocomplete app for the autoresearch-style showcase.

`app.py` is intentionally a tiny standalone program a developer can read first.
It simulates a shell-history autocomplete helper:

- it looks at recent commands,
- builds suffix suggestions for a few prefixes,
- and exposes a small set of tunable training/runtime knobs.

The real `autoclanker` / autoresearch-backed demo lives elsewhere. This file is
the readable mirror that makes the optimization surface easy to see.
"""

from __future__ import annotations

import argparse
import json
import runpy

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

DEMO_HISTORY = (
    "git status",
    "git checkout -b codex/demo",
    "git commit -am refine-showcase",
    "uv run pytest tests/test_cli.py",
    "uv run pytest tests/test_modeling.py",
    "uv sync",
    "python scripts/export_metrics.py --format json",
    "python scripts/export_metrics.py --format csv",
    "python scripts/replay_session.py --session alpha",
    "git push origin codex/demo",
)

DEMO_PREFIXES = ("git p", "uv r", "python s")


@dataclass(frozen=True, slots=True)
class AutocompleteProfile:
    """Tunable knobs for the tiny autocomplete trainer."""

    history_depth: int = 4
    rerank_passes: int = 1
    micro_batch_tokens: int = 2_048
    learning_rate: float = 0.04
    warmup_ratio: float = 0.0


DEFAULT_PROFILE = AutocompleteProfile()
VARIANT_NAMES = ("baseline", "optimized", "failure_variant")
SHOWCASE_DIR = Path(__file__).resolve().parent


def _load_variant_overrides(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    namespace = runpy.run_path(str(SHOWCASE_DIR / "variants" / f"{variant_name}.py"))
    return cast(dict[str, object], cast(dict[str, object], namespace)["OVERRIDES"])


def _recent_commands(profile: AutocompleteProfile) -> tuple[str, ...]:
    return DEMO_HISTORY[-profile.history_depth :]


def suggest_commands(prefix: str, profile: AutocompleteProfile) -> list[str]:
    """Return a few deterministic autocomplete suggestions for one prefix."""

    suffix_counts: Counter[str] = Counter()
    for command in _recent_commands(profile):
        if not command.startswith(prefix):
            continue
        suffix_counts[command] += 1

    ranked = sorted(
        suffix_counts.items(),
        key=lambda item: (-item[1], len(item[0]), item[0]),
    )
    suggestions = [command for command, _ in ranked]

    if not suggestions:
        fallback = {
            "git p": "git push origin codex/demo",
            "uv r": "uv run pytest",
            "python s": "python scripts/replay_session.py",
        }
        suggestions.append(fallback.get(prefix, prefix))

    if profile.rerank_passes > 1 and len(suggestions) > 1:
        suggestions = sorted(
            suggestions, key=lambda suggestion: (len(suggestion), suggestion)
        )

    return suggestions[:3]


def render_demo(profile: AutocompleteProfile) -> dict[str, Any]:
    """Show what the tiny app does for a few sample prefixes."""

    prompt_results = {
        prefix: suggest_commands(prefix, profile) for prefix in DEMO_PREFIXES
    }
    return {
        "app_kind": "toy_command_autocomplete",
        "what_the_app_does": (
            "Given a short shell prefix, suggest likely recent commands from a "
            "tiny in-memory history."
        ),
        "sample_history": list(_recent_commands(profile)),
        "sample_prefixes": list(DEMO_PREFIXES),
        "sample_suggestions": prompt_results,
        "optimization_surface": [
            "history_depth changes how much recent context the helper uses.",
            "rerank_passes changes how aggressively candidates are re-ordered.",
            "micro_batch_tokens approximates the training batch budget.",
            "learning_rate and warmup_ratio affect the toy quality/resource trade-off.",
        ],
    }


def evaluate_profile(profile: AutocompleteProfile) -> dict[str, Any]:
    """Return deterministic quality and resource metrics for one profile."""

    val_bpb = 1.000
    peak_vram_gb = 18.0
    throughput_tokens_per_sec = 250_000.0
    status = "valid"

    # More recent command context improves suggestion quality but costs memory.
    if profile.history_depth == 6:
        val_bpb -= 0.010
        peak_vram_gb += 2.2
        throughput_tokens_per_sec -= 6_000.0

    # Extra reranking is not worth it for this tiny workload.
    if profile.rerank_passes == 2:
        val_bpb += 0.014
        throughput_tokens_per_sec += 9_000.0

    # Slightly smaller micro-batches improve the toy objective on this budget.
    if profile.micro_batch_tokens == 1_536:
        val_bpb -= 0.011
        peak_vram_gb -= 1.1
        throughput_tokens_per_sec -= 3_000.0
    elif profile.micro_batch_tokens == 4_096:
        val_bpb += 0.006
        peak_vram_gb += 4.8
        throughput_tokens_per_sec += 3_000.0

    if profile.learning_rate == 0.03:
        val_bpb -= 0.008
    elif profile.learning_rate == 0.05:
        val_bpb += 0.007

    if profile.warmup_ratio == 0.1:
        val_bpb -= 0.005

    # Deep recent context plus an oversized batch trips the memory budget.
    if profile.history_depth == 6 and profile.micro_batch_tokens == 4_096:
        status = "oom"
        val_bpb += 0.025
        peak_vram_gb += 1.0
        throughput_tokens_per_sec = 0.0

    return {
        "status": status,
        "val_bpb": round(val_bpb, 3),
        "peak_vram_gb": round(peak_vram_gb, 3),
        "throughput_tokens_per_sec": round(throughput_tokens_per_sec, 3),
        "profile": asdict(profile),
    }


def _profile_for_variant(variant_name: str) -> AutocompleteProfile:
    return AutocompleteProfile(**_load_variant_overrides(variant_name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the tiny autocomplete app directly. This shows what the toy app "
            "does, not the real autoclanker-backed demo."
        )
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=VARIANT_NAMES,
        help="Which code variant to render.",
    )
    args = parser.parse_args(argv)
    profile = _profile_for_variant(args.variant)
    payload = render_demo(profile)
    payload.update(
        {
            "variant": args.variant,
            "profile": asdict(profile),
            "benchmark_preview": evaluate_profile(profile),
            "next_step_benchmark_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/autoresearch_command_autocomplete/benchmark.py "
                f"--variant {args.variant}"
            ),
            "next_step_replay_command": (
                "./bin/dev exec -- python "
                "scripts/showcase/replay_backing_exercise.py --showcase "
                "autoresearch_command_autocomplete"
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
