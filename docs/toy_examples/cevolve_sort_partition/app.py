"""Small integer-sorting app for the cEvolve-style showcase.

This file is meant to read like a tiny real program:

- it sorts a list of integers,
- it exposes algorithm choices a developer can reason about,
- and it demonstrates why a combination of compatible edits matters more than
  isolated local tweaks.
"""

from __future__ import annotations

import argparse
import json
import runpy

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

DEMO_BATCHES = (
    [18, 3, 15, 1, 9, 14, 7, 6, 10, 5, 11, 2, 16, 12, 4, 8, 13, 17],
    [45, 22, 41, 5, 31, 17, 29, 3, 37, 11, 19, 7, 43, 13],
    [88, 12, 64, 32, 48, 16, 80, 24, 72, 40, 56, 8],
)


@dataclass(frozen=True, slots=True)
class SortStrategy:
    """Tunable quicksort-style strategy knobs."""

    insertion_threshold: int = 16
    partition_scheme: str = "lomuto"
    pivot_strategy: str = "median_of_three"
    use_iterative: bool = False


@dataclass(slots=True)
class OperationTrace:
    comparisons: int = 0
    swaps: int = 0
    stack_pushes: int = 0
    insertion_steps: int = 0


DEFAULT_STRATEGY = SortStrategy()
VARIANT_NAMES = ("baseline", "single_threshold", "single_partition", "optimized")
SHOWCASE_DIR = Path(__file__).resolve().parent


def _load_variant_overrides(variant_name: str) -> dict[str, object]:
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    namespace = runpy.run_path(str(SHOWCASE_DIR / "variants" / f"{variant_name}.py"))
    return cast(dict[str, object], cast(dict[str, object], namespace)["OVERRIDES"])


def _pivot_index(values: list[int], low: int, high: int, strategy: SortStrategy) -> int:
    if strategy.pivot_strategy == "middle":
        return (low + high) // 2
    if strategy.pivot_strategy == "random":
        return low + ((high - low) * 37 % max(high - low + 1, 1))

    mid = (low + high) // 2
    trio = [(values[low], low), (values[mid], mid), (values[high], high)]
    trio.sort(key=lambda item: item[0])
    return trio[1][1]


def _swap(values: list[int], left: int, right: int, trace: OperationTrace) -> None:
    if left == right:
        return
    values[left], values[right] = values[right], values[left]
    trace.swaps += 1


def _insertion_sort(
    values: list[int], low: int, high: int, trace: OperationTrace
) -> None:
    for index in range(low + 1, high + 1):
        current = values[index]
        cursor = index - 1
        while cursor >= low:
            trace.comparisons += 1
            if values[cursor] <= current:
                break
            values[cursor + 1] = values[cursor]
            trace.insertion_steps += 1
            cursor -= 1
        values[cursor + 1] = current


def _partition_lomuto(
    values: list[int],
    low: int,
    high: int,
    strategy: SortStrategy,
    trace: OperationTrace,
) -> int:
    pivot_idx = _pivot_index(values, low, high, strategy)
    _swap(values, pivot_idx, high, trace)
    pivot = values[high]
    store = low
    for cursor in range(low, high):
        trace.comparisons += 1
        if values[cursor] <= pivot:
            _swap(values, store, cursor, trace)
            store += 1
    _swap(values, store, high, trace)
    return store


def _partition_hoare(
    values: list[int],
    low: int,
    high: int,
    strategy: SortStrategy,
    trace: OperationTrace,
) -> int:
    pivot = values[_pivot_index(values, low, high, strategy)]
    left = low - 1
    right = high + 1
    while True:
        while True:
            left += 1
            trace.comparisons += 1
            if values[left] >= pivot:
                break
        while True:
            right -= 1
            trace.comparisons += 1
            if values[right] <= pivot:
                break
        if left >= right:
            return right
        _swap(values, left, right, trace)


def _sort_range_recursive(
    values: list[int],
    low: int,
    high: int,
    strategy: SortStrategy,
    trace: OperationTrace,
) -> None:
    if low >= high:
        return
    if high - low + 1 <= strategy.insertion_threshold:
        _insertion_sort(values, low, high, trace)
        return
    if strategy.partition_scheme == "hoare":
        pivot = _partition_hoare(values, low, high, strategy, trace)
        _sort_range_recursive(values, low, pivot, strategy, trace)
        _sort_range_recursive(values, pivot + 1, high, strategy, trace)
        return
    pivot = _partition_lomuto(values, low, high, strategy, trace)
    _sort_range_recursive(values, low, pivot - 1, strategy, trace)
    _sort_range_recursive(values, pivot + 1, high, strategy, trace)


def _sort_range_iterative(
    values: list[int], strategy: SortStrategy, trace: OperationTrace
) -> None:
    stack: list[tuple[int, int]] = [(0, len(values) - 1)]
    while stack:
        low, high = stack.pop()
        trace.stack_pushes += 1
        if low >= high:
            continue
        if high - low + 1 <= strategy.insertion_threshold:
            _insertion_sort(values, low, high, trace)
            continue
        if strategy.partition_scheme == "hoare":
            pivot = _partition_hoare(values, low, high, strategy, trace)
            stack.append((low, pivot))
            stack.append((pivot + 1, high))
        else:
            pivot = _partition_lomuto(values, low, high, strategy, trace)
            stack.append((low, pivot - 1))
            stack.append((pivot + 1, high))


def sort_numbers(
    values: list[int], strategy: SortStrategy
) -> tuple[list[int], OperationTrace]:
    """Sort one list using the chosen strategy and collect lightweight trace data."""

    copy = list(values)
    trace = OperationTrace()
    if not copy:
        return copy, trace
    if strategy.use_iterative:
        _sort_range_iterative(copy, strategy, trace)
    else:
        _sort_range_recursive(copy, 0, len(copy) - 1, strategy, trace)
    return copy, trace


def render_demo(strategy: SortStrategy) -> dict[str, Any]:
    """Show one sample input/output so the toy app is understandable on its own."""

    sample_input = list(DEMO_BATCHES[0])
    sorted_output, trace = sort_numbers(sample_input, strategy)
    return {
        "app_kind": "toy_integer_sorter",
        "what_the_app_does": (
            "Sort a small batch of integers using a configurable quicksort-style "
            "strategy."
        ),
        "sample_input": sample_input,
        "sample_output": sorted_output,
        "operation_trace": asdict(trace),
        "optimization_surface": [
            "insertion_threshold changes when the algorithm switches to insertion sort.",
            "partition_scheme changes how the pivot partition is formed.",
            "pivot_strategy changes which element becomes the pivot.",
            "use_iterative toggles recursion versus an explicit stack.",
        ],
    }


def evaluate_strategy(strategy: SortStrategy) -> dict[str, Any]:
    """Return deterministic performance metrics for one strategy."""

    workload_traces = []
    for batch in DEMO_BATCHES:
        _, trace = sort_numbers(list(batch), strategy)
        workload_traces.append(trace)

    total_comparisons = sum(trace.comparisons for trace in workload_traces)
    total_swaps = sum(trace.swaps for trace in workload_traces)
    total_stack_pushes = sum(trace.stack_pushes for trace in workload_traces)

    time_ms = 120.0
    synergy_bonus = 0.0
    robustness_score = 0.94

    if strategy.insertion_threshold == 32:
        time_ms -= 9.0
    elif strategy.insertion_threshold == 64:
        time_ms += 6.0

    if strategy.partition_scheme == "hoare":
        time_ms -= 5.0
        robustness_score += 0.01

    if strategy.pivot_strategy == "middle":
        time_ms += 3.0
        robustness_score -= 0.01
    elif strategy.pivot_strategy == "random":
        time_ms += 8.0
        robustness_score -= 0.03

    if strategy.use_iterative:
        time_ms -= 1.0

    if strategy.insertion_threshold == 32 and strategy.partition_scheme == "hoare":
        time_ms -= 12.0
        synergy_bonus += 12.0
        robustness_score += 0.02

    if (
        strategy.insertion_threshold == 32
        and strategy.partition_scheme == "hoare"
        and strategy.use_iterative
    ):
        time_ms -= 6.0
        synergy_bonus += 6.0
        robustness_score += 0.01

    if strategy.pivot_strategy == "middle" and strategy.partition_scheme == "hoare":
        time_ms += 4.0
    if strategy.pivot_strategy == "random" and strategy.use_iterative:
        time_ms += 5.0
    if strategy.insertion_threshold == 64 and strategy.use_iterative:
        time_ms += 7.0

    return {
        "status": "valid",
        "time_ms": round(time_ms, 3),
        "synergy_bonus": round(synergy_bonus, 3),
        "robustness_score": round(robustness_score, 3),
        "strategy": asdict(strategy),
        "operation_counts": {
            "comparisons": total_comparisons,
            "swaps": total_swaps,
            "stack_pushes": total_stack_pushes,
        },
    }


def _strategy_for_variant(variant_name: str) -> SortStrategy:
    return SortStrategy(**_load_variant_overrides(variant_name))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the tiny sorter app directly. This shows the toy program, not the "
            "real autoclanker-backed demo."
        )
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=VARIANT_NAMES,
        help="Which code variant to render.",
    )
    args = parser.parse_args(argv)
    strategy = _strategy_for_variant(args.variant)
    payload = render_demo(strategy)
    payload.update(
        {
            "variant": args.variant,
            "strategy": asdict(strategy),
            "benchmark_preview": evaluate_strategy(strategy),
            "next_step_benchmark_command": (
                "./bin/dev exec -- python "
                "docs/toy_examples/cevolve_sort_partition/benchmark.py "
                f"--variant {args.variant}"
            ),
            "next_step_replay_command": (
                "./bin/dev exec -- python "
                "scripts/showcase/replay_backing_exercise.py --showcase "
                "cevolve_sort_partition"
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
