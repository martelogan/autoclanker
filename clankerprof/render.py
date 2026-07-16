from __future__ import annotations

import csv
import io
import json
import math
import re

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import cast

from clankerprof.analysis import (
    DEFAULT_RUNTIME_RULES,
    BoundaryAnalysisResult,
    BoundaryStats,
    RuntimeRuleSet,
    SliceAnalysisOptions,
    SliceAnalysisResult,
    SliceStats,
    extract_library_path,
    format_time,
)
from clankerprof.model import TimeNs
from clankerprof.stats import CategoryStats


def render_target_json(
    results: Mapping[str, Mapping[str, CategoryStats]],
) -> dict[str, object]:
    parents: dict[str, object] = {}
    for parent, categories in results.items():
        total = sum(stats.cpu_time for stats in categories.values())
        parents[parent] = {
            "total_time_ns": total,
            "categories": [
                {
                    "name": category,
                    "time_ns": stats.cpu_time,
                    "pct": (_f64_ratio(stats.cpu_time, total) * 100) if total else 0,
                    "samples": stats.sample_count,
                    "leaf_functions": {
                        name: asdict(metrics)
                        for name, metrics in sorted(
                            stats.functions.items(),
                            key=lambda item: item[1].cpu_time,
                            reverse=True,
                        )
                    },
                    "files": sorted(stats.files),
                    "folded_from": dict(
                        sorted(
                            stats.folded_from.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ),
                    "semantic_callers": {
                        leaf: {
                            "count": metrics.count,
                            "caller_names": dict(
                                sorted(
                                    metrics.caller_names.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            ),
                            "caller_files": dict(
                                sorted(
                                    metrics.caller_files.items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            ),
                        }
                        for leaf, metrics in sorted(
                            stats.semantic_callers.items(),
                            key=lambda item: item[1].count,
                            reverse=True,
                        )
                    },
                }
                for category, stats in sorted(
                    categories.items(), key=lambda item: item[1].cpu_time, reverse=True
                )
            ],
        }
    return {"tool": "clankerprof_targets", "parents": parents}


def render_boundary_json(
    result: BoundaryAnalysisResult,
    *,
    top: int | None = None,
) -> dict[str, object]:
    return {
        "tool": "clankerprof_boundaries",
        "summary": {
            "total_time_ns": result.total_time_ns,
            "unique_frame_count": result.unique_frame_count,
        },
        "boundaries": [
            _render_boundary(
                boundary,
                profile_total=result.total_time_ns,
                top=top,
            )
            for boundary in result.boundaries
        ],
    }


def _f64_ratio(numerator: TimeNs, denominator: TimeNs) -> float:
    """Aggregate ratio with both operands rounded to f64 BEFORE dividing.

    Mirrors the Rust core's `as f64` operand casts operand-for-operand: this
    is the shared arithmetic contract for percentage fields, so artifacts
    stay byte-identical even when aggregates exceed 2**53 (below that the
    result is identical to exact integer division). Adopted from Rust as the
    shared contract in audit round 3 (R3-04).
    """
    return float(numerator) / float(denominator)


def _pct_of(numerator: TimeNs, total: TimeNs) -> float:
    """Percentage with the shared zero-total arm.

    Valid signed samples can cancel a parent's total to exactly zero; every
    percentage over a zero total renders as 0 in both implementations —
    never a ZeroDivisionError, inf, or NaN.
    """
    return _f64_ratio(numerator, total) * 100 if total else 0.0


def _finite_attributable(name: str, estimate: float) -> float:
    """Attributable estimates — input or scaled — must stay JSON-representable.

    Scaling a finite metric by a >100% bucket share can overflow to infinity;
    failing closed here (instead of at serialization) names the offending
    metric and keeps the message identical across both implementations.
    """
    if not math.isfinite(estimate):
        raise ValueError(f"Attributable estimate for '{name}' is not finite.")
    return estimate


def _scale_attributables(
    attributables: Mapping[str, float],
    total: TimeNs,
    value: TimeNs,
) -> dict[str, float]:
    # Signed shares: a -10/-10 row is 100% of its scope, so estimates scale
    # for any nonzero total; only an exactly-zero total (undefined share)
    # suppresses them, mirroring the zero-arm of the pct fields.
    if not attributables or total == 0:
        return {}
    return {
        name: _finite_attributable(name, metric_value * _f64_ratio(value, total))
        for name, metric_value in sorted(attributables.items())
    }


def _render_boundary(
    boundary: BoundaryStats,
    *,
    profile_total: TimeNs,
    top: int | None,
) -> dict[str, object]:
    total = boundary.total_time
    bucketed_categories = {
        category for categories in boundary.buckets.values() for category in categories
    }
    buckets = [
        _render_boundary_bucket(boundary, label, categories)
        for label, categories in boundary.buckets.items()
    ]
    leftover = tuple(
        category
        for category, stats in sorted(
            boundary.categories.items(),
            key=lambda item: item[1].cpu_time,
            reverse=True,
        )
        if category not in bucketed_categories and _category_renderable(stats)
    )
    if leftover:
        buckets.append(_render_boundary_bucket(boundary, "Other", leftover))
    # A bucket is kept while any category row rendered: signed categories can
    # cancel to a zero bucket total without being omittable zero rows.
    buckets = [
        bucket for bucket in buckets if cast("list[object]", bucket["categories"])
    ]
    buckets.sort(key=lambda bucket: cast(int, bucket["time_ns"]), reverse=True)
    return {
        "name": boundary.name,
        "total_time_ns": total,
        "pct_of_profile": (
            _f64_ratio(total, profile_total) * 100 if profile_total else 0
        ),
        "samples": boundary.sample_count,
        "attributable_estimates": {
            name: _finite_attributable(name, value)
            for name, value in sorted(boundary.attributables.items())
        },
        "buckets": buckets,
        "domains": [
            _render_domain(boundary, name, top=top)
            for name, stats in sorted(
                boundary.domains.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )
            if stats.cpu_time != 0
            or any(metrics.cpu_time != 0 for metrics in stats.cost_kinds.values())
            or any(
                metrics.cpu_time != 0
                for file_stats in stats.files.values()
                for metrics in file_stats.functions.values()
            )
        ][:top],
    }


def _category_renderable(stats: CategoryStats) -> bool:
    # A category may be omitted only when its aggregate AND its entire
    # rendered subtree are zero: signed leaf functions can cancel to a zero
    # category total without being omittable zero rows.
    return stats.cpu_time != 0 or any(
        metrics.cpu_time != 0 for metrics in stats.functions.values()
    )


def _render_boundary_bucket(
    boundary: BoundaryStats,
    label: str,
    categories: Sequence[str],
) -> dict[str, object]:
    total = boundary.total_time
    category_rows = [
        _render_boundary_category(boundary, category, stats)
        for category in categories
        if (stats := boundary.categories.get(category)) is not None
        and _category_renderable(stats)
    ]
    cpu_time = sum(cast(int, row["time_ns"]) for row in category_rows)
    return {
        "name": label,
        "time_ns": cpu_time,
        "pct": (_f64_ratio(cpu_time, total) * 100) if total else 0,
        "attributable_estimates": _scale_attributables(
            boundary.attributables,
            total,
            cpu_time,
        ),
        "samples": sum(cast(int, row["samples"]) for row in category_rows),
        "categories": category_rows,
    }


def _render_boundary_category(
    boundary: BoundaryStats,
    category: str,
    stats: CategoryStats,
) -> dict[str, object]:
    total = boundary.total_time
    return {
        "name": category,
        "time_ns": stats.cpu_time,
        "pct": (_f64_ratio(stats.cpu_time, total) * 100) if total else 0,
        "attributable_estimates": _scale_attributables(
            boundary.attributables,
            total,
            stats.cpu_time,
        ),
        "samples": stats.sample_count,
        "leaf_functions": {
            name: asdict(metrics)
            for name, metrics in sorted(
                stats.functions.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )
        },
        "files": sorted(stats.files),
        "folded_from": dict(
            sorted(
                stats.folded_from.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "caller_leaf_pairs": [
            {
                "pair": f"{caller} -> {leaf}",
                "time_ns": metrics.cpu_time,
                "samples": metrics.count,
                "pct": (_f64_ratio(metrics.cpu_time, total) * 100) if total else 0,
                "attributable_estimates": _scale_attributables(
                    boundary.attributables,
                    total,
                    metrics.cpu_time,
                ),
            }
            for (caller, leaf), metrics in sorted(
                stats.caller_leaf_pairs.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )
        ],
    }


def _render_domain(
    boundary: BoundaryStats,
    name: str,
    *,
    top: int | None,
) -> dict[str, object]:
    stats = boundary.domains[name]
    total = boundary.total_time
    return {
        "name": name,
        "time_ns": stats.cpu_time,
        "pct": (_f64_ratio(stats.cpu_time, total) * 100) if total else 0,
        "attributable_estimates": _scale_attributables(
            boundary.attributables,
            total,
            stats.cpu_time,
        ),
        "samples": stats.sample_count,
        "cost_kinds": [
            {
                "name": cost_kind,
                "time_ns": metrics.cpu_time,
                "samples": metrics.count,
                "pct_of_domain": (
                    _f64_ratio(metrics.cpu_time, stats.cpu_time) * 100
                    if stats.cpu_time
                    else 0
                ),
                "pct_of_boundary": (
                    _f64_ratio(metrics.cpu_time, total) * 100 if total else 0
                ),
                "attributable_estimates": _scale_attributables(
                    boundary.attributables,
                    total,
                    metrics.cpu_time,
                ),
            }
            for cost_kind, metrics in sorted(
                stats.cost_kinds.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )
        ][:top],
        "files": [
            {
                "filename": file_stats.filename,
                "time_ns": file_stats.cpu_time,
                "samples": file_stats.sample_count,
                "pct_of_domain": (
                    _f64_ratio(file_stats.cpu_time, stats.cpu_time) * 100
                    if stats.cpu_time
                    else 0
                ),
                "pct_of_boundary": (
                    _f64_ratio(file_stats.cpu_time, total) * 100 if total else 0
                ),
                "attributable_estimates": _scale_attributables(
                    boundary.attributables,
                    total,
                    file_stats.cpu_time,
                ),
                "functions": {
                    function: asdict(metrics)
                    for function, metrics in sorted(
                        file_stats.functions.items(),
                        key=lambda item: item[1].cpu_time,
                        reverse=True,
                    )[:top]
                },
                "cost_kinds": [
                    {
                        "name": cost_kind,
                        "time_ns": metrics.cpu_time,
                        "samples": metrics.count,
                        "pct_of_file": (
                            _f64_ratio(metrics.cpu_time, file_stats.cpu_time) * 100
                            if file_stats.cpu_time
                            else 0
                        ),
                    }
                    for cost_kind, metrics in sorted(
                        file_stats.cost_kinds.items(),
                        key=lambda item: item[1].cpu_time,
                        reverse=True,
                    )
                ][:top],
                "caller_leaf_pairs": [
                    {
                        "caller": caller,
                        "leaf": leaf,
                        "time_ns": metrics.cpu_time,
                        "samples": metrics.count,
                        "pct_of_file": (
                            _f64_ratio(metrics.cpu_time, file_stats.cpu_time) * 100
                            if file_stats.cpu_time
                            else 0
                        ),
                    }
                    for (caller, leaf), metrics in sorted(
                        file_stats.caller_leaf_pairs.items(),
                        key=lambda item: item[1].cpu_time,
                        reverse=True,
                    )
                ][:top],
            }
            for file_stats in sorted(
                stats.files.values(),
                key=lambda item: item.cpu_time,
                reverse=True,
            )
        ][:top],
    }


def render_semantic_callers_csv(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    dependency_prefix: str = "deps",
    legacy_layout: bool = False,
) -> str:
    if legacy_layout:
        return _render_legacy_semantic_callers_csv(
            results,
            runtime_rules=runtime_rules,
            dependency_prefix=dependency_prefix,
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "Parent Function",
            "Category",
            "Leaf Function",
            "Leaf Samples",
            "Top Caller",
            "Caller Samples",
            "Caller File",
        ]
    )
    for parent, categories in results.items():
        for category, stats in categories.items():
            for leaf, metrics in sorted(
                stats.semantic_callers.items(),
                key=lambda item: item[1].count,
                reverse=True,
            ):
                top_caller, caller_samples = max(
                    metrics.caller_names.items(),
                    key=lambda item: item[1],
                    default=("", 0),
                )
                caller_file, _ = max(
                    metrics.caller_files.items(),
                    key=lambda item: item[1],
                    default=("", 0),
                )
                writer.writerow(
                    [
                        parent,
                        category,
                        leaf,
                        metrics.count,
                        top_caller,
                        caller_samples,
                        _shorten_semantic_caller_file(
                            caller_file,
                            runtime_rules,
                            dependency_prefix=dependency_prefix,
                        ),
                    ]
                )
    return output.getvalue().rstrip("\r\n")


def _render_legacy_semantic_callers_csv(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    runtime_rules: RuntimeRuleSet,
    dependency_prefix: str,
) -> str:
    lines = [
        "Parent Function,Category,Leaf Function,Leaf Samples,Top Caller,"
        "Caller Samples,Caller File"
    ]
    for parent, categories in results.items():
        for category, stats in categories.items():
            for leaf, metrics in stats.semantic_callers.items():
                if metrics.count == 0:
                    continue
                top_caller, caller_samples = max(
                    metrics.caller_names.items(),
                    key=lambda item: item[1],
                    default=("", 0),
                )
                caller_file, _ = max(
                    metrics.caller_files.items(),
                    key=lambda item: item[1],
                    default=("", 0),
                )
                shortened_file = _shorten_semantic_caller_file(
                    caller_file,
                    runtime_rules,
                    dependency_prefix=dependency_prefix,
                )
                lines.append(
                    f"{_quote_legacy_csv(parent)},{_quote_legacy_csv(category)},"
                    f"{_quote_legacy_csv(leaf)},{metrics.count},"
                    f"{_quote_legacy_csv(top_caller)},{caller_samples},"
                    f"{_quote_legacy_csv(shortened_file)}"
                )
    return "\n".join(lines)


def _shorten_semantic_caller_file(
    file_path: str,
    runtime_rules: RuntimeRuleSet,
    *,
    dependency_prefix: str,
) -> str:
    library_path = extract_library_path(file_path, runtime_rules)
    if library_path is not None:
        return f"{dependency_prefix}/{library_path.relative_path}"
    return file_path


def render_target_csv(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    attributables: Mapping[str, Mapping[str, float]] | None = None,
    simplified: bool = False,
    legacy_layout: bool = False,
) -> str:
    if legacy_layout:
        return _render_legacy_target_csv(
            results,
            attributables=attributables,
            simplified=simplified,
        )

    output = io.StringIO()
    attributable_columns = sorted(attributables.keys()) if attributables else []
    if simplified:
        header = ["Parent Function", "Category", "CPU %"]
        header.extend(attributable_columns)
        header.extend(["Top 3 Callsites", "Top Leaf Functions"])
    else:
        header = [
            "Parent Function",
            "Category",
            "CPU Time (ns)",
            "CPU Time",
            "%",
        ]
        header.extend(attributable_columns)
        header.extend(
            [
                "Samples",
                "Leaf Functions",
                "Files",
                "Top 3 Callsites",
                "Top Leaf Functions",
                "Top Caller->Leaf Pair",
                "Rank-2 Caller->Leaf Pair",
                "Rank-3 Caller->Leaf Pair",
            ]
        )
    writer = csv.writer(output)
    writer.writerow(header)
    for parent, categories in results.items():
        total = sum(stats.cpu_time for stats in categories.values())
        for category, stats in sorted(
            categories.items(), key=lambda item: item[1].cpu_time, reverse=True
        ):
            pct = (_f64_ratio(stats.cpu_time, total) * 100) if total else 0
            if (
                simplified
                and category != "Other"
                and stats.cpu_time >= 0
                and abs(pct) < 0.1
                and (total != 0 or stats.cpu_time == 0)
            ):
                # The noise gate only hides small nonnegative shares: negative
                # rows are signed data that must render at any magnitude, and
                # a zero parent total may omit only exactly-zero rows.
                continue
            top_functions = sorted(
                stats.functions.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[: 3 if simplified else 5]
            function_summary = "; ".join(
                f"{name} ({_pct_of(metrics.cpu_time, total):.1f}%)"
                for name, metrics in top_functions
            )
            caller_totals: dict[str, int] = {}
            for (caller, _leaf), metrics in stats.caller_leaf_pairs.items():
                caller_totals[caller] = caller_totals.get(caller, 0) + metrics.cpu_time
            top_callers = sorted(
                caller_totals.items(), key=lambda item: item[1], reverse=True
            )[:3]
            callsites_summary = "; ".join(
                f"{caller} ({_pct_of(time_ns, total):.1f}%)"
                for caller, time_ns in top_callers
            )
            attributable_values = [
                (
                    f"{_finite_attributable(column, (pct / 100.0) * attributables[column][parent]):.1f}"
                    if attributables
                    and column in attributables
                    and parent in attributables[column]
                    else "N/A"
                )
                for column in attributable_columns
            ]
            if simplified:
                writer.writerow(
                    [
                        parent,
                        category,
                        f"{pct:.1f}",
                        *attributable_values,
                        callsites_summary,
                        function_summary,
                    ]
                )
                continue
            top_pairs = sorted(
                stats.caller_leaf_pairs.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[:3]
            pair_columns = [
                f"{caller} -> {leaf} "
                f"({metrics.count} samples, {_pct_of(metrics.cpu_time, total):.1f}%)"
                for (caller, leaf), metrics in top_pairs
            ]
            while len(pair_columns) < 3:
                pair_columns.append("")
            writer.writerow(
                [
                    parent,
                    category,
                    stats.cpu_time,
                    format_time(stats.cpu_time),
                    f"{pct:.2f}",
                    *attributable_values,
                    stats.sample_count,
                    len(stats.functions),
                    len(stats.files),
                    callsites_summary,
                    function_summary,
                    *pair_columns,
                ]
            )
    return output.getvalue().rstrip("\r\n")


def _quote_legacy_csv(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _render_legacy_target_csv(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    attributables: Mapping[str, Mapping[str, float]] | None = None,
    simplified: bool = False,
) -> str:
    attributable_columns = sorted(attributables.keys()) if attributables else []
    if simplified:
        header = "Parent Function,Category,CPU %"
        if attributable_columns:
            header += "," + ",".join(attributable_columns)
        header += ",Top 3 Callsites,Top Leaf Functions"
    else:
        header = "Parent Function,Category,CPU Time (ns),CPU Time,%"
        if attributable_columns:
            header += "," + ",".join(attributable_columns)
        header += (
            ",Samples,Leaf Functions,Files,Top 3 Callsites,Top Leaf Functions,"
            "Top Caller\u2192Leaf Pair,Rank-2 Caller\u2192Leaf Pair,"
            "Rank-3 Caller\u2192Leaf Pair"
        )

    lines = [header]
    for parent, categories in results.items():
        total = sum(stats.cpu_time for stats in categories.values())
        for category, stats in sorted(
            categories.items(), key=lambda item: item[1].cpu_time, reverse=True
        ):
            pct = (_f64_ratio(stats.cpu_time, total) * 100) if total else 0
            if (
                simplified
                and category != "Other"
                and stats.cpu_time >= 0
                and abs(pct) < 0.1
                and (total != 0 or stats.cpu_time == 0)
            ):
                # The noise gate only hides small nonnegative shares: negative
                # rows are signed data that must render at any magnitude, and
                # a zero parent total may omit only exactly-zero rows.
                continue

            top_functions = sorted(
                stats.functions.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[: 3 if simplified else 5]
            if simplified:
                function_summary = "; ".join(
                    f"{name} ({_pct_of(metrics.cpu_time, total):.1f}%)"
                    for name, metrics in top_functions
                )
            else:
                function_summary = "; ".join(
                    f"{name} ({metrics.count} samples, {_pct_of(metrics.cpu_time, total):.1f}%)"
                    for name, metrics in top_functions
                )

            caller_totals: dict[str, int] = {}
            for (caller, _leaf), metrics in stats.caller_leaf_pairs.items():
                caller_totals[caller] = caller_totals.get(caller, 0) + metrics.cpu_time
            top_callers = sorted(
                caller_totals.items(), key=lambda item: item[1], reverse=True
            )[:3]
            callsite_separator = "; " if simplified else ", "
            callsites_summary = callsite_separator.join(
                f"{caller} ({_pct_of(time_ns, total):.1f}%)"
                for caller, time_ns in top_callers
            )

            attributable_values = [
                (
                    f"{_finite_attributable(column, (pct / 100.0) * attributables[column][parent]):.1f}"
                    if attributables
                    and column in attributables
                    and parent in attributables[column]
                    else "N/A"
                )
                for column in attributable_columns
            ]

            if simplified:
                row = (
                    f"{_quote_legacy_csv(parent)},{_quote_legacy_csv(category)},"
                    f"{pct:.1f}"
                )
                if attributable_values:
                    row += "," + ",".join(attributable_values)
                row += (
                    f",{_quote_legacy_csv(callsites_summary)},"
                    f"{_quote_legacy_csv(function_summary)}"
                )
                lines.append(row)
                continue

            top_pairs = sorted(
                stats.caller_leaf_pairs.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[:3]
            pair_columns = [
                _quote_legacy_csv(
                    f"{caller} \u2192 {leaf} "
                    f"({metrics.count} samples, {_pct_of(metrics.cpu_time, total):.1f}%)"
                )
                for (caller, leaf), metrics in top_pairs
            ]
            while len(pair_columns) < 3:
                pair_columns.append('""')

            row = (
                f"{_quote_legacy_csv(parent)},{_quote_legacy_csv(category)},"
                f"{stats.cpu_time},{_quote_legacy_csv(format_time(stats.cpu_time))},"
                f"{pct:.2f}"
            )
            if attributable_values:
                row += "," + ",".join(attributable_values)
            row += (
                f",{stats.sample_count},{len(stats.functions)},{len(stats.files)},"
                f"{_quote_legacy_csv(callsites_summary)},"
                f"{_quote_legacy_csv(function_summary)},"
                f"{','.join(pair_columns)}"
            )
            lines.append(row)

    return "\n".join(lines)


def render_target_text(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    show_folded: bool = False,
    show_semantic_callers: bool = False,
) -> str:
    lines: list[str] = []
    for parent, categories in results.items():
        total = sum(stats.cpu_time for stats in categories.values())
        lines.append("=" * 100)
        lines.append(f"Parent Function: {parent}")
        lines.append("=" * 100)
        lines.append(f"Total CPU time under this function: {format_time(total)}")
        if show_folded:
            folded_total = sum(
                sum(stats.folded_from.values()) for stats in categories.values()
            )
            if folded_total:
                lines.append(
                    "Total runtime internals folded into categories: "
                    f"{format_time(folded_total)}"
                )
        lines.append("")
        lines.append(
            f"{'Category':<35} {'CPU Time':<15} {'%':<8} {'Samples':<10} {'Leaf Functions':<15} {'Files':<8}"
        )
        lines.append("-" * 110)
        for category, stats in sorted(
            categories.items(), key=lambda item: item[1].cpu_time, reverse=True
        ):
            pct = (_f64_ratio(stats.cpu_time, total) * 100) if total else 0
            lines.append(
                f"{category:<35} {format_time(stats.cpu_time):<15} {pct:>6.2f}% "
                f"{stats.sample_count:>9} {len(stats.functions):>14} {len(stats.files):>7}"
            )
        lines.append("-" * 110)
        lines.append(
            f"{'TOTAL':<35} {format_time(total):<15} "
            f"{'100.00%' if total else '0.00%':>8}"
        )
        if show_folded:
            folded_categories = [
                (category, stats)
                for category, stats in sorted(
                    categories.items(),
                    key=lambda item: item[1].cpu_time,
                    reverse=True,
                )
                if stats.folded_from
            ]
            if folded_categories:
                lines.append("")
                lines.append("Runtime internals folded into categories:")
                for category, stats in folded_categories[:5]:
                    lines.append(f"  {category}:")
                    for function, time_ns in sorted(
                        stats.folded_from.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:10]:
                        lines.append(f"    - {function}: {format_time(time_ns)}")
        if show_semantic_callers:
            semantic_categories = [
                (category, stats)
                for category, stats in sorted(
                    categories.items(),
                    key=lambda item: item[1].cpu_time,
                    reverse=True,
                )
                if stats.semantic_callers
            ]
            if semantic_categories:
                lines.append("")
                lines.append("Semantic callers for runtime internals:")
                for category, stats in semantic_categories[:5]:
                    lines.append(f"  {category}:")
                    for leaf, metrics in sorted(
                        stats.semantic_callers.items(),
                        key=lambda item: item[1].count,
                        reverse=True,
                    )[:5]:
                        lines.append(f"    {leaf} ({metrics.count} samples):")
                        top_callers = sorted(
                            metrics.caller_names.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:3]
                        for caller, count in top_callers:
                            lines.append(f"      - {caller} ({count} samples)")
    return "\n".join(lines)


_BY_SLICE_INT_MESSAGE = "--by-slice values must be integers."
_BY_SLICE_THRESHOLD_MESSAGE = "--by-slice percentage thresholds must be finite numbers."


def strict_int64(raw: str, *, message: str) -> int:
    """Strict shared grammar with the Rust core (`i64::from_str`): optional
    sign, ASCII digits, int64 range — bare int() would also accept
    whitespace, underscores, and non-ASCII digits."""
    if re.fullmatch(r"[+-]?\d+", raw, re.ASCII) is None:
        raise ValueError(message)
    value = int(raw)
    if not -(2**63) <= value <= 2**63 - 1:
        raise ValueError(message)
    return value


def _by_slice_limit(raw: str) -> int:
    return strict_int64(raw, message=_BY_SLICE_INT_MESSAGE)


def strict_float(raw: str, *, message: str) -> float:
    """Strict shared grammar with the Rust core (`f64::from_str` plus a
    finiteness check): no surrounding whitespace, underscores, or non-ASCII
    digits; overflowing literals parse to infinity and are rejected."""
    if raw != raw.strip() or "_" in raw or not raw.isascii():
        raise ValueError(message)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(value):
        raise ValueError(message)
    return value


def _by_slice_threshold(raw: str) -> float:
    return strict_float(raw, message=_BY_SLICE_THRESHOLD_MESSAGE)


def render_slice_json(
    result: SliceAnalysisResult,
    options: SliceAnalysisOptions | None = None,
) -> dict[str, object]:
    resolved_options = options or SliceAnalysisOptions()
    # Percentages are shares of the filtered matching total; when matched
    # signed samples cancel to exactly zero the zero arms below render 0 —
    # never a silent fallback to the unrelated whole-profile total.
    total = result.matching_time_ns
    library_limit = (
        resolved_options.unattributed_libraries
        if resolved_options.unattributed_libraries is not None
        else resolved_options.unattributed_gems
    )
    legacy_library_limit = (
        resolved_options.unattributed_gems
        if resolved_options.unattributed_gems is not None
        else resolved_options.unattributed_libraries
    )
    selected_slices = list(result.slices)
    selected_slices = [
        item for item in selected_slices if item.name not in {"(gc)", "(uncollapsible)"}
    ]
    metadata_by_slice = {
        item.name: dict(item.metadata)
        for item in resolved_options.slices
        if item.metadata
    }
    if resolved_options.by_slice is not None:
        by_slice = resolved_options.by_slice
        if by_slice.endswith("%"):
            threshold = _by_slice_threshold(by_slice.removesuffix("%"))
            # At a zero matching total every rendered percentage is 0, so
            # threshold selection compares against 0 rather than deleting
            # every signed row through a nonzero-total short-circuit.
            selected_slices = [
                item
                for item in selected_slices
                if (_f64_ratio(item.time_ns, total) * 100 if total else 0.0)
                >= threshold
            ]
        else:
            selected_slices = selected_slices[: _by_slice_limit(by_slice)]

    def slice_payload(slice_item: SliceStats) -> dict[str, object]:
        metadata = metadata_by_slice.get(slice_item.name, {})
        payload_item: dict[str, object] = {
            "name": slice_item.name,
            "time_ns": slice_item.time_ns,
            "pct": (_f64_ratio(slice_item.time_ns, total) * 100) if total else 0,
            "is_default": slice_item.is_default,
            "frames": [
                {
                    "function": frame.function,
                    "filename": frame.filename,
                    "line": frame.line,
                    "time_ns": frame.time_ns,
                    "pct": (_f64_ratio(frame.time_ns, total) * 100) if total else 0,
                }
                for frame in sorted(
                    slice_item.frames.values(),
                    key=lambda frame: frame.time_ns,
                    reverse=True,
                )[: resolved_options.top]
            ],
            "unattributed_gems": [
                {
                    "name": name,
                    "time_ns": time_ns,
                    "pct": (_f64_ratio(time_ns, total) * 100) if total else 0,
                }
                for name, time_ns in sorted(
                    slice_item.unattributed_gems.items(),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:legacy_library_limit]
            ],
            "unattributed_libraries": [
                {
                    "name": name,
                    "time_ns": time_ns,
                    "pct": (_f64_ratio(time_ns, total) * 100) if total else 0,
                }
                for name, time_ns in sorted(
                    slice_item.unattributed_libraries.items(),
                    key=lambda pair: pair[1],
                    reverse=True,
                )[:library_limit]
            ],
        }
        if metadata:
            payload_item["metadata"] = metadata
        return payload_item

    payload: dict[str, object] = {
        "tool": "clankerprof_slices",
        "summary": {
            "matching_time_ns": result.matching_time_ns,
            "total_time_ns": result.total_time_ns,
            "matching_pct": (
                _f64_ratio(result.matching_time_ns, result.total_time_ns) * 100
            )
            if result.total_time_ns
            else 0,
        },
        "slices": [slice_payload(item) for item in selected_slices],
    }
    # Zero-aggregate pseudo-outputs stay omitted; negative aggregates are
    # signed data and must be reported (R2-07 rule, extended by R3-05).
    if result.gc_time_ns != 0:
        payload["gc"] = {
            "time_ns": result.gc_time_ns,
            "pct": (_f64_ratio(result.gc_time_ns, total) * 100) if total else 0,
        }
    if result.uncollapsible is not None:
        payload["uncollapsible"] = {
            "name": result.uncollapsible.name,
            "time_ns": result.uncollapsible.time_ns,
            "pct": (_f64_ratio(result.uncollapsible.time_ns, total) * 100)
            if total
            else 0,
            "frames": [
                {
                    "function": frame.function,
                    "filename": frame.filename,
                    "line": frame.line,
                    "time_ns": frame.time_ns,
                    "pct": (_f64_ratio(frame.time_ns, total) * 100) if total else 0,
                }
                for frame in sorted(
                    result.uncollapsible.frames.values(),
                    key=lambda frame: frame.time_ns,
                    reverse=True,
                )[: resolved_options.top]
            ],
            "unattributed_gems": [],
            "unattributed_libraries": [],
        }
    return payload


def render_json_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
