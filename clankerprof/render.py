from __future__ import annotations

import csv
import io
import json

from collections.abc import Mapping
from dataclasses import asdict

from clankerprof.analysis import (
    DEFAULT_RUNTIME_RULES,
    RuntimeRuleSet,
    SliceAnalysisOptions,
    SliceAnalysisResult,
    SliceStats,
    extract_library_path,
    format_time,
)
from clankerprof.model import CategoryStats


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
                    "pct": (stats.cpu_time / total * 100) if total else 0,
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
            pct = (stats.cpu_time / total * 100) if total else 0
            if simplified and pct < 0.1 and category != "Other":
                continue
            top_functions = sorted(
                stats.functions.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[: 3 if simplified else 5]
            function_summary = "; ".join(
                f"{name} ({metrics.cpu_time / total * 100:.1f}%)"
                for name, metrics in top_functions
            )
            caller_totals: dict[str, int] = {}
            for pair, metrics in stats.caller_leaf_pairs.items():
                caller = pair.split(" -> ", 1)[0]
                caller_totals[caller] = caller_totals.get(caller, 0) + metrics.cpu_time
            top_callers = sorted(
                caller_totals.items(), key=lambda item: item[1], reverse=True
            )[:3]
            callsites_summary = "; ".join(
                f"{caller} ({time_ns / total * 100:.1f}%)"
                for caller, time_ns in top_callers
            )
            attributable_values = [
                (
                    f"{(pct / 100.0) * attributables[column][parent]:.1f}"
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
                f"{pair} ({metrics.count} samples, {metrics.cpu_time / total * 100:.1f}%)"
                for pair, metrics in top_pairs
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
            pct = (stats.cpu_time / total * 100) if total else 0
            if simplified and pct < 0.1 and category != "Other":
                continue

            top_functions = sorted(
                stats.functions.items(),
                key=lambda item: item[1].cpu_time,
                reverse=True,
            )[: 3 if simplified else 5]
            if simplified:
                function_summary = "; ".join(
                    f"{name} ({metrics.cpu_time / total * 100:.1f}%)"
                    for name, metrics in top_functions
                )
            else:
                function_summary = "; ".join(
                    f"{name} ({metrics.count} samples, {metrics.cpu_time / total * 100:.1f}%)"
                    for name, metrics in top_functions
                )

            caller_totals: dict[str, int] = {}
            for pair, metrics in stats.caller_leaf_pairs.items():
                caller = pair.split(" -> ", 1)[0]
                caller_totals[caller] = caller_totals.get(caller, 0) + metrics.cpu_time
            top_callers = sorted(
                caller_totals.items(), key=lambda item: item[1], reverse=True
            )[:3]
            callsite_separator = "; " if simplified else ", "
            callsites_summary = callsite_separator.join(
                f"{caller} ({time_ns / total * 100:.1f}%)"
                for caller, time_ns in top_callers
            )

            attributable_values = [
                (
                    f"{(pct / 100.0) * attributables[column][parent]:.1f}"
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
                    f"{_legacy_pair_arrow(pair)} "
                    f"({metrics.count} samples, {metrics.cpu_time / total * 100:.1f}%)"
                )
                for pair, metrics in top_pairs
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


def _legacy_pair_arrow(pair: str) -> str:
    return pair.replace(" -> ", " \u2192 ")


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
            pct = (stats.cpu_time / total * 100) if total else 0
            lines.append(
                f"{category:<35} {format_time(stats.cpu_time):<15} {pct:>6.2f}% "
                f"{stats.sample_count:>9} {len(stats.functions):>14} {len(stats.files):>7}"
            )
        lines.append("-" * 110)
        lines.append(f"{'TOTAL':<35} {format_time(total):<15} {'100.00%':>8}")
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


def render_slice_json(
    result: SliceAnalysisResult,
    options: SliceAnalysisOptions | None = None,
) -> dict[str, object]:
    resolved_options = options or SliceAnalysisOptions()
    total = result.matching_time_ns or result.total_time_ns
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
    if resolved_options.by_slice:
        by_slice = resolved_options.by_slice
        if by_slice.endswith("%"):
            threshold = float(by_slice.removesuffix("%"))
            selected_slices = [
                item
                for item in selected_slices
                if total and item.time_ns / total * 100 >= threshold
            ]
        else:
            selected_slices = selected_slices[: int(by_slice)]

    def slice_payload(slice_item: SliceStats) -> dict[str, object]:
        metadata = metadata_by_slice.get(slice_item.name, {})
        payload_item: dict[str, object] = {
            "name": slice_item.name,
            "time_ns": slice_item.time_ns,
            "pct": (slice_item.time_ns / total * 100) if total else 0,
            "is_default": slice_item.is_default,
            "frames": [
                {
                    "function": frame.function,
                    "filename": frame.filename,
                    "line": frame.line,
                    "time_ns": frame.time_ns,
                    "pct": (frame.time_ns / total * 100) if total else 0,
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
                    "pct": (time_ns / total * 100) if total else 0,
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
                    "pct": (time_ns / total * 100) if total else 0,
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
            "matching_pct": (result.matching_time_ns / result.total_time_ns * 100)
            if result.total_time_ns
            else 0,
        },
        "slices": [slice_payload(item) for item in selected_slices],
    }
    if result.gc_time_ns > 0:
        payload["gc"] = {
            "time_ns": result.gc_time_ns,
            "pct": (result.gc_time_ns / total * 100) if total else 0,
        }
    if result.uncollapsible is not None:
        payload["uncollapsible"] = {
            "name": result.uncollapsible.name,
            "time_ns": result.uncollapsible.time_ns,
            "pct": (result.uncollapsible.time_ns / total * 100) if total else 0,
            "frames": [
                {
                    "function": frame.function,
                    "filename": frame.filename,
                    "line": frame.line,
                    "time_ns": frame.time_ns,
                    "pct": (frame.time_ns / total * 100) if total else 0,
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
    return json.dumps(payload, indent=2, sort_keys=True)
