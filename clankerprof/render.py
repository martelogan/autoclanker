from __future__ import annotations

import csv
import io
import json

from collections.abc import Mapping
from dataclasses import asdict

from clankerprof.analysis import (
    SliceAnalysisOptions,
    SliceAnalysisResult,
    SliceStats,
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
) -> str:
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
                        _shorten_semantic_caller_file(caller_file),
                    ]
                )
    return output.getvalue().rstrip("\r\n")


def _shorten_semantic_caller_file(file_path: str) -> str:
    if "/gems/" in file_path:
        parts = file_path.split("/gems/")
        return f"gems/{parts[-1]}" if len(parts) > 1 else file_path
    return file_path


def render_target_csv(
    results: Mapping[str, Mapping[str, CategoryStats]],
    *,
    attributables: Mapping[str, Mapping[str, float]] | None = None,
    simplified: bool = False,
) -> str:
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
                )[: resolved_options.unattributed_gems]
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
        }
    return payload


def render_json_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
