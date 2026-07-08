from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class CompareOptions:
    threshold_abs: float = 2.0
    threshold_rel: float = 15.0
    focus_slices: frozenset[str] = frozenset()
    focus_boundaries: frozenset[str] = frozenset()


def _slice_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slices = payload.get("slices", [])
    if not isinstance(raw_slices, list):
        raise ValueError("Profile comparison input must contain a slices array.")
    result: dict[str, dict[str, Any]] = {}
    for item in cast(list[object], raw_slices):
        if isinstance(item, dict):
            raw_item = cast(dict[str, Any], item)
            result[str(raw_item.get("name", ""))] = raw_item
    return result


def _frames_by_function(slice_payload: dict[str, Any]) -> dict[str, float]:
    raw_frames = slice_payload.get("frames", [])
    if not isinstance(raw_frames, list):
        return {}
    frames: dict[str, float] = {}
    for item in cast(list[object], raw_frames):
        if isinstance(item, dict):
            raw_item = cast(dict[str, Any], item)
            frames[str(raw_item.get("function", ""))] = float(raw_item.get("pct", 0))
    return frames


def _delta_rel(before_pct: float, after_pct: float) -> float:
    delta_abs = after_pct - before_pct
    if before_pct > 0:
        return (delta_abs / before_pct) * 100
    if after_pct > 0:
        return float("inf")
    return 0.0


def _status(
    *,
    before_pct: float,
    after_pct: float,
    threshold_abs: float,
    threshold_rel: float,
    in_focus: bool,
) -> tuple[str, bool]:
    delta_abs = after_pct - before_pct
    delta_rel = _delta_rel(before_pct, after_pct)
    is_regression = in_focus and delta_abs > threshold_abs and delta_rel > threshold_rel
    is_improvement = delta_abs < -threshold_abs and delta_rel < -threshold_rel
    status = (
        "regression" if is_regression else "improvement" if is_improvement else "stable"
    )
    return status, is_regression


def compare_slice_json(
    before: dict[str, Any],
    after: dict[str, Any],
    options: CompareOptions | None = None,
) -> dict[str, Any]:
    resolved = options or CompareOptions()
    before_slices = _slice_map(before)
    after_slices = _slice_map(after)
    names = sorted(set(before_slices) | set(after_slices))
    slice_deltas: list[dict[str, Any]] = []
    frame_deltas_all: list[dict[str, Any]] = []
    has_regression = False

    for name in names:
        before_payload = before_slices.get(name, {})
        after_payload = after_slices.get(name, {})
        before_pct = float(before_payload.get("pct", 0))
        after_pct = float(after_payload.get("pct", 0))
        delta_abs = after_pct - before_pct
        delta_rel = _delta_rel(before_pct, after_pct)

        in_focus = not resolved.focus_slices or name in resolved.focus_slices
        status, is_regression = _status(
            before_pct=before_pct,
            after_pct=after_pct,
            threshold_abs=resolved.threshold_abs,
            threshold_rel=resolved.threshold_rel,
            in_focus=in_focus,
        )
        if is_regression:
            has_regression = True

        before_frames = _frames_by_function(before_payload)
        after_frames = _frames_by_function(after_payload)
        frame_deltas = [
            {
                "function": function,
                "slice": name,
                "before_pct": before_frames.get(function, 0.0),
                "after_pct": after_frames.get(function, 0.0),
                "delta_abs": after_frames.get(function, 0.0)
                - before_frames.get(function, 0.0),
            }
            for function in sorted(set(before_frames) | set(after_frames))
        ]
        frame_deltas.sort(
            key=lambda item: abs(cast(float, item["delta_abs"])), reverse=True
        )
        frame_deltas_all.extend(frame_deltas)
        slice_deltas.append(
            {
                "name": name,
                "before_pct": before_pct,
                "after_pct": after_pct,
                "delta_abs": delta_abs,
                "delta_rel": delta_rel,
                "status": status,
                "frame_deltas": frame_deltas,
            }
        )

    slice_deltas.sort(
        key=lambda item: abs(cast(float, item["delta_abs"])), reverse=True
    )
    frame_deltas_all.sort(key=lambda item: cast(float, item["delta_abs"]), reverse=True)
    top_regressions = [
        item for item in frame_deltas_all if cast(float, item["delta_abs"]) > 0.1
    ][:10]
    top_improvements = [
        item
        for item in reversed(frame_deltas_all)
        if cast(float, item["delta_abs"]) < -0.1
    ][:10]

    return {
        "tool": "clankerprof_compare",
        "before_total_ns": int(
            cast(dict[str, Any], before.get("summary", {})).get("total_time_ns", 0)
        ),
        "after_total_ns": int(
            cast(dict[str, Any], after.get("summary", {})).get("total_time_ns", 0)
        ),
        "slices": slice_deltas,
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
        "has_regression": has_regression,
    }


def _boundary_rows(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    raw_boundaries = payload.get("boundaries", [])
    if not isinstance(raw_boundaries, list):
        raise ValueError("Boundary comparison input must contain a boundaries array.")
    rows: dict[tuple[str, str, str], float] = {}
    for item in cast(list[object], raw_boundaries):
        if not isinstance(item, dict):
            continue
        boundary = cast(dict[str, Any], item)
        boundary_name = str(boundary.get("name", ""))
        rows[("boundary", boundary_name, boundary_name)] = float(
            boundary.get("pct_of_profile", 0)
        )
        raw_buckets = boundary.get("buckets", [])
        if isinstance(raw_buckets, list):
            for bucket_item in cast(list[object], raw_buckets):
                if not isinstance(bucket_item, dict):
                    continue
                bucket = cast(dict[str, Any], bucket_item)
                bucket_name = str(bucket.get("name", ""))
                rows[("bucket", boundary_name, bucket_name)] = float(
                    bucket.get("pct", 0)
                )
                raw_categories = bucket.get("categories", [])
                if isinstance(raw_categories, list):
                    for category_item in cast(list[object], raw_categories):
                        if not isinstance(category_item, dict):
                            continue
                        category = cast(dict[str, Any], category_item)
                        category_name = str(category.get("name", ""))
                        rows[("category", boundary_name, category_name)] = float(
                            category.get("pct", 0)
                        )
        raw_domains = boundary.get("domains", [])
        if isinstance(raw_domains, list):
            for domain_item in cast(list[object], raw_domains):
                if not isinstance(domain_item, dict):
                    continue
                domain = cast(dict[str, Any], domain_item)
                domain_name = str(domain.get("name", ""))
                rows[("domain", boundary_name, domain_name)] = float(
                    domain.get("pct", 0)
                )
    return rows


def compare_boundary_json(
    before: dict[str, Any],
    after: dict[str, Any],
    options: CompareOptions | None = None,
) -> dict[str, Any]:
    resolved = options or CompareOptions()
    before_rows = _boundary_rows(before)
    after_rows = _boundary_rows(after)
    row_deltas: list[dict[str, Any]] = []
    has_regression = False

    for kind, boundary, name in sorted(set(before_rows) | set(after_rows)):
        before_pct = before_rows.get((kind, boundary, name), 0.0)
        after_pct = after_rows.get((kind, boundary, name), 0.0)
        delta_abs = after_pct - before_pct
        delta_rel = _delta_rel(before_pct, after_pct)
        in_focus = (
            not resolved.focus_boundaries or boundary in resolved.focus_boundaries
        )
        status, is_regression = _status(
            before_pct=before_pct,
            after_pct=after_pct,
            threshold_abs=resolved.threshold_abs,
            threshold_rel=resolved.threshold_rel,
            in_focus=in_focus,
        )
        if is_regression:
            has_regression = True
        row_deltas.append(
            {
                "kind": kind,
                "boundary": boundary,
                "name": name,
                "before_pct": before_pct,
                "after_pct": after_pct,
                "delta_abs": delta_abs,
                "delta_rel": delta_rel,
                "status": status,
            }
        )

    row_deltas.sort(key=lambda item: abs(cast(float, item["delta_abs"])), reverse=True)
    return {
        "tool": "clankerprof_compare",
        "projection": "boundaries",
        "before_total_ns": int(
            cast(dict[str, Any], before.get("summary", {})).get("total_time_ns", 0)
        ),
        "after_total_ns": int(
            cast(dict[str, Any], after.get("summary", {})).get("total_time_ns", 0)
        ),
        "rows": row_deltas,
        "top_regressions": [
            item for item in row_deltas if cast(float, item["delta_abs"]) > 0.1
        ][:10],
        "top_improvements": [
            item
            for item in reversed(row_deltas)
            if cast(float, item["delta_abs"]) < -0.1
        ][:10],
        "has_regression": has_regression,
    }


def compare_json(
    before: dict[str, Any],
    after: dict[str, Any],
    options: CompareOptions | None = None,
) -> dict[str, Any]:
    before_tool = before.get("tool")
    after_tool = after.get("tool")
    if before_tool != after_tool:
        raise ValueError("Compare inputs must use the same clankerprof projection.")
    if before_tool == "clankerprof_boundaries":
        return compare_boundary_json(before, after, options)
    return compare_slice_json(before, after, options)
