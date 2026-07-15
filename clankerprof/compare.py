from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any, cast

from clankerprof.model import AGGREGATE_MAX, AGGREGATE_MIN


@dataclass(frozen=True, slots=True)
class CompareOptions:
    threshold_abs: float = 2.0
    threshold_rel: float = 15.0
    focus_slices: frozenset[str] = frozenset()
    focus_boundaries: frozenset[str] = frozenset()


def _finite_or_none(value: float) -> float | None:
    """Compare artifacts are strict JSON: non-finite ratios serialize as null."""
    return value if math.isfinite(value) else None


def _validated_options(options: CompareOptions | None) -> CompareOptions:
    resolved = options or CompareOptions()
    if (
        not (
            math.isfinite(resolved.threshold_abs)
            and math.isfinite(resolved.threshold_rel)
        )
        or resolved.threshold_abs < 0
        or resolved.threshold_rel < 0
    ):
        # A non-finite threshold would silently disable gating; a negative one
        # would gate identical reports as regressions (0 > -1).
        raise ValueError("Compare thresholds must be finite, non-negative numbers.")
    return resolved


def _finite_value(value: float, name: str) -> float:
    """Derived compare values (frame-percentage sums and absolute deltas) must
    stay finite; overflow fails closed rather than serializing an uncontracted
    null. Only ``delta_rel`` has a documented null path."""
    if not math.isfinite(value):
        raise ValueError(f"Compare values for '{name}' are not finite.")
    return value


def _require_number(payload: dict[str, Any], key: str, context: str) -> float:
    """Rows present in a report must carry their numeric fields; absent or
    non-numeric is malformed input. Row-level absence (a name missing from one
    report entirely) is handled by the callers, never by defaulting here."""
    value: object = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} field {key!r} must be a number.")
    return float(value)


def _require_name(payload: dict[str, Any], key: str, context: str) -> str:
    value: object = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context} rows must carry a string {key!r}.")
    return value


def _optional_rows(
    payload: dict[str, Any], key: str, context: str
) -> list[dict[str, Any]]:
    """Structural row arrays may be absent, but a present key must be an array
    of objects — wrong shapes (including a present null) are malformed input,
    never silently empty. Conflating null with absence would let a nulled-out
    array turn a real regression into an apparent removal."""
    if key not in payload:
        return []
    raw: object = payload[key]
    if not isinstance(raw, list):
        raise ValueError(f"{context} field {key!r} must be an array.")
    rows: list[dict[str, Any]] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise ValueError(f"{_ROW_CONTEXTS[key]} rows must be objects.")
        rows.append(cast(dict[str, Any], item))
    return rows


_ROW_CONTEXTS = {
    "slices": "Slice",
    "frames": "Frame",
    "boundaries": "Boundary",
    "buckets": "Bucket",
    "categories": "Category",
    "domains": "Domain",
}


def _summary_total(payload: dict[str, Any]) -> int:
    raw_summary: object = payload.get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError("Report summary must be an object.")
    value: object = cast(dict[str, Any], raw_summary).get("total_time_ns")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not AGGREGATE_MIN <= value <= AGGREGATE_MAX
    ):
        # Absent and out-of-range values share the message because Rust's JSON
        # parser cannot distinguish out-of-range integers from non-integers.
        raise ValueError("Report summary field 'total_time_ns' must be an integer.")
    return value


def _slice_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_slices: object = payload.get("slices")
    if not isinstance(raw_slices, list):
        raise ValueError("Profile comparison input must contain a slices array.")
    result: dict[str, dict[str, Any]] = {}
    for item in cast(list[object], raw_slices):
        if not isinstance(item, dict):
            raise ValueError("Slice rows must be objects.")
        raw_item = cast(dict[str, Any], item)
        name = _require_name(raw_item, "name", "Slice")
        if name in result:
            # Projections never emit duplicate top-level names; a duplicate is
            # malformed input and last-wins would make the gate order-dependent.
            raise ValueError(f"Duplicate Slice row '{name}' in comparison input.")
        result[name] = raw_item
    return result


def _frames_by_function(slice_payload: dict[str, Any]) -> dict[str, float]:
    frames: dict[str, float] = {}
    for raw_item in _optional_rows(slice_payload, "frames", "Slice"):
        function = _require_name(raw_item, "function", "Frame")
        frames[function] = frames.get(function, 0.0) + _require_number(
            raw_item, "pct", "Frame"
        )
    return frames


def _delta_rel(before_pct: float, after_pct: float) -> float:
    """Relative delta against the magnitude of the baseline.

    Sample values are signed, so rows can carry negative percentages; dividing
    by ``abs(before_pct)`` keeps the sign of the change meaningful (-10% ->
    -5% is a +50% increase and must be gateable). A zero baseline yields an
    unbounded delta in the direction of the absolute change, serialized as
    null by the finite-JSON path. Positive baselines are bit-identical to the
    plain ``delta/before`` form.
    """
    delta_abs = after_pct - before_pct
    if before_pct != 0:
        return (delta_abs / abs(before_pct)) * 100
    if delta_abs > 0:
        return float("inf")
    if delta_abs < 0:
        return float("-inf")
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
    resolved = _validated_options(options)
    before_slices = _slice_map(before)
    after_slices = _slice_map(after)
    names = sorted(set(before_slices) | set(after_slices))
    slice_deltas: list[dict[str, Any]] = []
    frame_deltas_all: list[dict[str, Any]] = []
    has_regression = False

    for name in names:
        before_payload = before_slices.get(name)
        after_payload = after_slices.get(name)
        before_pct = (
            0.0
            if before_payload is None
            else _require_number(before_payload, "pct", "Slice")
        )
        after_pct = (
            0.0
            if after_payload is None
            else _require_number(after_payload, "pct", "Slice")
        )
        delta_abs = _finite_value(after_pct - before_pct, name)
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

        before_frames = (
            {} if before_payload is None else _frames_by_function(before_payload)
        )
        after_frames = (
            {} if after_payload is None else _frames_by_function(after_payload)
        )
        # Finiteness is checked while walking the sorted union so both
        # languages report the same first offender when several overflow.
        frame_deltas: list[dict[str, Any]] = []
        for function in sorted(set(before_frames) | set(after_frames)):
            before_frame_pct = _finite_value(
                before_frames.get(function, 0.0), function
            )
            after_frame_pct = _finite_value(after_frames.get(function, 0.0), function)
            frame_deltas.append(
                {
                    "function": function,
                    "slice": name,
                    "before_pct": before_frame_pct,
                    "after_pct": after_frame_pct,
                    "delta_abs": _finite_value(
                        after_frame_pct - before_frame_pct, function
                    ),
                }
            )
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
                "delta_rel": _finite_or_none(delta_rel),
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
        "before_total_ns": _summary_total(before),
        "after_total_ns": _summary_total(after),
        "slices": slice_deltas,
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
        "has_regression": has_regression,
    }


def _insert_row(
    rows: dict[tuple[str, str, str], float],
    key: tuple[str, str, str],
    value: float,
    context: str,
) -> None:
    if key in rows:
        # Projections never emit duplicate row names; a duplicate is malformed
        # input and last-wins would make the gate order-dependent.
        raise ValueError(f"Duplicate {context} row '{key[2]}' in comparison input.")
    rows[key] = value


def _boundary_rows(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    raw_boundaries: object = payload.get("boundaries")
    if not isinstance(raw_boundaries, list):
        raise ValueError("Boundary comparison input must contain a boundaries array.")
    rows: dict[tuple[str, str, str], float] = {}
    for item in cast(list[object], raw_boundaries):
        if not isinstance(item, dict):
            raise ValueError("Boundary rows must be objects.")
        boundary = cast(dict[str, Any], item)
        boundary_name = _require_name(boundary, "name", "Boundary")
        _insert_row(
            rows,
            ("boundary", boundary_name, boundary_name),
            _require_number(boundary, "pct_of_profile", "Boundary"),
            "Boundary",
        )
        for bucket in _optional_rows(boundary, "buckets", "Boundary"):
            bucket_name = _require_name(bucket, "name", "Bucket")
            _insert_row(
                rows,
                ("bucket", boundary_name, bucket_name),
                _require_number(bucket, "pct", "Bucket"),
                "Bucket",
            )
            for category in _optional_rows(bucket, "categories", "Bucket"):
                category_name = _require_name(category, "name", "Category")
                _insert_row(
                    rows,
                    ("category", boundary_name, category_name),
                    _require_number(category, "pct", "Category"),
                    "Category",
                )
        for domain in _optional_rows(boundary, "domains", "Boundary"):
            domain_name = _require_name(domain, "name", "Domain")
            _insert_row(
                rows,
                ("domain", boundary_name, domain_name),
                _require_number(domain, "pct", "Domain"),
                "Domain",
            )
    return rows


def compare_boundary_json(
    before: dict[str, Any],
    after: dict[str, Any],
    options: CompareOptions | None = None,
) -> dict[str, Any]:
    resolved = _validated_options(options)
    before_rows = _boundary_rows(before)
    after_rows = _boundary_rows(after)
    row_deltas: list[dict[str, Any]] = []
    has_regression = False

    for kind, boundary, name in sorted(set(before_rows) | set(after_rows)):
        before_pct = before_rows.get((kind, boundary, name), 0.0)
        after_pct = after_rows.get((kind, boundary, name), 0.0)
        delta_abs = _finite_value(after_pct - before_pct, name)
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
                "delta_rel": _finite_or_none(delta_rel),
                "status": status,
            }
        )

    row_deltas.sort(key=lambda item: abs(cast(float, item["delta_abs"])), reverse=True)
    return {
        "tool": "clankerprof_compare",
        "projection": "boundaries",
        "before_total_ns": _summary_total(before),
        "after_total_ns": _summary_total(after),
        "rows": row_deltas,
        "top_regressions": [
            item for item in row_deltas if cast(float, item["delta_abs"]) > 0.1
        ][:10],
        "top_improvements": sorted(
            (item for item in row_deltas if cast(float, item["delta_abs"]) < -0.1),
            key=lambda item: cast(float, item["delta_abs"]),
        )[:10],
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
    if before_tool == "clankerprof_slices":
        return compare_slice_json(before, after, options)
    raise ValueError(
        "Compare inputs must be clankerprof_slices or clankerprof_boundaries "
        f"reports; got tool {before_tool!r}."
    )
