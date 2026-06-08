from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class CompareOptions:
    threshold_abs: float = 2.0
    threshold_rel: float = 15.0
    focus_slices: frozenset[str] = frozenset()


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
        if before_pct > 0:
            delta_rel = (delta_abs / before_pct) * 100
        elif after_pct > 0:
            delta_rel = float("inf")
        else:
            delta_rel = 0.0

        in_focus = not resolved.focus_slices or name in resolved.focus_slices
        is_regression = (
            in_focus
            and delta_abs > resolved.threshold_abs
            and delta_rel > resolved.threshold_rel
        )
        is_improvement = (
            delta_abs < -resolved.threshold_abs and delta_rel < -resolved.threshold_rel
        )
        if is_regression:
            has_regression = True
        status = (
            "regression"
            if is_regression
            else "improvement"
            if is_improvement
            else "stable"
        )

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
