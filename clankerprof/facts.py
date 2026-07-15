from __future__ import annotations

import json

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

from clankerprof.jsonio import parse_strict_json
from clankerprof.model import (
    Frame,
    ProfileFacts,
    Sample,
    SampleFact,
    TimeNs,
    ValueType,
    check_aggregate_bounds,
)

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
SampleFactsInput: TypeAlias = Iterable[SampleFact] | ProfileFacts
JsonObject: TypeAlias = dict[str, Any]

SAMPLE_FACTS_SCHEMA_VERSION: Final = "clankerprof.sample_facts.v2"
SAMPLE_FACTS_SCHEMA_VERSION_V1: Final = "clankerprof.sample_facts.v1"

_I64_MIN: Final = -(2**63)
_I64_MAX: Final = 2**63 - 1
_U64_MAX: Final = 2**64 - 1


@dataclass(frozen=True, slots=True)
class BottomFrameSelection:
    bottom: Frame
    first_eligible: Frame | None
    root_eligible: Frame | None
    bottom_is_collapsed: bool = False
    found_uncollapsed_eligible: bool = False


@dataclass(frozen=True, slots=True)
class ProfileFactIndex:
    facts: ProfileFacts

    @classmethod
    def from_input(cls, sample_facts: SampleFactsInput) -> ProfileFactIndex:
        if isinstance(sample_facts, ProfileFacts):
            return cls(sample_facts)
        samples = tuple(sample_facts)
        return cls(
            ProfileFacts(
                samples=samples,
                total_primary_value=sum(fact.primary_value for fact in samples),
                empty_sample_count=sum(1 for fact in samples if fact.is_empty),
            )
        )

    @property
    def total_primary_value(self) -> TimeNs:
        return self.facts.total_primary_value

    def samples(self) -> tuple[SampleFact, ...]:
        return self.facts.samples

    def non_empty_samples(self) -> tuple[SampleFact, ...]:
        return self.facts.non_empty_samples()

    def target_frames(
        self,
        fact: SampleFact,
        target_names: Iterable[str],
    ) -> tuple[Frame, ...]:
        names = frozenset(target_names)
        return tuple(frame for frame in fact.stack if frame.name in names)

    def first_caller_after_leaf(
        self,
        fact: SampleFact,
        predicate: Callable[[Frame], bool],
        *,
        limit: int | None = None,
    ) -> Frame | None:
        stack = fact.stack[1 : limit + 1 if limit is not None else None]
        return next((frame for frame in stack if predicate(frame)), None)

    def any_frame_matches(
        self,
        fact: SampleFact,
        predicate: Callable[[Frame], bool],
    ) -> bool:
        return any(predicate(frame) for frame in fact.stack)

    def select_bottom_frame(
        self,
        fact: SampleFact,
        *,
        is_eligible: Callable[[Frame], bool],
        is_collapsed: Callable[[Frame], bool],
    ) -> BottomFrameSelection | None:
        stack = fact.stack
        if not stack:
            return None

        bottom = stack[0]
        first_eligible: Frame | None = None
        found_uncollapsed = False
        for frame in stack:
            if not is_eligible(frame):
                continue
            first_eligible = first_eligible or frame
            if is_collapsed(frame):
                continue
            bottom = frame
            found_uncollapsed = True
            break

        if (
            first_eligible is not None
            and bottom == stack[0]
            and not is_eligible(bottom)
        ):
            bottom = first_eligible

        if first_eligible is None or found_uncollapsed:
            return BottomFrameSelection(
                bottom=bottom,
                first_eligible=first_eligible,
                root_eligible=None,
                bottom_is_collapsed=False,
                found_uncollapsed_eligible=found_uncollapsed,
            )

        root_eligible = next(
            (frame for frame in reversed(stack) if is_eligible(frame)),
            None,
        )
        return BottomFrameSelection(
            bottom=first_eligible,
            first_eligible=first_eligible,
            root_eligible=root_eligible,
            bottom_is_collapsed=is_collapsed(first_eligible),
            found_uncollapsed_eligible=False,
        )


def sample_facts_to_jsonable(facts: ProfileFacts) -> dict[str, JsonValue]:
    """Export sample facts in the compact interned v2 layout.

    Interning order is normative (mirrored by the Rust port): samples in
    order, stack frames in order; each new frame interns its name, then its
    filename, then appends its own row.
    """
    strings: list[str] = []
    string_indexes: dict[str, int] = {}
    frames: list[JsonValue] = []
    frame_indexes: dict[Frame, int] = {}

    def intern_string(value: str) -> int:
        index = string_indexes.get(value)
        if index is None:
            index = len(strings)
            strings.append(value)
            string_indexes[value] = index
        return index

    def intern_frame(frame: Frame) -> int:
        index = frame_indexes.get(frame)
        if index is None:
            row: list[JsonValue] = [
                frame.location_id,
                frame.function_id,
                intern_string(frame.name),
                intern_string(frame.filename),
                frame.line,
                frame.location_is_folded,
            ]
            index = len(frames)
            frames.append(row)
            frame_indexes[frame] = index
        return index

    samples_payload: list[JsonValue] = [
        {
            "sample_index": fact.sample_index,
            "values": list(fact.sample.values),
            "location_ids": list(fact.sample.location_ids),
            "stack": [intern_frame(frame) for frame in fact.stack],
        }
        for fact in facts.samples
    ]

    return {
        "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
        "tool": "clankerprof_facts",
        "profile": {
            "value_types": [
                {"type": value_type.type_name, "unit": value_type.unit}
                for value_type in facts.value_types
            ],
            "period_type": (
                {
                    "type": facts.period_type.type_name,
                    "unit": facts.period_type.unit,
                }
                if facts.period_type is not None
                else None
            ),
            "period": facts.period,
            "default_sample_type": facts.default_sample_type,
            "primary_value_index": facts.primary_value_index,
        },
        "summary": {
            "sample_count": len(facts.samples),
            "empty_sample_count": facts.empty_sample_count,
            "non_empty_sample_count": facts.non_empty_sample_count,
            "total_primary_value": facts.total_primary_value,
        },
        "strings": list(strings),
        "frames": frames,
        "samples": samples_payload,
    }


def sample_facts_from_jsonable(payload: JsonObject) -> ProfileFacts:
    schema_version = payload.get("schema_version")
    try:
        if schema_version == SAMPLE_FACTS_SCHEMA_VERSION:
            return _sample_facts_from_v2(payload)
        if schema_version == SAMPLE_FACTS_SCHEMA_VERSION_V1:
            return _sample_facts_from_v1(payload)
    except KeyError as exc:
        raise ValueError(
            f"Sample facts payload missing required key: {exc.args[0]!r}."
        ) from exc
    raise ValueError(
        "Unsupported sample facts schema version: "
        f"{schema_version!r}; expected {SAMPLE_FACTS_SCHEMA_VERSION!r} "
        f"or {SAMPLE_FACTS_SCHEMA_VERSION_V1!r}."
    )


def dumps_sample_facts(facts: ProfileFacts, *, pretty: bool = False) -> str:
    payload = sample_facts_to_jsonable(facts)
    if pretty:
        return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    )


def loads_sample_facts(payload: str) -> ProfileFacts:
    raw = parse_strict_json(payload)
    if not isinstance(raw, dict):
        raise ValueError("Sample facts JSON must be an object.")
    return sample_facts_from_jsonable(cast(JsonObject, raw))


def write_sample_facts(
    path: str | Path,
    facts: ProfileFacts,
    *,
    pretty: bool = False,
) -> None:
    Path(path).write_text(
        dumps_sample_facts(facts, pretty=pretty) + "\n",
        encoding="utf-8",
    )


def read_sample_facts(path: str | Path) -> ProfileFacts:
    return loads_sample_facts(Path(path).read_text(encoding="utf-8"))


def _sample_facts_from_v2(payload: JsonObject) -> ProfileFacts:
    profile_meta = _validated_profile_meta(payload.get("profile"))
    strings = _validated_strings(payload.get("strings"))
    frames = _validated_frames(payload.get("frames"), strings)
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Sample facts payload must contain a samples array.")

    primary_value_index = profile_meta.primary_value_index
    samples: list[SampleFact] = []
    for item in cast(list[object], raw_samples):
        if not isinstance(item, dict):
            raise ValueError("Each sample facts entry must be an object.")
        entry = cast(JsonObject, item)
        raw_stack = entry["stack"]
        if not isinstance(raw_stack, list):
            raise ValueError("Sample fact stack must be an array.")
        stack: list[Frame] = []
        for frame_index in cast(list[object], raw_stack):
            if isinstance(frame_index, bool) or not isinstance(frame_index, int):
                raise ValueError("Sample fact stack entries must be frame indexes.")
            if frame_index < 0 or frame_index >= len(frames):
                raise ValueError(
                    f"Sample fact frame index {frame_index} is out of range."
                )
            stack.append(frames[frame_index])
        samples.append(
            SampleFact(
                sample_index=_sample_index(entry),
                sample=Sample(
                    location_ids=_u64_tuple(
                        entry["location_ids"],
                        field_name="location_ids",
                    ),
                    values=_i64_tuple(entry["values"], field_name="values"),
                    primary_index=primary_value_index,
                ),
                stack=tuple(stack),
            )
        )

    facts_samples = tuple(samples)
    check_aggregate_bounds(facts_samples)
    total_primary_value = sum(fact.primary_value for fact in facts_samples)
    empty_sample_count = sum(1 for fact in facts_samples if fact.is_empty)
    _validate_summary(
        payload.get("summary"),
        sample_count=len(facts_samples),
        total_primary_value=total_primary_value,
        empty_sample_count=empty_sample_count,
    )
    return ProfileFacts(
        samples=facts_samples,
        total_primary_value=total_primary_value,
        empty_sample_count=empty_sample_count,
        value_types=profile_meta.value_types,
        period_type=profile_meta.period_type,
        period=profile_meta.period,
        default_sample_type=profile_meta.default_sample_type,
        primary_value_index=profile_meta.primary_value_index,
    )


@dataclass(frozen=True, slots=True)
class _ProfileMeta:
    value_types: tuple[ValueType, ...]
    period_type: ValueType | None
    period: int
    default_sample_type: str
    primary_value_index: int


def _validated_profile_meta(raw: object) -> _ProfileMeta:
    if not isinstance(raw, dict):
        raise ValueError("Sample facts payload must contain a profile object.")
    payload = cast(JsonObject, raw)
    raw_value_types = payload.get("value_types", [])
    if not isinstance(raw_value_types, list):
        raise ValueError("Sample facts profile value_types must be an array.")
    value_types = tuple(
        _value_type_from_jsonable(item) for item in cast(list[object], raw_value_types)
    )
    raw_period_type = payload.get("period_type")
    period_type = (
        _value_type_from_jsonable(raw_period_type)
        if raw_period_type is not None
        else None
    )
    raw_primary_value_index = payload.get("primary_value_index", 0)
    primary_value_index = _strict_int(raw_primary_value_index)
    if primary_value_index is None or primary_value_index > _I64_MAX:
        raise ValueError("Sample facts primary_value_index must be an integer.")
    if primary_value_index < 0:
        raise ValueError("Sample facts primary_value_index must be non-negative.")
    raw_period = payload.get("period", 0)
    period = _strict_int(raw_period)
    if period is None or period < _I64_MIN or period > _I64_MAX:
        raise ValueError("Sample facts profile period must be an integer.")
    return _ProfileMeta(
        value_types=value_types,
        period_type=period_type,
        period=period,
        default_sample_type=str(payload.get("default_sample_type", "")),
        primary_value_index=primary_value_index,
    )


def _value_type_from_jsonable(raw: object) -> ValueType:
    if not isinstance(raw, dict):
        raise ValueError("Sample facts value type must be an object.")
    payload = cast(JsonObject, raw)
    return ValueType(
        type_name=str(payload.get("type", "")),
        unit=str(payload.get("unit", "")),
    )


def _validated_strings(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError("Sample facts payload must contain a strings array.")
    strings: list[str] = []
    for item in cast(list[object], raw):
        if not isinstance(item, str):
            raise ValueError("Sample facts strings entries must be strings.")
        strings.append(item)
    return strings


def _validated_frames(raw: object, strings: list[str]) -> list[Frame]:
    if not isinstance(raw, list):
        raise ValueError("Sample facts payload must contain a frames array.")
    frames: list[Frame] = []
    for item in cast(list[object], raw):
        if not isinstance(item, list) or len(cast(list[object], item)) != 6:
            raise ValueError(
                "Each sample facts frame must be a six-element array of "
                "[location_id, function_id, name, filename, line, folded]."
            )
        row = cast(list[object], item)
        location_id, function_id, name_index, filename_index, line, folded = row
        parsed_ids: dict[str, int] = {}
        for label, value in (
            ("location_id", location_id),
            ("function_id", function_id),
        ):
            parsed = _strict_int(value)
            if parsed is None or parsed < 0 or parsed > _U64_MAX:
                raise ValueError(
                    f"Sample facts frame {label} must be an unsigned 64-bit integer."
                )
            parsed_ids[label] = parsed
        parsed_cells: dict[str, int] = {}
        for label, value in (
            ("name index", name_index),
            ("filename index", filename_index),
            ("line", line),
        ):
            parsed = _strict_int(value)
            if parsed is None or parsed < _I64_MIN or parsed > _I64_MAX:
                raise ValueError(f"Sample facts frame {label} must be an integer.")
            parsed_cells[label] = parsed
        if not isinstance(folded, bool):
            raise ValueError("Sample facts frame folded flag must be a boolean.")
        name_position = parsed_cells["name index"]
        filename_position = parsed_cells["filename index"]
        for position in (name_position, filename_position):
            if position < 0 or position >= len(strings):
                raise ValueError(
                    f"Sample facts frame string index {position} is out of range."
                )
        frames.append(
            Frame(
                location_id=parsed_ids["location_id"],
                function_id=parsed_ids["function_id"],
                name=strings[name_position],
                filename=strings[filename_position],
                line=parsed_cells["line"],
                location_is_folded=folded,
            )
        )
    return frames


def _validate_summary(
    raw: object,
    *,
    sample_count: int,
    total_primary_value: TimeNs,
    empty_sample_count: int,
) -> None:
    if not isinstance(raw, dict):
        return
    summary_payload = cast(JsonObject, raw)

    def summary_int(key: str) -> int | None:
        value = summary_payload.get(key)
        if value is None:
            return None
        parsed = _strict_int(value)
        if parsed is None or parsed < _I64_MIN or parsed > _U64_MAX:
            raise ValueError(f"Sample facts summary {key} must be an integer.")
        return parsed

    expected_count = summary_int("sample_count")
    if expected_count is not None and expected_count != sample_count:
        raise ValueError("Sample facts summary sample count does not match samples.")
    expected_total = summary_int("total_primary_value")
    if expected_total is not None and expected_total != total_primary_value:
        raise ValueError("Sample facts summary total does not match samples.")
    expected_empty = summary_int("empty_sample_count")
    if expected_empty is not None and expected_empty != empty_sample_count:
        raise ValueError("Sample facts empty count does not match samples.")
    expected_non_empty = summary_int("non_empty_sample_count")
    if (
        expected_non_empty is not None
        and expected_non_empty != sample_count - empty_sample_count
    ):
        raise ValueError("Sample facts non-empty count does not match samples.")


def _sample_facts_from_v1(payload: JsonObject) -> ProfileFacts:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Sample facts payload must contain a samples array.")
    raw_sample_items = cast(list[object], raw_samples)
    samples = tuple(
        _sample_fact_from_v1_jsonable(cast(JsonObject, item))
        for item in raw_sample_items
        if isinstance(item, dict)
    )
    if len(samples) != len(raw_sample_items):
        raise ValueError("Each sample facts entry must be an object.")
    check_aggregate_bounds(samples)
    total_primary_value = sum(fact.primary_value for fact in samples)
    empty_sample_count = sum(1 for fact in samples if fact.is_empty)
    _validate_summary(
        payload.get("summary"),
        sample_count=len(samples),
        total_primary_value=total_primary_value,
        empty_sample_count=empty_sample_count,
    )
    return ProfileFacts(
        samples=samples,
        total_primary_value=total_primary_value,
        empty_sample_count=empty_sample_count,
    )


def _sample_fact_from_v1_jsonable(payload: JsonObject) -> SampleFact:
    raw_stack = payload.get("stack", [])
    if not isinstance(raw_stack, list):
        raise ValueError("Sample fact stack must be an array.")
    raw_stack_items = cast(list[object], raw_stack)
    raw_values = _i64_tuple(payload.get("values", []), field_name="values")
    raw_locations = _u64_tuple(
        payload.get("location_ids", []),
        field_name="location_ids",
    )
    result = SampleFact(
        sample_index=_sample_index(payload),
        sample=Sample(
            location_ids=raw_locations,
            values=raw_values,
        ),
        stack=tuple(
            _frame_from_v1_jsonable(cast(JsonObject, item))
            for item in raw_stack_items
            if isinstance(item, dict)
        ),
    )
    if len(result.stack) != len(raw_stack_items):
        raise ValueError("Each sample fact frame must be an object.")
    raw_primary_value = payload.get("primary_value")
    if raw_primary_value is not None:
        expected_primary = _strict_int(raw_primary_value)
        if (
            expected_primary is None
            or expected_primary < _I64_MIN
            or expected_primary > _I64_MAX
        ):
            raise ValueError("Sample fact primary_value must be an integer.")
        if expected_primary != result.primary_value:
            raise ValueError("Sample fact primary value does not match values.")
    raw_is_empty = payload.get("is_empty")
    if raw_is_empty is not None:
        if not isinstance(raw_is_empty, bool):
            raise ValueError("Sample fact is_empty must be a boolean.")
        if raw_is_empty != result.is_empty:
            raise ValueError("Sample fact is_empty does not match stack.")
    return result


def _frame_from_v1_jsonable(payload: JsonObject) -> Frame:
    parsed_ids: dict[str, int] = {}
    for key in ("location_id", "function_id"):
        parsed = _strict_int(payload[key])
        if parsed is None or parsed < 0 or parsed > _U64_MAX:
            raise ValueError(
                f"Sample facts frame {key} must be an unsigned 64-bit integer."
            )
        parsed_ids[key] = parsed
    strings: dict[str, str] = {}
    for key in ("name", "filename"):
        raw = payload[key]
        if not isinstance(raw, str):
            raise ValueError(f"Sample fact frame {key} must be a string.")
        strings[key] = raw
    raw_line = payload.get("line", 0)
    line = _strict_int(raw_line)
    if line is None or line < _I64_MIN or line > _I64_MAX:
        raise ValueError("Sample facts frame line must be an integer.")
    raw_folded = payload.get("location_is_folded", False)
    if not isinstance(raw_folded, bool):
        raise ValueError("Sample facts frame folded flag must be a boolean.")
    return Frame(
        location_id=parsed_ids["location_id"],
        function_id=parsed_ids["function_id"],
        name=strings["name"],
        filename=strings["filename"],
        line=line,
        location_is_folded=raw_folded,
    )


def _strict_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _i64_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Sample fact {field_name} must be an array.")
    items: list[int] = []
    for item in cast(Sequence[object], value):
        parsed = _strict_int(item)
        if parsed is None or parsed < _I64_MIN or parsed > _I64_MAX:
            raise ValueError(
                f"Sample fact {field_name} entries must be signed 64-bit integers."
            )
        items.append(parsed)
    return tuple(items)


def _u64_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Sample fact {field_name} must be an array.")
    items: list[int] = []
    for item in cast(Sequence[object], value):
        parsed = _strict_int(item)
        if parsed is None or parsed < 0 or parsed > _U64_MAX:
            raise ValueError(
                f"Sample fact {field_name} entries must be unsigned 64-bit integers."
            )
        items.append(parsed)
    return tuple(items)


def _sample_index(entry: JsonObject) -> int:
    parsed = _strict_int(entry["sample_index"])
    if parsed is None or parsed < 0 or parsed > _U64_MAX:
        raise ValueError("Sample fact sample_index must be a non-negative integer.")
    return parsed
