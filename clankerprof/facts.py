from __future__ import annotations

import json

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypeAlias, cast

from clankerprof.model import Frame, ProfileFacts, Sample, SampleFact, TimeNs

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
SampleFactsInput: TypeAlias = Iterable[SampleFact] | ProfileFacts
JsonObject: TypeAlias = dict[str, Any]

SAMPLE_FACTS_SCHEMA_VERSION: Final = "clankerprof.sample_facts.v1"


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


def sample_facts_to_jsonable(
    facts: ProfileFacts,
) -> dict[str, JsonValue]:
    return {
        "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
        "tool": "clankerprof_facts",
        "summary": {
            "sample_count": len(facts.samples),
            "empty_sample_count": facts.empty_sample_count,
            "non_empty_sample_count": facts.non_empty_sample_count,
            "total_primary_value": facts.total_primary_value,
        },
        "samples": [_sample_fact_to_jsonable(fact) for fact in facts.samples],
    }


def sample_facts_from_jsonable(payload: JsonObject) -> ProfileFacts:
    schema_version = payload.get("schema_version")
    if schema_version != SAMPLE_FACTS_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported sample facts schema version: "
            f"{schema_version!r}; expected {SAMPLE_FACTS_SCHEMA_VERSION!r}."
        )
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError("Sample facts payload must contain a samples array.")
    raw_sample_items = cast(list[object], raw_samples)
    samples = tuple(
        _sample_fact_from_jsonable(cast(JsonObject, item))
        for item in raw_sample_items
        if isinstance(item, dict)
    )
    if len(samples) != len(raw_sample_items):
        raise ValueError("Each sample facts entry must be an object.")
    total_primary_value = sum(fact.primary_value for fact in samples)
    empty_sample_count = sum(1 for fact in samples if fact.is_empty)

    summary = payload.get("summary", {})
    if isinstance(summary, dict):
        summary_payload = cast(JsonObject, summary)
        expected_count = summary_payload.get("sample_count")
        if expected_count is not None and int(expected_count) != len(samples):
            raise ValueError(
                "Sample facts summary sample count does not match samples."
            )
        expected_total = summary_payload.get("total_primary_value")
        if expected_total is not None and int(expected_total) != total_primary_value:
            raise ValueError("Sample facts summary total does not match samples.")
        expected_empty = summary_payload.get("empty_sample_count")
        if expected_empty is not None and int(expected_empty) != empty_sample_count:
            raise ValueError("Sample facts empty count does not match samples.")
        expected_non_empty = summary_payload.get("non_empty_sample_count")
        if (
            expected_non_empty is not None
            and int(expected_non_empty) != len(samples) - empty_sample_count
        ):
            raise ValueError("Sample facts non-empty count does not match samples.")

    return ProfileFacts(
        samples=samples,
        total_primary_value=total_primary_value,
        empty_sample_count=empty_sample_count,
    )


def dumps_sample_facts(facts: ProfileFacts) -> str:
    return json.dumps(sample_facts_to_jsonable(facts), indent=2, sort_keys=True)


def loads_sample_facts(payload: str) -> ProfileFacts:
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("Sample facts JSON must be an object.")
    return sample_facts_from_jsonable(cast(JsonObject, raw))


def write_sample_facts(path: str | Path, facts: ProfileFacts) -> None:
    Path(path).write_text(dumps_sample_facts(facts) + "\n", encoding="utf-8")


def read_sample_facts(path: str | Path) -> ProfileFacts:
    return loads_sample_facts(Path(path).read_text(encoding="utf-8"))


def _sample_fact_to_jsonable(fact: SampleFact) -> dict[str, JsonValue]:
    return {
        "sample_index": fact.sample_index,
        "primary_value": fact.primary_value,
        "values": list(fact.sample.values),
        "location_ids": list(fact.sample.location_ids),
        "is_empty": fact.is_empty,
        "stack": [_frame_to_jsonable(frame) for frame in fact.stack],
    }


def _sample_fact_from_jsonable(payload: JsonObject) -> SampleFact:
    raw_stack = payload.get("stack", [])
    if not isinstance(raw_stack, list):
        raise ValueError("Sample fact stack must be an array.")
    raw_stack_items = cast(list[object], raw_stack)
    raw_values = _int_tuple(payload.get("values", []), field_name="values")
    raw_locations = _int_tuple(
        payload.get("location_ids", []),
        field_name="location_ids",
    )
    result = SampleFact(
        sample_index=int(payload["sample_index"]),
        sample=Sample(
            location_ids=raw_locations,
            values=raw_values,
        ),
        stack=tuple(
            _frame_from_jsonable(cast(JsonObject, item))
            for item in raw_stack_items
            if isinstance(item, dict)
        ),
    )
    if len(result.stack) != len(raw_stack_items):
        raise ValueError("Each sample fact frame must be an object.")
    raw_primary_value = payload.get("primary_value")
    if raw_primary_value is not None and int(raw_primary_value) != result.primary_value:
        raise ValueError("Sample fact primary value does not match values.")
    raw_is_empty = payload.get("is_empty")
    if raw_is_empty is not None and bool(raw_is_empty) != result.is_empty:
        raise ValueError("Sample fact is_empty does not match stack.")
    return result


def _frame_to_jsonable(frame: Frame) -> dict[str, JsonValue]:
    return {
        "location_id": frame.location_id,
        "function_id": frame.function_id,
        "name": frame.name,
        "filename": frame.filename,
        "line": frame.line,
        "location_is_folded": frame.location_is_folded,
    }


def _frame_from_jsonable(payload: JsonObject) -> Frame:
    return Frame(
        location_id=int(payload["location_id"]),
        function_id=int(payload["function_id"]),
        name=str(payload["name"]),
        filename=str(payload["filename"]),
        line=int(payload.get("line", 0)),
        location_is_folded=bool(payload.get("location_is_folded", False)),
    )


def _int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"Sample fact {field_name} must be an array.")
    return tuple(int(cast(Any, item)) for item in cast(Sequence[object], value))
