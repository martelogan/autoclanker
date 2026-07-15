from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TypeAlias

TimeNs: TypeAlias = int


@dataclass(frozen=True, slots=True)
class Function:
    function_id: int
    name: str
    system_name: str
    filename: str
    start_line: int = 0


@dataclass(frozen=True, slots=True)
class ValueType:
    type_name: str
    unit: str


def select_primary_value_index(
    sample_types: tuple[ValueType, ...],
    default_sample_type: str,
) -> int:
    """Pick the value index projections aggregate, per pprof convention.

    `default_sample_type` wins when it names a declared sample type; otherwise
    the last declared type is the default. Profiles that declare no sample
    types keep index 0.
    """
    if default_sample_type:
        for index, value_type in enumerate(sample_types):
            if value_type.type_name == default_sample_type:
                return index
    if sample_types:
        return len(sample_types) - 1
    return 0


@dataclass(frozen=True, slots=True)
class Frame:
    location_id: int
    function_id: int
    name: str
    filename: str
    line: int = 0
    location_is_folded: bool = False


@dataclass(frozen=True, slots=True)
class Sample:
    location_ids: tuple[int, ...]
    values: tuple[int, ...]
    primary_index: int = 0

    @property
    def primary_value(self) -> int:
        if not self.values:
            return 0
        if 0 <= self.primary_index < len(self.values):
            return self.values[self.primary_index]
        return self.values[0]


@dataclass(frozen=True, slots=True)
class SampleFact:
    sample_index: int
    sample: Sample
    stack: tuple[Frame, ...]

    @property
    def primary_value(self) -> TimeNs:
        return self.sample.primary_value

    @property
    def leaf(self) -> Frame | None:
        return self.stack[0] if self.stack else None

    @property
    def is_empty(self) -> bool:
        return not self.stack


AGGREGATE_MAX = 2**64 - 1
AGGREGATE_MIN = -(2**63)

AGGREGATE_BOUNDS_ERROR = "Aggregate sample values exceed the supported integer range."


def check_aggregate_bounds(samples: Iterable[SampleFact]) -> None:
    """Enforce the facts aggregate bound shared with the Rust port.

    The sum of positive primary values must fit ``u64`` and the sum of
    negative primary values must fit ``i64``; every subset sum any projection
    can produce then lies in ``[AGGREGATE_MIN, AGGREGATE_MAX]``, so all
    derived aggregates are exact and representable as JSON integers in both
    languages.
    """
    positive = 0
    negative = 0
    for fact in samples:
        value = fact.primary_value
        if value > 0:
            positive += value
        else:
            negative += value
    if positive > AGGREGATE_MAX or negative < AGGREGATE_MIN:
        raise ValueError(AGGREGATE_BOUNDS_ERROR)


@dataclass(frozen=True, slots=True)
class ProfileFacts:
    samples: tuple[SampleFact, ...]
    total_primary_value: TimeNs
    empty_sample_count: int
    value_types: tuple[ValueType, ...] = ()
    period_type: ValueType | None = None
    period: int = 0
    default_sample_type: str = ""
    primary_value_index: int = 0

    @property
    def non_empty_sample_count(self) -> int:
        return len(self.samples) - self.empty_sample_count

    def non_empty_samples(self) -> tuple[SampleFact, ...]:
        return tuple(fact for fact in self.samples if not fact.is_empty)


@dataclass(frozen=True, slots=True)
class Location:
    location_id: int
    lines: tuple[tuple[int, int], ...]
    is_folded: bool = False


def _frames_by_location_cache() -> dict[int, tuple[Frame, ...]]:
    return {}


def _profile_facts_cache() -> dict[int, ProfileFacts]:
    return {}


@dataclass(frozen=True, slots=True)
class Profile:
    string_table: tuple[str, ...]
    functions: dict[int, Function]
    locations: dict[int, Location]
    samples: tuple[Sample, ...]
    sample_types: tuple[ValueType, ...] = ()
    period_type: ValueType | None = None
    period: int = 0
    default_sample_type: str = ""
    primary_value_index: int = 0
    # Interning caches: the location→frames expansion is invariant per
    # profile, so unique Frame instances are shared across every sample
    # occurrence instead of reallocated per stack.
    _frames_by_location: dict[int, tuple[Frame, ...]] = field(
        default_factory=_frames_by_location_cache,
        init=False,
        repr=False,
        compare=False,
    )
    _facts_cache: dict[int, ProfileFacts] = field(
        default_factory=_profile_facts_cache,
        init=False,
        repr=False,
        compare=False,
    )

    def _frames_for_location(self, location_id: int) -> tuple[Frame, ...]:
        cached = self._frames_by_location.get(location_id)
        if cached is not None:
            return cached
        frames: list[Frame] = []
        location = self.locations.get(location_id)
        if location is not None:
            for function_id, line in location.lines:
                function = self.functions.get(function_id)
                if function is None:
                    continue
                frames.append(
                    Frame(
                        location_id=location_id,
                        function_id=function_id,
                        name=function.name,
                        filename=function.filename,
                        line=line,
                        location_is_folded=location.is_folded,
                    )
                )
        interned = tuple(frames)
        self._frames_by_location[location_id] = interned
        return interned

    def stack_for_sample(self, sample: Sample) -> tuple[Frame, ...]:
        frames: list[Frame] = []
        for location_id in sample.location_ids:
            frames.extend(self._frames_for_location(location_id))
        return tuple(frames)

    def to_sample_facts(self) -> ProfileFacts:
        cached = self._facts_cache.get(0)
        if cached is not None:
            return cached
        facts = tuple(
            SampleFact(
                sample_index=index,
                sample=sample,
                stack=self.stack_for_sample(sample),
            )
            for index, sample in enumerate(self.samples)
        )
        check_aggregate_bounds(facts)
        result = ProfileFacts(
            samples=facts,
            total_primary_value=sum(fact.primary_value for fact in facts),
            empty_sample_count=sum(1 for fact in facts if fact.is_empty),
            value_types=self.sample_types,
            period_type=self.period_type,
            period=self.period,
            default_sample_type=self.default_sample_type,
            primary_value_index=self.primary_value_index,
        )
        self._facts_cache[0] = result
        return result

    def sample_facts(self) -> tuple[SampleFact, ...]:
        return self.to_sample_facts().samples

    def total_primary_value(self) -> TimeNs:
        return self.to_sample_facts().total_primary_value
