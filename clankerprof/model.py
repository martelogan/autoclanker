from __future__ import annotations

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

    @property
    def primary_value(self) -> int:
        return self.values[0] if self.values else 0


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


@dataclass(frozen=True, slots=True)
class ProfileFacts:
    samples: tuple[SampleFact, ...]
    total_primary_value: TimeNs
    empty_sample_count: int

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


@dataclass(frozen=True, slots=True)
class Profile:
    string_table: tuple[str, ...]
    functions: dict[int, Function]
    locations: dict[int, Location]
    samples: tuple[Sample, ...]

    def stack_for_sample(self, sample: Sample) -> tuple[Frame, ...]:
        frames: list[Frame] = []
        for location_id in sample.location_ids:
            location = self.locations.get(location_id)
            if location is None:
                continue
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
        return tuple(frames)

    def to_sample_facts(self) -> ProfileFacts:
        facts = tuple(
            SampleFact(
                sample_index=index,
                sample=sample,
                stack=self.stack_for_sample(sample),
            )
            for index, sample in enumerate(self.samples)
        )
        return ProfileFacts(
            samples=facts,
            total_primary_value=sum(fact.primary_value for fact in facts),
            empty_sample_count=sum(1 for fact in facts if fact.is_empty),
        )

    def sample_facts(self) -> tuple[SampleFact, ...]:
        return self.to_sample_facts().samples

    def total_primary_value(self) -> TimeNs:
        return self.to_sample_facts().total_primary_value


@dataclass(slots=True)
class FunctionMetrics:
    count: int = 0
    cpu_time: TimeNs = 0


@dataclass(slots=True)
class CallerMetrics:
    count: int = 0
    cpu_time: TimeNs = 0


def _int_by_string() -> dict[str, int]:
    return {}


@dataclass(slots=True)
class SemanticCallerMetrics:
    count: int = 0
    caller_names: dict[str, int] = field(default_factory=_int_by_string)
    caller_files: dict[str, int] = field(default_factory=_int_by_string)


def _function_metrics_by_string() -> dict[str, FunctionMetrics]:
    return {}


def _string_set() -> set[str]:
    return set()


def _time_by_string() -> dict[str, TimeNs]:
    return {}


def _semantic_metrics_by_string() -> dict[str, SemanticCallerMetrics]:
    return {}


def _caller_metrics_by_string() -> dict[str, CallerMetrics]:
    return {}


def _domain_file_stats_by_string() -> dict[str, DomainFileStats]:
    return {}


def _caller_metrics_by_tuple() -> dict[tuple[str, str], CallerMetrics]:
    return {}


@dataclass(slots=True)
class CategoryStats:
    cpu_time: TimeNs = 0
    sample_count: int = 0
    functions: dict[str, FunctionMetrics] = field(
        default_factory=_function_metrics_by_string
    )
    files: set[str] = field(default_factory=_string_set)
    folded_from: dict[str, TimeNs] = field(default_factory=_time_by_string)
    semantic_callers: dict[str, SemanticCallerMetrics] = field(
        default_factory=_semantic_metrics_by_string
    )
    caller_leaf_pairs: dict[str, CallerMetrics] = field(
        default_factory=_caller_metrics_by_string
    )

    def add_function(self, name: str, value: TimeNs) -> None:
        metrics = self.functions.setdefault(name, FunctionMetrics())
        metrics.count += 1
        metrics.cpu_time += value

    def add_caller_leaf_pair(self, caller: str, leaf: str, value: TimeNs) -> None:
        pair_key = f"{caller} -> {leaf}"
        metrics = self.caller_leaf_pairs.setdefault(pair_key, CallerMetrics())
        metrics.count += 1
        metrics.cpu_time += value

    def add_semantic_caller(self, leaf: str, caller: Frame) -> None:
        metrics = self.semantic_callers.setdefault(leaf, SemanticCallerMetrics())
        metrics.count += 1
        metrics.caller_names[caller.name] = metrics.caller_names.get(caller.name, 0) + 1
        metrics.caller_files[caller.filename] = (
            metrics.caller_files.get(caller.filename, 0) + 1
        )


@dataclass(slots=True)
class DomainFileStats:
    filename: str
    cpu_time: TimeNs = 0
    sample_count: int = 0
    functions: dict[str, FunctionMetrics] = field(
        default_factory=_function_metrics_by_string
    )
    cost_kinds: dict[str, CallerMetrics] = field(
        default_factory=_caller_metrics_by_string
    )
    caller_leaf_pairs: dict[tuple[str, str], CallerMetrics] = field(
        default_factory=_caller_metrics_by_tuple
    )

    def add(
        self,
        owner_function: str,
        leaf_function: str,
        cost_kind: str,
        value: TimeNs,
    ) -> None:
        self.cpu_time += value
        self.sample_count += 1

        function_metrics = self.functions.setdefault(owner_function, FunctionMetrics())
        function_metrics.count += 1
        function_metrics.cpu_time += value

        cost_metrics = self.cost_kinds.setdefault(cost_kind, CallerMetrics())
        cost_metrics.count += 1
        cost_metrics.cpu_time += value

        pair_metrics = self.caller_leaf_pairs.setdefault(
            (owner_function, leaf_function),
            CallerMetrics(),
        )
        pair_metrics.count += 1
        pair_metrics.cpu_time += value


@dataclass(slots=True)
class DomainStats:
    cpu_time: TimeNs = 0
    sample_count: int = 0
    cost_kinds: dict[str, CallerMetrics] = field(
        default_factory=_caller_metrics_by_string
    )
    files: dict[str, DomainFileStats] = field(
        default_factory=_domain_file_stats_by_string
    )

    def add(
        self,
        owner: Frame,
        leaf: Frame,
        cost_kind: str,
        value: TimeNs,
    ) -> None:
        self.cpu_time += value
        self.sample_count += 1

        cost_metrics = self.cost_kinds.setdefault(cost_kind, CallerMetrics())
        cost_metrics.count += 1
        cost_metrics.cpu_time += value

        file_stats = self.files.setdefault(
            owner.filename,
            DomainFileStats(filename=owner.filename),
        )
        file_stats.add(owner.name, leaf.name, cost_kind, value)
