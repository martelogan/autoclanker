"""Projection accumulators: mutable aggregation state for targets and scopes.

These are working-state classes owned by the projection layer, not part of
the immutable profile fact model in `model.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clankerprof.model import Frame, TimeNs


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
    # Tuple identity: delimiter-composed string keys merged distinct pairs
    # whose symbols contained the delimiter; display strings are render-time.
    caller_leaf_pairs: dict[tuple[str, str], CallerMetrics] = field(
        default_factory=_caller_metrics_by_tuple
    )

    def add_function(self, name: str, value: TimeNs) -> None:
        metrics = self.functions.setdefault(name, FunctionMetrics())
        metrics.count += 1
        metrics.cpu_time += value

    def add_caller_leaf_pair(self, caller: str, leaf: str, value: TimeNs) -> None:
        metrics = self.caller_leaf_pairs.setdefault((caller, leaf), CallerMetrics())
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
