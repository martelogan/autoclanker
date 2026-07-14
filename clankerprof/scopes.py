"""Scope (boundary) decomposition: cost kinds, rollups, and owners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from clankerprof.categorize import (
    RuntimeCategoryCache,
    first_non_runtime_file_caller,
    frame_cache_key,
    is_internal_category_for_rules,
    should_fold_category,
    simplify_category,
)
from clankerprof.facts import ProfileFactIndex, SampleFactsInput
from clankerprof.model import CategoryStats, DomainStats, Frame, Profile, TimeNs
from clankerprof.patterns import (
    DEFAULT_LIBRARY_SELECTORS,
    DEFAULT_RUNTIME_RULES,
    extract_library_name,
    is_native_path,
    is_runtime_stdlib_path,
    match_path_pattern,
    match_regex,
)
from clankerprof.rules import RuntimeRuleSet
from clankerprof.slices import SliceDefinition

BoundaryCountMode: TypeAlias = Literal["occurrence", "once_per_sample"]


def _category_stats_by_string() -> dict[str, CategoryStats]:
    return {}


def _domain_stats_by_string() -> dict[str, DomainStats]:
    return {}


def _tuple_by_string() -> dict[str, tuple[str, ...]]:
    return {}


def _float_by_string() -> dict[str, float]:
    return {}


@dataclass(frozen=True, slots=True)
class FramePredicate:
    key: str
    value: str


@dataclass(frozen=True, slots=True)
class FramePredicateExpr:
    predicates: tuple[FramePredicate, ...] = ()
    any: tuple[FramePredicateExpr, ...] = ()
    all: tuple[FramePredicateExpr, ...] = ()
    not_: tuple[FramePredicateExpr, ...] = ()


FramePredicateInput: TypeAlias = FramePredicateExpr | Sequence[FramePredicate]


@dataclass(frozen=True, slots=True)
class BoundaryCategoryDefinition:
    name: str
    predicates: FramePredicateInput


@dataclass(frozen=True, slots=True)
class BoundaryDomainDefinition:
    name: str
    predicates: FramePredicateInput
    fallback: bool = False


@dataclass(frozen=True, slots=True)
class BoundaryDefinition:
    name: str
    predicates: FramePredicateInput
    buckets: Mapping[str, tuple[str, ...]] = field(default_factory=_tuple_by_string)
    attributables: Mapping[str, float] = field(default_factory=_float_by_string)
    exclude_descendants: FramePredicateInput = ()
    count: BoundaryCountMode = "occurrence"


@dataclass(frozen=True, slots=True)
class BoundaryAnalysisOptions:
    boundaries: tuple[BoundaryDefinition, ...]
    categories: tuple[BoundaryCategoryDefinition, ...] = ()
    domains: tuple[BoundaryDomainDefinition, ...] = ()
    slices: tuple[SliceDefinition, ...] = ()
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES
    enhanced_runtime_categorization: bool = True
    fold_runtime_internals: bool = False
    caller_fallback_when_uncategorized: bool = False
    legacy_no_enhanced_caller_fallback: bool = False


@dataclass(slots=True)
class BoundaryStats:
    name: str
    buckets: Mapping[str, tuple[str, ...]]
    attributables: Mapping[str, float]
    total_time: TimeNs = 0
    sample_count: int = 0
    categories: dict[str, CategoryStats] = field(
        default_factory=_category_stats_by_string
    )
    domains: dict[str, DomainStats] = field(default_factory=_domain_stats_by_string)


@dataclass(frozen=True, slots=True)
class BoundaryAnalysisResult:
    total_time_ns: TimeNs
    boundaries: tuple[BoundaryStats, ...]
    unique_frame_count: int


def parse_frame_predicate(
    raw: str,
    *,
    default_key: str = "path",
) -> FramePredicate:
    if raw == "native":
        return FramePredicate("native", "")
    key, separator, value = raw.partition(":")
    if not separator:
        return FramePredicate(default_key, raw)
    if not key:
        raise ValueError(f"Predicate key cannot be empty: {raw}")
    if key == "native":
        if value not in {"", "true", "false"}:
            raise ValueError(
                f"native: predicate value must be true or false, got {value!r}."
            )
        return FramePredicate("native", value)
    if not value:
        raise ValueError(f"Predicate value cannot be empty: {raw}")
    if key == "cost_kind":
        return FramePredicate("category", value)
    if key == "runtime_label":
        return FramePredicate("runtime_category", value)
    return FramePredicate(key, value)


def parse_frame_predicates(
    values: Sequence[str],
    *,
    default_key: str = "path",
) -> tuple[FramePredicate, ...]:
    return tuple(
        parse_frame_predicate(value, default_key=default_key) for value in values
    )


def frame_predicate_expr(
    values: Sequence[str],
    *,
    default_key: str = "path",
) -> FramePredicateExpr:
    return FramePredicateExpr(
        predicates=parse_frame_predicates(values, default_key=default_key)
    )


def normalize_frame_predicate_expr(value: FramePredicateInput) -> FramePredicateExpr:
    if isinstance(value, FramePredicateExpr):
        return value
    return FramePredicateExpr(predicates=tuple(value))


def frame_predicate_expr_leaf_predicates(
    value: FramePredicateInput,
) -> tuple[FramePredicate, ...]:
    expr = normalize_frame_predicate_expr(value)
    predicates = list(expr.predicates)
    for child in expr.any:
        predicates.extend(frame_predicate_expr_leaf_predicates(child))
    for child in expr.all:
        predicates.extend(frame_predicate_expr_leaf_predicates(child))
    for child in expr.not_:
        predicates.extend(frame_predicate_expr_leaf_predicates(child))
    return tuple(predicates)


def _validate_boundary_options(options: BoundaryAnalysisOptions) -> None:
    for definition in options.categories:
        if any(
            predicate.key == "category"
            for predicate in frame_predicate_expr_leaf_predicates(definition.predicates)
        ):
            raise ValueError(
                "Boundary category definitions cannot reference category: "
                f"predicates: {definition.name}"
            )


class _FramePredicateMatcher:
    def __init__(
        self,
        *,
        rules: RuntimeRuleSet,
        slices: Sequence[SliceDefinition],
        runtime_cache: RuntimeCategoryCache,
    ) -> None:
        self._rules = rules
        self._slices = tuple(slices)
        self._runtime_cache = runtime_cache
        self._configured_category_matcher: _BoundaryCategoryMatcher | None = None
        self._cache: dict[tuple[FramePredicate, tuple[int, str, str]], bool] = {}

    @property
    def unique_frame_count(self) -> int:
        return len({frame_key for _, frame_key in self._cache})

    def matches_any(
        self,
        frame: Frame,
        predicates: Sequence[FramePredicate],
    ) -> bool:
        return any(self.matches(frame, predicate) for predicate in predicates)

    def set_configured_category_matcher(
        self,
        matcher: _BoundaryCategoryMatcher,
    ) -> None:
        self._configured_category_matcher = matcher

    def matches_expr(
        self,
        frame: Frame,
        expression: FramePredicateInput,
    ) -> bool:
        expr = normalize_frame_predicate_expr(expression)
        has_term = bool(expr.predicates or expr.any or expr.all or expr.not_)
        if not has_term:
            return False
        if expr.predicates and not self.matches_any(frame, expr.predicates):
            return False
        if expr.any and not any(self.matches_expr(frame, child) for child in expr.any):
            return False
        if expr.all and not all(self.matches_expr(frame, child) for child in expr.all):
            return False
        return not (
            expr.not_ and any(self.matches_expr(frame, child) for child in expr.not_)
        )

    def matches(self, frame: Frame, predicate: FramePredicate) -> bool:
        key = (predicate, frame_cache_key(frame))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._matches_uncached(frame, predicate)
        self._cache[key] = result
        return result

    def _matches_uncached(self, frame: Frame, predicate: FramePredicate) -> bool:
        key = predicate.key
        value = predicate.value
        if key == "name":
            return value in frame.name
        if key == "name_eq":
            return value == frame.name
        if key in {"path", "glob"}:
            return match_path_pattern(value, frame.filename, self._rules)
        if key == "regex":
            return match_regex(value, frame.filename)
        if key == "native":
            is_native = is_native_path(frame.filename, self._rules)
            return not is_native if value == "false" else is_native
        if key == "category":
            if self._configured_category_matcher is None:
                raise ValueError("category: predicates require configured categories.")
            return self._configured_category_matcher.category_for(frame) == value
        if key == "runtime_category":
            category = self._runtime_cache.category_for(frame)
            return category == value or (
                simplify_category(
                    category,
                    verbose=self._rules.verbose,
                    rules=self._rules,
                )
                == value
            )
        if key == "slice":
            return any(
                definition.name == value
                and definition.matches_frame(frame, self._rules)
                for definition in self._slices
            )
        if key in DEFAULT_LIBRARY_SELECTORS or key in (
            self._rules.library_selector_path_patterns
        ):
            library_name = extract_library_name(
                frame.filename,
                self._rules,
                selector=key,
            )
            return library_name is not None if value == "*" else library_name == value
        raise ValueError(f"Unsupported predicate key: {key}")


class _BoundaryCategoryMatcher:
    def __init__(
        self,
        definitions: Sequence[BoundaryCategoryDefinition],
        predicate_matcher: _FramePredicateMatcher,
    ) -> None:
        self._definitions = tuple(definitions)
        self._predicate_matcher = predicate_matcher
        self._cache: dict[tuple[int, str, str], str | None] = {}

    def category_for(self, frame: Frame) -> str | None:
        key = frame_cache_key(frame)
        if key not in self._cache:
            self._cache[key] = self._category_for_uncached(frame)
        return self._cache[key]

    def _category_for_uncached(self, frame: Frame) -> str | None:
        for definition in self._definitions:
            if self._predicate_matcher.matches_expr(frame, definition.predicates):
                return definition.name
        return None


class _BoundaryDomainMatcher:
    def __init__(
        self,
        definitions: Sequence[BoundaryDomainDefinition],
        predicate_matcher: _FramePredicateMatcher,
    ) -> None:
        self._definitions = tuple(definitions)
        self._predicate_matcher = predicate_matcher
        self._fallbacks = {
            definition.name: definition.fallback for definition in definitions
        }
        self._cache: dict[tuple[int, str, str], str | None] = {}

    def is_fallback(self, domain_name: str) -> bool:
        return self._fallbacks.get(domain_name, False)

    def domain_for(self, frame: Frame) -> str | None:
        key = frame_cache_key(frame)
        if key not in self._cache:
            self._cache[key] = self._domain_for_uncached(frame)
        return self._cache[key]

    def _domain_for_uncached(self, frame: Frame) -> str | None:
        for definition in self._definitions:
            if self._predicate_matcher.matches_expr(frame, definition.predicates):
                return definition.name
        return None


def _boundary_category(
    stack: Sequence[Frame],
    options: BoundaryAnalysisOptions,
    runtime_cache: RuntimeCategoryCache,
    category_matcher: _BoundaryCategoryMatcher,
) -> tuple[str, Frame, bool, str | None]:
    leaf = stack[0]
    category = (
        runtime_cache.category_for(leaf)
        if options.enhanced_runtime_categorization
        else None
    )
    frame_to_categorize = leaf
    folded = False
    folded_category: str | None = None

    if should_fold_category(
        category,
        stack,
        options.runtime_rules,
        options.fold_runtime_internals,
    ):
        for caller in stack[1:]:
            caller_category = runtime_cache.category_for(caller)
            if is_internal_category_for_rules(
                caller_category,
                options.runtime_rules,
            ) or is_runtime_stdlib_path(caller.filename, options.runtime_rules):
                continue
            frame_to_categorize = caller
            folded = True
            folded_category = category
            category = caller_category
            break

    if category is None and (
        options.caller_fallback_when_uncategorized
        or options.legacy_no_enhanced_caller_fallback
    ):
        should_walk_up = leaf.filename.startswith("<") or (
            is_runtime_stdlib_path(leaf.filename, options.runtime_rules)
            and any(
                leaf.name.startswith(prefix)
                for prefix in options.runtime_rules.caller_fallback_name_prefixes
            )
        )
        if should_walk_up:
            caller = first_non_runtime_file_caller(stack, options.runtime_rules)
            if caller is not None:
                frame_to_categorize = caller

    if category is None and folded and frame_to_categorize.filename.startswith("<"):
        for rule in options.runtime_rules.native_name_category_rules:
            if rule.matches(frame_to_categorize.name, frame_to_categorize.filename):
                category = rule.category
                break

    if category is None:
        category = category_matcher.category_for(frame_to_categorize)

    if category is None:
        category = "Other"

    if category not in options.runtime_rules.main_simplified_categories:
        category = (
            simplify_category(
                category,
                verbose=options.runtime_rules.verbose,
                rules=options.runtime_rules,
            )
            or "Other"
        )

    return category, frame_to_categorize, folded, folded_category


def _is_non_runtime_file(frame: Frame, rules: RuntimeRuleSet) -> bool:
    return not frame.filename.startswith("<") and not is_runtime_stdlib_path(
        frame.filename,
        rules,
    )


def _domain_owner_so_far(
    current: tuple[str, Frame] | None,
    frame: Frame,
    matcher: _BoundaryDomainMatcher,
) -> tuple[str, Frame] | None:
    domain = matcher.domain_for(frame)
    if domain is None:
        return current
    if current is None:
        return (domain, frame)
    current_domain, _ = current
    if matcher.is_fallback(current_domain) and not matcher.is_fallback(domain):
        return (domain, frame)
    return current


def analyze_boundaries(
    profile: Profile,
    options: BoundaryAnalysisOptions,
) -> BoundaryAnalysisResult:
    return analyze_boundary_facts(profile.sample_facts(), options)


def analyze_boundary_facts(
    sample_facts: SampleFactsInput,
    options: BoundaryAnalysisOptions,
) -> BoundaryAnalysisResult:
    if not options.boundaries:
        raise ValueError("Boundary analysis requires at least one boundary.")
    _validate_boundary_options(options)

    results = tuple(
        BoundaryStats(
            name=boundary.name,
            buckets=boundary.buckets,
            attributables=boundary.attributables,
        )
        for boundary in options.boundaries
    )
    index = ProfileFactIndex.from_input(sample_facts)
    runtime_cache = RuntimeCategoryCache(options.runtime_rules)
    predicate_matcher = _FramePredicateMatcher(
        rules=options.runtime_rules,
        slices=options.slices,
        runtime_cache=runtime_cache,
    )
    category_matcher = _BoundaryCategoryMatcher(
        options.categories,
        predicate_matcher,
    )
    predicate_matcher.set_configured_category_matcher(category_matcher)
    domain_matcher = _BoundaryDomainMatcher(options.domains, predicate_matcher)

    for fact in index.non_empty_samples():
        value = fact.primary_value
        stack = fact.stack
        if not stack:
            continue
        leaf = stack[0]
        category, frame_to_categorize, folded, folded_category = _boundary_category(
            stack,
            options,
            runtime_cache,
            category_matcher,
        )
        domain_owner: tuple[str, Frame] | None = None
        fallback_owner: Frame | None = None
        first_non_runtime_caller_below: Frame | None = None
        excluded_boundaries: set[int] = set()
        counted_once_boundaries: set[int] = set()

        for position, frame in enumerate(stack):
            owner_before_boundary = domain_owner
            fallback_before_boundary = fallback_owner
            caller_below_boundary = (
                None if position == 0 else first_non_runtime_caller_below or leaf
            )

            for boundary_index, boundary in enumerate(options.boundaries):
                if boundary_index in excluded_boundaries:
                    continue
                if (
                    boundary.count == "once_per_sample"
                    and boundary_index in counted_once_boundaries
                ):
                    continue
                if not predicate_matcher.matches_expr(frame, boundary.predicates):
                    continue

                boundary_stats = results[boundary_index]
                boundary_stats.total_time += value
                boundary_stats.sample_count += 1
                if boundary.count == "once_per_sample":
                    counted_once_boundaries.add(boundary_index)

                category_stats = boundary_stats.categories.setdefault(
                    category,
                    CategoryStats(),
                )
                category_stats.cpu_time += value
                category_stats.sample_count += 1
                category_stats.add_function(leaf.name, value)
                category_stats.files.add(frame_to_categorize.filename)

                caller = caller_below_boundary
                if caller is None and len(stack) > 1:
                    caller = stack[1]
                if caller is not None:
                    category_stats.add_caller_leaf_pair(caller.name, leaf.name, value)
                if folded and folded_category:
                    category_stats.folded_from[leaf.name] = (
                        category_stats.folded_from.get(leaf.name, 0) + value
                    )

                if options.domains:
                    domain_name: str
                    owner: Frame
                    if owner_before_boundary is not None:
                        domain_name, owner = owner_before_boundary
                    else:
                        domain_name = "Uncategorized"
                        owner = fallback_before_boundary or leaf
                    domain_stats = boundary_stats.domains.setdefault(
                        domain_name,
                        DomainStats(),
                    )
                    domain_stats.add(owner, leaf, category, value)

            domain_owner = _domain_owner_so_far(domain_owner, frame, domain_matcher)
            if (
                position > 0
                and first_non_runtime_caller_below is None
                and _is_non_runtime_file(frame, options.runtime_rules)
            ):
                first_non_runtime_caller_below = frame
            if position > 0 and fallback_owner is None:
                fallback_owner = frame
            for boundary_index, boundary in enumerate(options.boundaries):
                if predicate_matcher.matches_expr(frame, boundary.exclude_descendants):
                    excluded_boundaries.add(boundary_index)

    return BoundaryAnalysisResult(
        total_time_ns=index.total_primary_value,
        boundaries=results,
        unique_frame_count=predicate_matcher.unique_frame_count,
    )
