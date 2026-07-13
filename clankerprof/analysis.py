from __future__ import annotations

import csv
import fnmatch
import io
import json
import re

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal, TypeAlias, cast

from clankerprof.facts import ProfileFactIndex, SampleFactsInput
from clankerprof.model import (
    CategoryStats,
    DomainStats,
    Frame,
    Profile,
    TimeNs,
)
from clankerprof.rules import (
    RuntimeRuleSet,
    load_runtime_rules,
    load_runtime_rules_file,
)

OutputMode = Literal["text", "csv", "json", "simple-csv"]
BoundaryCountMode: TypeAlias = Literal["occurrence", "once_per_sample"]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
PatternMode: TypeAlias = Literal["auto", "library", "path", "regex"]
LibrarySelector: TypeAlias = str
DEFAULT_LIBRARY_SELECTORS = frozenset(
    {"dependency", "gem", "library", "package", "vendor"}
)
DEFAULT_RUNTIME_RULES = load_runtime_rules("generic")


def _json_metadata() -> dict[str, JsonValue]:
    return {}


@dataclass(frozen=True, slots=True)
class TargetAnalysisOptions:
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES
    enhanced_runtime_categorization: bool = True
    fold_runtime_internals: bool = False
    track_semantic_callers: bool = False
    attributables: Mapping[str, Mapping[str, float]] | None = None
    caller_fallback_when_uncategorized: bool = False
    legacy_no_enhanced_caller_fallback: bool = False


@dataclass(frozen=True, slots=True)
class LibraryPath:
    name: str
    relative_path: str


@dataclass(frozen=True, slots=True)
class SliceDefinition:
    name: str
    path_patterns: tuple[str, ...] = ()
    is_default: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=_json_metadata)

    def matches_path(self, path: str) -> bool:
        return any(
            match_path_pattern(pattern, path, DEFAULT_RUNTIME_RULES)
            for pattern in self.path_patterns
        )

    def matches_frame(self, frame: Frame, rules: RuntimeRuleSet) -> bool:
        return any(
            match_path_pattern(pattern, frame.filename, rules)
            for pattern in self.path_patterns
        )


@dataclass(frozen=True, slots=True)
class AttributionRule:
    key: str
    value: str
    target_slice: str
    descendant: bool = False


@dataclass(frozen=True, slots=True)
class SliceAnalysisOptions:
    slices: tuple[SliceDefinition, ...] = ()
    filters: tuple[str, ...] = ()
    collapse: tuple[str, ...] = ()
    attributes: tuple[AttributionRule, ...] = ()
    top: int | None = None
    by_slice: str | None = None
    show_paths: bool = False
    no_collapse_native: bool = False
    unattributed_gems: int | None = None
    unattributed_libraries: int | None = None
    allow_virtual_attribute_slices: bool = False
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES


@dataclass(slots=True)
class SliceFrameStats:
    function: str
    filename: str
    line: int | None = None
    time_ns: TimeNs = 0


def _slice_frame_stats_by_string() -> dict[str, SliceFrameStats]:
    return {}


def _time_by_string() -> dict[str, TimeNs]:
    return {}


def _category_stats_by_string() -> dict[str, CategoryStats]:
    return {}


def _domain_stats_by_string() -> dict[str, DomainStats]:
    return {}


def _tuple_by_string() -> dict[str, tuple[str, ...]]:
    return {}


def _float_by_string() -> dict[str, float]:
    return {}


@dataclass(slots=True)
class SliceStats:
    name: str
    time_ns: TimeNs = 0
    frames: dict[str, SliceFrameStats] = field(
        default_factory=_slice_frame_stats_by_string
    )
    unattributed_gems: dict[str, TimeNs] = field(default_factory=_time_by_string)
    unattributed_libraries: dict[str, TimeNs] = field(default_factory=_time_by_string)
    is_default: bool = False


@dataclass(frozen=True, slots=True)
class SliceAnalysisResult:
    matching_time_ns: TimeNs
    total_time_ns: TimeNs
    slices: tuple[SliceStats, ...]
    gc_time_ns: TimeNs = 0
    uncollapsible: SliceStats | None = None


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


GC_PSEUDO_SLICE = "(gc)"
UNCOLLAPSIBLE_PSEUDO_SLICE = "(uncollapsible)"


def load_json_mapping(path: str | Path) -> dict[str, dict[str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Target config must be a JSON object.")
    result: dict[str, dict[str, str]] = {}
    for parent, categories in cast(dict[object, object], payload).items():
        if not isinstance(categories, dict):
            raise ValueError(f"Target config for {parent} must be an object.")
        raw_categories = cast(dict[object, object], categories)
        result[str(parent)] = {
            str(category): str(pattern) for category, pattern in raw_categories.items()
        }
    return result


def load_ruby_core_classes(path: str | Path) -> frozenset[str]:
    values: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[0].strip()
            if value and not value.startswith("#"):
                values.add(value)
    return frozenset(values)


def load_default_ruby_core_classes() -> frozenset[str]:
    values: set[str] = set()
    resource = resources.files("clankerprof.runtime_rules").joinpath(
        "ruby_core_classes.csv"
    )
    for row in csv.reader(io.StringIO(resource.read_text(encoding="utf-8"))):
        if not row:
            continue
        value = row[0].strip()
        if value and not value.startswith("#"):
            values.add(value)
    return frozenset(values)


def ruby_rules(core_classes: Iterable[str], *, verbose: bool = False) -> RuntimeRuleSet:
    return load_runtime_rules("ruby", core_classes=core_classes, verbose=verbose)


def runtime_rules_from_file(
    path: str | Path,
    *,
    core_classes: Iterable[str] = (),
    verbose: bool = False,
) -> RuntimeRuleSet:
    return load_runtime_rules_file(path, core_classes=core_classes, verbose=verbose)


def format_time(nanoseconds: TimeNs) -> str:
    milliseconds = nanoseconds / 1_000_000
    if milliseconds >= 60_000:
        return f"{milliseconds / 60_000:.2f} min"
    if milliseconds >= 1_000:
        return f"{milliseconds / 1_000:.2f} s"
    return f"{milliseconds:.2f} ms"


def _pattern_mode(
    pattern: str,
    rules: RuntimeRuleSet | None = None,
) -> tuple[PatternMode, str, LibrarySelector | None]:
    mode, separator, rest = pattern.partition(":")
    if separator and mode in {"path", "glob"}:
        return "path", rest, None
    if separator and mode == "regex":
        return "regex", rest, None
    selector_names = DEFAULT_LIBRARY_SELECTORS
    if rules is not None:
        selector_names = selector_names | frozenset(
            rules.library_selector_path_patterns
        )
    if separator and mode in selector_names:
        return "library", rest, mode
    return "auto", pattern, None


def _normalize_profile_path(path: str) -> str:
    return path.replace("\\", "/")


def _has_glob_token(pattern: str) -> bool:
    return any(token in pattern for token in ("*", "?", "["))


def _looks_like_path_pattern(pattern: str) -> bool:
    return "/" in pattern or "\\" in pattern or pattern.startswith(("./", "../"))


def match_path_pattern(
    pattern: str,
    path: str,
    rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
) -> bool:
    mode, resolved_pattern, selector = _pattern_mode(pattern, rules)
    if mode == "library":
        library_name = extract_library_name(path, rules, selector=selector)
        return (
            library_name is not None
            if resolved_pattern == "*"
            else library_name == resolved_pattern
        )
    if mode == "regex":
        return match_regex(resolved_pattern, path)

    normalized = _normalize_profile_path(path)
    normalized_pattern = _normalize_profile_path(resolved_pattern).removeprefix("./")
    if not normalized_pattern:
        return False

    if _has_glob_token(normalized_pattern):
        if fnmatch.fnmatch(normalized, normalized_pattern):
            return True
        return not normalized_pattern.startswith(("*", "/")) and fnmatch.fnmatch(
            normalized, f"**/{normalized_pattern}"
        )

    path_segments = normalized.strip("/")
    pattern_segments = normalized_pattern.strip("/")
    return (
        path_segments == pattern_segments
        or path_segments.startswith(f"{pattern_segments}/")
        or path_segments.endswith(f"/{pattern_segments}")
        or f"/{pattern_segments}/" in f"/{path_segments}/"
    )


def match_regex(pattern: str, path: str) -> bool:
    try:
        return bool(re.search(pattern, _normalize_profile_path(path)))
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc


def match_category_pattern(
    pattern: str,
    path: str,
    rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
) -> bool:
    mode, resolved_pattern, selector = _pattern_mode(pattern, rules)
    if mode == "path":
        return match_path_pattern(resolved_pattern, path, rules)
    if mode == "regex":
        return match_regex(resolved_pattern, path)
    if mode == "library":
        library_name = extract_library_name(path, rules, selector=selector)
        return (
            library_name is not None
            if resolved_pattern == "*"
            else library_name == resolved_pattern
        )

    try:
        if match_regex(resolved_pattern, path):
            return True
    except ValueError:
        if _looks_like_path_pattern(resolved_pattern):
            return match_path_pattern(resolved_pattern, path, rules)
        raise
    if _looks_like_path_pattern(resolved_pattern):
        return match_path_pattern(resolved_pattern, path, rules)
    return False


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


def _frame_cache_key(frame: Frame) -> tuple[int, str, str]:
    return (frame.function_id, frame.name, frame.filename)


class _FramePredicateMatcher:
    def __init__(
        self,
        *,
        rules: RuntimeRuleSet,
        slices: Sequence[SliceDefinition],
        runtime_cache: _RuntimeCategoryCache,
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
        key = (predicate, _frame_cache_key(frame))
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
            return _is_native_path(frame.filename, self._rules)
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


class _RuntimeCategoryCache:
    def __init__(self, rules: RuntimeRuleSet) -> None:
        self._rules = rules
        self._cache: dict[tuple[int, str, str], str | None] = {}

    def category_for(self, frame: Frame) -> str | None:
        key = _frame_cache_key(frame)
        if key not in self._cache:
            self._cache[key] = categorize_frame(frame, self._rules)
        return self._cache[key]


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
        key = _frame_cache_key(frame)
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
        key = _frame_cache_key(frame)
        if key not in self._cache:
            self._cache[key] = self._domain_for_uncached(frame)
        return self._cache[key]

    def _domain_for_uncached(self, frame: Frame) -> str | None:
        for definition in self._definitions:
            if self._predicate_matcher.matches_expr(frame, definition.predicates):
                return definition.name
        return None


def is_runtime_stdlib_path(path: str, rules: RuntimeRuleSet) -> bool:
    if not path or path.startswith("<"):
        return False
    return any(marker in path for marker in rules.stdlib_path_markers)


def is_ruby_stdlib_path(path: str) -> bool:
    return is_runtime_stdlib_path(path, ruby_rules(()))


def simplify_category(
    category: str | None,
    *,
    verbose: bool,
    rules: RuntimeRuleSet | None = None,
) -> str | None:
    if verbose or category is None:
        return category
    resolved_rules = rules or DEFAULT_RUNTIME_RULES
    return resolved_rules.simplification_map.get(category, category)


def _direct_core_class(function_name: str, rules: RuntimeRuleSet) -> str | None:
    clean = function_name.lstrip(":")
    core_classes = rules.core_classes
    class_name: str | None = None
    if "#" in clean:
        class_name = clean.split("#", 1)[0]
    elif "." in clean:
        class_name = clean.split(".", 1)[0]
    elif "::" in clean:
        class_name = clean.rsplit("::", 1)[0]
    if not class_name:
        return None
    if "::" in class_name:
        namespace = class_name.split("::", 1)[0]
        if namespace in rules.special_namespace_prefixes:
            return None
        for component in class_name.split("::"):
            if component in core_classes:
                return component
        return None
    if class_name in core_classes:
        return class_name
    return None


def _bare_guarded_namespace(function_name: str, rules: RuntimeRuleSet) -> bool:
    """Whether the name is a bare module-function on a guarded namespace.

    Qualified guarded names (`OpenSSL::Cipher#encrypt`) never reach the
    core-class table; bare forms (`Zlib.inflate`) do, so native-path
    categorization gives the pack's native-name rules the first claim before
    the core table's default swallows them.
    """
    clean = function_name.lstrip(":")
    if "#" in clean:
        class_name = clean.split("#", 1)[0]
    elif "." in clean:
        class_name = clean.split(".", 1)[0]
    else:
        return False
    return "::" not in class_name and class_name in rules.special_namespace_prefixes


def _is_runtime_owned_path(path: str, rules: RuntimeRuleSet) -> bool:
    """Whether the runtime, its stdlib, or a dependency owns this path.

    Semantic rules label runtime and dependency overhead by name; frames on
    plain application paths are never claimed, no matter how their names read.
    A pack that declares no path-ownership configuration cannot distinguish
    application paths, so its semantic rules apply to every frame.
    """
    if not (
        rules.native_path_markers
        or rules.native_path_patterns
        or rules.stdlib_path_markers
        or rules.library_path_patterns
        or rules.library_selector_path_patterns
    ):
        return True
    if _is_native_path(path, rules):
        return True
    if is_runtime_stdlib_path(path, rules):
        return True
    return extract_library_name(path, rules) is not None


def categorize_runtime_frame(frame: Frame, rules: RuntimeRuleSet) -> str | None:
    name = frame.name
    path = frame.filename

    if rules.semantic_rules and _is_runtime_owned_path(path, rules):
        for rule in rules.semantic_rules:
            if rule.matches(name, path):
                if path == "<cfunc>" and rule.native_category is not None:
                    return rule.native_category
                return rule.category

    core_class = _direct_core_class(name, rules)
    if core_class is not None:
        explicit_native_path = path == "<cfunc>" or (
            not path.startswith("<internal:")
            and (
                path.startswith("<")
                or any(marker in path for marker in rules.native_path_markers)
            )
        )
        if explicit_native_path:
            if _bare_guarded_namespace(name, rules):
                for rule in rules.native_rules:
                    if rule.matches(name, path):
                        return rule.category
            return rules.core_native_categories.get(
                core_class, rules.core_native_default_category
            )
        if is_runtime_stdlib_path(path, rules):
            return rules.core_stdlib_categories.get(core_class, rules.stdlib_category)
        if path.startswith("<internal:"):
            return rules.core_internal_categories.get(
                core_class, rules.internals_category
            )
        if core_class in rules.core_semantic_categories:
            return rules.core_semantic_categories[core_class]
        return None

    if path == "<cfunc>":
        for rule in rules.native_rules:
            if rule.matches(name, path):
                return rule.category
    if path.startswith("<internal:"):
        return rules.internals_category
    if is_runtime_stdlib_path(path, rules):
        return rules.stdlib_category
    return None


def categorize_ruby_frame(frame: Frame, rules: RuntimeRuleSet) -> str | None:
    return categorize_runtime_frame(frame, rules)


def categorize_frame(frame: Frame, rules: RuntimeRuleSet) -> str | None:
    if rules.enabled and _has_runtime_categories(rules):
        return categorize_runtime_frame(frame, rules)
    return None


def _should_fold_category(
    category: str | None,
    stack: Sequence[Frame],
    rules: RuntimeRuleSet,
    fold_runtime_internals: bool,
) -> bool:
    if category is None:
        return False
    if not rules.verbose:
        simplified = simplify_category(category, verbose=False, rules=rules)
        if simplified in rules.main_simplified_categories:
            return False
    if fold_runtime_internals and category in rules.always_foldable_categories:
        return True
    if (
        fold_runtime_internals
        and rules.verbose
        and category in rules.verbose_only_foldable_categories
    ):
        return True
    if category in rules.always_foldable_categories:
        for caller in stack[1:3]:
            caller_category = categorize_frame(caller, rules)
            if caller_category and not _is_internal_category_for_rules(
                caller_category, rules
            ):
                return True
    return False


def _is_internal_category_for_rules(
    category: str | None,
    rules: RuntimeRuleSet,
) -> bool:
    if category in rules.always_foldable_categories:
        return True
    return rules.verbose and category in rules.verbose_only_foldable_categories


def _first_non_runtime_file_caller(
    stack: Sequence[Frame],
    rules: RuntimeRuleSet,
) -> Frame | None:
    for caller in stack[1:]:
        if not caller.filename.startswith("<") and not is_runtime_stdlib_path(
            caller.filename,
            rules,
        ):
            return caller
    return None


def _target_category(
    stack: Sequence[Frame],
    parent_config: Mapping[str, str],
    options: TargetAnalysisOptions,
) -> tuple[str, Frame, bool, str | None]:
    leaf = stack[0]
    category = (
        categorize_frame(leaf, options.runtime_rules)
        if options.enhanced_runtime_categorization
        else None
    )
    frame_to_categorize = leaf
    folded = False
    folded_category: str | None = None

    if _should_fold_category(
        category,
        stack,
        options.runtime_rules,
        options.fold_runtime_internals,
    ):
        for caller in stack[1:]:
            caller_category = categorize_frame(caller, options.runtime_rules)
            if _is_internal_category_for_rules(
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
            caller = _first_non_runtime_file_caller(stack, options.runtime_rules)
            if caller is not None:
                frame_to_categorize = caller

    if category is None and folded and frame_to_categorize.filename.startswith("<"):
        for rule in options.runtime_rules.native_name_category_rules:
            if rule.matches(frame_to_categorize.name, frame_to_categorize.filename):
                category = rule.category
                break

    if category is None:
        for configured_category, pattern in parent_config.items():
            if match_category_pattern(
                pattern,
                frame_to_categorize.filename,
                options.runtime_rules,
            ):
                category = configured_category
                break

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


def analyze_targets(
    profile: Profile,
    target_config: Mapping[str, Mapping[str, str]],
    options: TargetAnalysisOptions | None = None,
) -> dict[str, dict[str, CategoryStats]]:
    return analyze_target_facts(profile.sample_facts(), target_config, options)


def analyze_target_facts(
    sample_facts: SampleFactsInput,
    target_config: Mapping[str, Mapping[str, str]],
    options: TargetAnalysisOptions | None = None,
) -> dict[str, dict[str, CategoryStats]]:
    resolved_options = options or TargetAnalysisOptions()
    results: dict[str, dict[str, CategoryStats]] = {}
    index = ProfileFactIndex.from_input(sample_facts)
    target_names = frozenset(target_config)

    for fact in index.non_empty_samples():
        value = fact.primary_value
        stack = fact.stack
        leaf = stack[0]
        target_frames = index.target_frames(fact, target_names)
        if not target_frames:
            continue
        seen_targets: set[str] = set()
        for target_frame in target_frames:
            if target_frame.name in seen_targets:
                continue
            seen_targets.add(target_frame.name)
            parent_results = results.setdefault(target_frame.name, {})
            category, frame_to_categorize, folded, folded_category = _target_category(
                stack,
                target_config[target_frame.name],
                resolved_options,
            )
            stats = parent_results.setdefault(category, CategoryStats())
            stats.cpu_time += value
            stats.sample_count += 1
            stats.add_function(leaf.name, value)
            stats.files.add(frame_to_categorize.filename)
            caller = index.first_caller_after_leaf(
                fact,
                lambda frame: (
                    not frame.filename.startswith("<")
                    and not is_runtime_stdlib_path(
                        frame.filename,
                        resolved_options.runtime_rules,
                    )
                ),
                limit=9,
            )
            if caller is None and len(stack) > 1:
                caller = stack[1]
            if caller is not None:
                stats.add_caller_leaf_pair(caller.name, leaf.name, value)
            if folded and folded_category:
                stats.folded_from[leaf.name] = (
                    stats.folded_from.get(leaf.name, 0) + value
                )
            if (
                resolved_options.track_semantic_callers
                and leaf.filename.startswith("<")
                and len(stack) > 1
            ):
                stats.add_semantic_caller(leaf.name, stack[1])

    return results


def _boundary_category(
    stack: Sequence[Frame],
    options: BoundaryAnalysisOptions,
    runtime_cache: _RuntimeCategoryCache,
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

    if _should_fold_category(
        category,
        stack,
        options.runtime_rules,
        options.fold_runtime_internals,
    ):
        for caller in stack[1:]:
            caller_category = runtime_cache.category_for(caller)
            if _is_internal_category_for_rules(
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
            caller = _first_non_runtime_file_caller(stack, options.runtime_rules)
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
    runtime_cache = _RuntimeCategoryCache(options.runtime_rules)
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


def _has_runtime_categories(rules: RuntimeRuleSet) -> bool:
    return bool(
        rules.semantic_rules
        or rules.native_rules
        or rules.core_classes
        or rules.stdlib_path_markers
        or rules.core_native_categories
    )


def _is_native_path(path: str, rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES) -> bool:
    if not path or path.startswith("<"):
        return True
    if any(marker in path for marker in rules.native_path_exclude_markers):
        return False
    if any(
        match_regex(pattern, path) for pattern in rules.native_path_exclude_patterns
    ):
        return False
    if any(marker in path for marker in rules.native_path_markers):
        return True
    return any(
        match_category_pattern(pattern, path, rules)
        for pattern in rules.native_path_patterns
    )


def is_gc_function(name: str) -> bool:
    return name in {"(marking)", "(sweeping)"}


def _normalize_library_component(component: str, rules: RuntimeRuleSet) -> str:
    normalized = component.strip("/")
    if not normalized:
        return normalized
    for pattern in rules.library_name_suffix_patterns:
        match = re.search(pattern, normalized)
        if match and match.start() > 0:
            return normalized[: match.start()]
    return normalized


def extract_library_path(
    path: str,
    rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    *,
    selector: str | None = None,
) -> LibraryPath | None:
    normalized = _normalize_profile_path(path)
    selector_patterns = (
        rules.library_selector_path_patterns.get(selector, ()) if selector else ()
    )
    patterns = selector_patterns or rules.library_path_patterns
    for pattern in patterns:
        mode, resolved_pattern, _selector = _pattern_mode(pattern, rules)
        if mode == "regex":
            try:
                match = re.search(resolved_pattern, normalized)
            except re.error as exc:
                raise ValueError(
                    f"Invalid library regex pattern {resolved_pattern!r}: {exc}"
                ) from exc
            if match:
                component = match.group(1) if match.groups() else match.group(0)
                relative_path = (
                    normalized[match.start(1) :] if match.groups() else component
                )
                return LibraryPath(
                    name=_normalize_library_component(component, rules),
                    relative_path=relative_path,
                )
            continue
        marker = _normalize_profile_path(resolved_pattern).strip("/")
        if not marker:
            continue
        marker_text = f"/{marker}/"
        haystack = f"/{normalized.strip('/')}/"
        if marker_text not in haystack:
            continue
        relative_path = haystack.split(marker_text, 1)[1].rstrip("/")
        component = relative_path.split("/", 1)[0]
        return LibraryPath(
            name=_normalize_library_component(component, rules),
            relative_path=relative_path,
        )
    return None


def extract_library_name(
    path: str,
    rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    *,
    selector: str | None = None,
) -> str | None:
    library_path = extract_library_path(path, rules, selector=selector)
    return library_path.name if library_path is not None else None


def extract_gem_name(path: str) -> str | None:
    return extract_library_name(path, ruby_rules(()), selector="gem")


def _matches_frame_filter(
    frame: Frame,
    raw_filter: str,
    rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
) -> bool:
    key, _, value = raw_filter.partition(":")
    if key == "name":
        return value in frame.name
    if key == "path":
        return value in frame.filename
    if key in DEFAULT_LIBRARY_SELECTORS or key in rules.library_selector_path_patterns:
        library_name = extract_library_name(frame.filename, rules, selector=key)
        return library_name is not None if value == "*" else library_name == value
    raise ValueError(f"Unsupported filter key: {key}")


def _parse_filter_prefixes(raw_filter: str) -> tuple[bool, bool, str]:
    inverted = False
    descendant = False
    body = raw_filter
    while body.startswith(("!", "<")):
        if body.startswith("!"):
            inverted = True
            body = body[1:]
            continue
        descendant = True
        body = body[1:]
    return inverted, descendant, body


def _filter_matches_stack(
    raw_filter: str,
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
    bottom: Frame | None = None,
) -> bool:
    inverted, descendant, body = _parse_filter_prefixes(raw_filter)
    key, _, value = body.partition(":")
    frames = stack if descendant else (bottom or stack[0],)
    if key == "slice":
        matches = (
            _slice_for_frame(
                frame,
                stack,
                options,
                options.slices,
                options.attributes,
                include_descendant_attributes=False,
            )
            == value
            for frame in frames
        )
        if (
            not descendant
            and not inverted
            and bottom is not None
            and _sample_has_descendant_attribute_for_slice(
                stack,
                options,
                options.attributes,
                value,
            )
        ):
            return True
    else:
        matches = (
            _matches_frame_filter(frame, body, options.runtime_rules)
            for frame in frames
        )
    if inverted:
        return any(not matched for matched in matches)
    return any(matches)


def _filters_match_sample(
    filters: Sequence[str],
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
    bottom: Frame,
) -> bool:
    bottom_filters: list[str] = []
    descendant_filters: list[str] = []
    for raw_filter in filters:
        _, descendant, _ = _parse_filter_prefixes(raw_filter)
        if descendant:
            descendant_filters.append(raw_filter)
        else:
            bottom_filters.append(raw_filter)
    return all(
        _filter_matches_stack(raw_filter, stack, options, bottom)
        for raw_filter in bottom_filters
    ) and (
        not descendant_filters
        or any(
            _filter_matches_stack(raw_filter, stack, options, bottom)
            for raw_filter in descendant_filters
        )
    )


def _collapse_matches_frame(
    raw_filter: str,
    frame: Frame,
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
) -> bool:
    key, _, value = raw_filter.partition(":")
    if key == "slice":
        return (
            _slice_for_frame(
                frame,
                stack,
                options,
                options.slices,
                options.attributes,
                include_descendant_attributes=False,
            )
            == value
        )
    return _matches_frame_filter(frame, raw_filter, options.runtime_rules)


def _is_collapsed_frame(
    frame: Frame,
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
) -> bool:
    return any(
        _collapse_matches_frame(collapse_filter, frame, stack, options)
        for collapse_filter in options.collapse
    )


def _slice_for_frame(
    frame: Frame,
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
    slices: Sequence[SliceDefinition],
    attributes: Sequence[AttributionRule],
    *,
    include_descendant_attributes: bool = True,
) -> str:
    for rule in attributes:
        if rule.descendant and not include_descendant_attributes:
            continue
        frames = stack if rule.descendant else (frame,)
        if any(
            _matches_frame_filter(
                candidate,
                f"{rule.key}:{rule.value}",
                options.runtime_rules,
            )
            for candidate in frames
        ):
            return rule.target_slice
    default = "(all)"
    for slice_definition in slices:
        if slice_definition.is_default:
            default = slice_definition.name
            continue
        if slice_definition.matches_frame(frame, options.runtime_rules):
            return slice_definition.name
    return default


def _sample_has_descendant_attribute_for_slice(
    stack: Sequence[Frame],
    options: SliceAnalysisOptions,
    attributes: Sequence[AttributionRule],
    slice_name: str,
) -> bool:
    for rule in attributes:
        if not rule.descendant or rule.target_slice != slice_name:
            continue
        if any(
            _matches_frame_filter(
                candidate,
                f"{rule.key}:{rule.value}",
                options.runtime_rules,
            )
            for candidate in stack
        ):
            return True
    return False


def analyze_slices(
    profile: Profile,
    options: SliceAnalysisOptions,
) -> SliceAnalysisResult:
    return analyze_slice_facts(profile.sample_facts(), options)


def analyze_slice_facts(
    sample_facts: SampleFactsInput,
    options: SliceAnalysisOptions,
) -> SliceAnalysisResult:
    total_time = 0
    matching_time = 0
    gc_time = 0
    stats_by_slice: dict[str, SliceStats] = {}
    uncollapsible_stats = SliceStats(name=UNCOLLAPSIBLE_PSEUDO_SLICE)
    default_slice = next(
        (item.name for item in options.slices if item.is_default), "(all)"
    )
    index = ProfileFactIndex.from_input(sample_facts)

    for fact in index.samples():
        value = fact.primary_value
        total_time += value
        stack = fact.stack
        if not stack:
            continue
        leaf = stack[0]

        def is_collapsed_for_sample(
            frame: Frame, sample_stack: Sequence[Frame] = stack
        ) -> bool:
            return _is_collapsed_frame(frame, sample_stack, options)

        selection = index.select_bottom_frame(
            fact,
            is_eligible=lambda frame: (
                options.no_collapse_native
                or not _is_native_path(frame.filename, options.runtime_rules)
            ),
            is_collapsed=is_collapsed_for_sample,
        )
        if selection is None:
            continue
        bottom = selection.bottom
        bottom_is_collapsed = selection.bottom_is_collapsed
        uncollapsible_frame = selection.root_eligible
        if options.filters and not _filters_match_sample(
            options.filters,
            stack,
            options,
            bottom,
        ):
            continue
        matching_time += value

        if is_gc_function(leaf.name):
            gc_time += value
            gc_stats = stats_by_slice.setdefault(
                GC_PSEUDO_SLICE,
                SliceStats(name=GC_PSEUDO_SLICE),
            )
            gc_stats.time_ns += value
            frame_key = f"{leaf.name}\0{leaf.filename}"
            frame_stats = gc_stats.frames.setdefault(
                frame_key,
                SliceFrameStats(
                    function=leaf.name,
                    filename=leaf.filename,
                    line=leaf.line or None,
                ),
            )
            frame_stats.time_ns += value
            continue

        slice_name = _slice_for_frame(
            bottom,
            stack,
            options,
            options.slices,
            options.attributes,
        )
        slice_stats = stats_by_slice.setdefault(
            slice_name,
            SliceStats(name=slice_name, is_default=slice_name == default_slice),
        )
        slice_stats.time_ns += value
        frame_key = f"{bottom.name}\0{bottom.filename}"
        frame_stats = slice_stats.frames.setdefault(
            frame_key,
            SliceFrameStats(
                function=bottom.name,
                filename=bottom.filename,
                line=bottom.line or None,
            ),
        )
        frame_stats.time_ns += value
        if slice_name == default_slice:
            library_name = extract_library_name(bottom.filename, options.runtime_rules)
            if library_name is not None:
                slice_stats.unattributed_libraries[library_name] = (
                    slice_stats.unattributed_libraries.get(library_name, 0) + value
                )
                slice_stats.unattributed_gems[library_name] = (
                    slice_stats.unattributed_gems.get(library_name, 0) + value
                )
        if bottom_is_collapsed:
            uncollapsible_stats.time_ns += value
            frame = uncollapsible_frame or bottom
            frame_key = f"{frame.name}\0{frame.filename}"
            frame_stats = uncollapsible_stats.frames.setdefault(
                frame_key,
                SliceFrameStats(
                    function=frame.name,
                    filename=frame.filename,
                    line=frame.line or None,
                ),
            )
            frame_stats.time_ns += value

    ordered = tuple(
        sorted(stats_by_slice.values(), key=lambda item: item.time_ns, reverse=True)
    )
    return SliceAnalysisResult(
        matching_time_ns=matching_time,
        total_time_ns=total_time,
        slices=ordered,
        gc_time_ns=gc_time,
        uncollapsible=uncollapsible_stats if uncollapsible_stats.time_ns > 0 else None,
    )
