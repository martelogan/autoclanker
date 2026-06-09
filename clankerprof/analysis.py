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

from clankerprof.model import (
    CategoryStats,
    Frame,
    Profile,
    ProfileFacts,
    SampleFact,
    TimeNs,
)
from clankerprof.rules import RuntimeRuleSet, load_runtime_rules

OutputMode = Literal["text", "csv", "json", "simple-csv"]
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
PatternMode: TypeAlias = Literal["auto", "library", "path", "regex"]
LibrarySelector: TypeAlias = Literal[
    "dependency", "gem", "library", "package", "vendor"
]
DEFAULT_RUNTIME_RULES = load_runtime_rules("generic")
SampleFactsInput: TypeAlias = Iterable[SampleFact] | ProfileFacts


def _json_metadata() -> dict[str, JsonValue]:
    return {}


@dataclass(frozen=True, slots=True)
class TargetAnalysisOptions:
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES
    enhanced_runtime_categorization: bool = True
    fold_runtime_internals: bool = False
    track_semantic_callers: bool = False
    attributables: Mapping[str, Mapping[str, float]] | None = None
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


def format_time(nanoseconds: TimeNs) -> str:
    milliseconds = nanoseconds / 1_000_000
    if milliseconds >= 60_000:
        return f"{milliseconds / 60_000:.2f} min"
    if milliseconds >= 1_000:
        return f"{milliseconds / 1_000:.2f} s"
    return f"{milliseconds:.2f} ms"


def _pattern_mode(pattern: str) -> tuple[PatternMode, str, LibrarySelector | None]:
    mode, separator, rest = pattern.partition(":")
    if separator and mode in {"dependency", "gem", "library", "package", "vendor"}:
        return "library", rest, cast(LibrarySelector, mode)
    if separator and mode in {"path", "glob"}:
        return "path", rest, None
    if separator and mode == "regex":
        return "regex", rest, None
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
    mode, resolved_pattern, selector = _pattern_mode(pattern)
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
    mode, resolved_pattern, selector = _pattern_mode(pattern)
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


def categorize_runtime_frame(frame: Frame, rules: RuntimeRuleSet) -> str | None:
    name = frame.name
    path = frame.filename

    for rule in rules.semantic_rules:
        if rule.matches(name, path):
            if path == "<cfunc>" and rule.native_category is not None:
                return rule.native_category
            return rule.category

    core_class = _direct_core_class(name, rules)
    if core_class is not None:
        if path == "<cfunc>":
            return rules.core_native_categories.get(
                core_class, rules.core_native_default_category
            )
        if is_runtime_stdlib_path(path, rules):
            return rules.core_native_categories.get(core_class, rules.stdlib_category)
        if path.startswith("<internal:"):
            return rules.core_native_categories.get(
                core_class, rules.internals_category
            )
        if core_class in rules.core_native_categories:
            return rules.core_native_categories[core_class]
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


def _first_meaningful_caller(
    stack: Sequence[Frame],
    rules: RuntimeRuleSet,
) -> Frame | None:
    for caller in stack[1:10]:
        if not caller.filename.startswith("<") and not is_runtime_stdlib_path(
            caller.filename,
            rules,
        ):
            return caller
    return stack[1] if len(stack) > 1 else None


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

    if category is None and options.legacy_no_enhanced_caller_fallback:
        should_walk_up = leaf.filename.startswith("<") or (
            is_runtime_stdlib_path(leaf.filename, options.runtime_rules)
            and any(
                leaf.name.startswith(prefix)
                for prefix in options.runtime_rules.legacy_caller_fallback_name_prefixes
            )
        )
        if should_walk_up:
            caller = _first_non_runtime_file_caller(stack, options.runtime_rules)
            if caller is not None:
                frame_to_categorize = caller

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

    for fact in _iter_sample_facts(sample_facts):
        value = fact.primary_value
        stack = fact.stack
        if not stack:
            continue
        leaf = stack[0]
        target_frames = [frame for frame in stack if frame.name in target_config]
        if not target_frames:
            continue
        for target_frame in target_frames:
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
            caller = _first_meaningful_caller(stack, resolved_options.runtime_rules)
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
        mode, resolved_pattern, _selector = _pattern_mode(pattern)
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
    if key in {"dependency", "gem", "library", "package", "vendor"}:
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


def _root_eligible_frame(
    stack: Sequence[Frame],
    *,
    no_collapse_native: bool,
    rules: RuntimeRuleSet,
) -> Frame | None:
    for frame in reversed(stack):
        if no_collapse_native or not _is_native_path(frame.filename, rules):
            return frame
    return None


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

    for fact in _iter_sample_facts(sample_facts):
        value = fact.primary_value
        total_time += value
        stack = fact.stack
        if not stack:
            continue
        leaf = stack[0]
        bottom = stack[0]
        first_eligible: Frame | None = None
        uncollapsible_frame: Frame | None = None
        bottom_is_collapsed = False
        found_uncollapsed_eligible = False
        for frame in stack:
            eligible = options.no_collapse_native or not _is_native_path(
                frame.filename,
                options.runtime_rules,
            )
            if not eligible:
                continue
            first_eligible = first_eligible or frame
            if _is_collapsed_frame(frame, stack, options):
                continue
            bottom = frame
            found_uncollapsed_eligible = True
            break
        if (
            first_eligible is not None
            and _is_native_path(bottom.filename, options.runtime_rules)
            and not options.no_collapse_native
        ):
            bottom = first_eligible
        if first_eligible is not None and not found_uncollapsed_eligible:
            bottom = first_eligible
            bottom_is_collapsed = _is_collapsed_frame(bottom, stack, options)
            uncollapsible_frame = _root_eligible_frame(
                stack,
                no_collapse_native=options.no_collapse_native,
                rules=options.runtime_rules,
            )
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


def _iter_sample_facts(sample_facts: SampleFactsInput) -> Iterable[SampleFact]:
    if isinstance(sample_facts, ProfileFacts):
        return sample_facts.samples
    return sample_facts
