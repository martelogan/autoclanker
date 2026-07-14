"""Slice projection: ownership attribution after filters and collapse."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from clankerprof.facts import ProfileFactIndex, SampleFactsInput
from clankerprof.model import Frame, Profile, TimeNs
from clankerprof.patterns import (
    DEFAULT_LIBRARY_SELECTORS,
    DEFAULT_RUNTIME_RULES,
    JsonValue,
    extract_library_name,
    is_gc_function,
    is_native_path,
    match_path_pattern,
)
from clankerprof.rules import RuntimeRuleSet


def _json_metadata() -> dict[str, JsonValue]:
    return {}


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
    if key == "slice" and not descendant:
        matched = any(
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
            not matched
            and bottom is not None
            and _sample_has_descendant_attribute_for_slice(
                stack,
                options,
                options.attributes,
                value,
            )
        ):
            matched = True
        return not matched if inverted else matched
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
    default_names = [item.name for item in options.slices if item.is_default]
    if len(default_names) > 1:
        raise ValueError(
            "Slice config declares multiple default slices: "
            f"{', '.join(default_names)}. Exactly one slice may set default."
        )
    default_slice = default_names[0] if default_names else "(all)"
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
                or not is_native_path(frame.filename, options.runtime_rules)
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
