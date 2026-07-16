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


def _slice_frame_stats_by_key() -> dict[tuple[str, str], SliceFrameStats]:
    return {}


def _time_by_string() -> dict[str, TimeNs]:
    return {}


@dataclass(slots=True)
class SliceStats:
    name: str
    time_ns: TimeNs = 0
    # Tuple identity: a "name\0filename" string key merged distinct frames
    # whose symbols contained the delimiter.
    frames: dict[tuple[str, str], SliceFrameStats] = field(
        default_factory=_slice_frame_stats_by_key
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


def _match_frame_predicate(
    frame: Frame,
    key: str,
    value: str,
    rules: RuntimeRuleSet,
) -> bool:
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


@dataclass(frozen=True, slots=True)
class _SliceFilter:
    inverted: bool
    descendant: bool
    key: str
    value: str


def _parse_slice_filter(raw_filter: str) -> _SliceFilter:
    inverted, descendant, body = _parse_filter_prefixes(raw_filter)
    key, _, value = body.partition(":")
    return _SliceFilter(inverted=inverted, descendant=descendant, key=key, value=value)


def _parse_collapse_rule(raw_filter: str) -> _SliceFilter:
    # Collapse rules take no ! / < prefixes; the raw string partitions as-is
    # so an unvalidated prefixed rule still fails as an unsupported key.
    key, _, value = raw_filter.partition(":")
    return _SliceFilter(inverted=False, descendant=False, key=key, value=value)


class _SliceMatcher:
    """Per-analysis slice evaluation state.

    The filter/collapse DSL is parsed once up front and every frame-level
    decision (predicate matches and descendant-free slice attribution) is
    memoized by frame identity, so rules evaluate once per unique frame
    instead of once per stack occurrence.
    """

    def __init__(self, options: SliceAnalysisOptions) -> None:
        self._options = options
        parsed = tuple(_parse_slice_filter(item) for item in options.filters)
        self._bottom_filters = tuple(item for item in parsed if not item.descendant)
        self._descendant_filters = tuple(item for item in parsed if item.descendant)
        self._collapse = tuple(_parse_collapse_rule(item) for item in options.collapse)
        self._has_descendant_attributes = any(
            rule.descendant for rule in options.attributes
        )
        self._predicate_cache: dict[tuple[str, str, str, str], bool] = {}
        self._slice_cache: dict[tuple[str, str], str] = {}
        self._native_cache: dict[str, bool] = {}
        self._default_slice = "(all)"
        for definition in options.slices:
            if definition.is_default:
                self._default_slice = definition.name

    def matches_predicate(self, frame: Frame, key: str, value: str) -> bool:
        cache_key = (frame.name, frame.filename, key, value)
        cached = self._predicate_cache.get(cache_key)
        if cached is None:
            cached = _match_frame_predicate(
                frame,
                key,
                value,
                self._options.runtime_rules,
            )
            self._predicate_cache[cache_key] = cached
        return cached

    def slice_without_descendant_attributes(self, frame: Frame) -> str:
        cache_key = (frame.name, frame.filename)
        cached = self._slice_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = self._resolve_frame_slice(frame)
        self._slice_cache[cache_key] = resolved
        return resolved

    def _resolve_frame_slice(self, frame: Frame) -> str:
        options = self._options
        for rule in options.attributes:
            if rule.descendant:
                continue
            if self.matches_predicate(frame, rule.key, rule.value):
                return rule.target_slice
        for definition in options.slices:
            if definition.is_default:
                continue
            if definition.matches_frame(frame, options.runtime_rules):
                return definition.name
        return self._default_slice

    def is_eligible_bottom(self, frame: Frame) -> bool:
        if self._options.no_collapse_native:
            return True
        cached = self._native_cache.get(frame.filename)
        if cached is None:
            cached = is_native_path(frame.filename, self._options.runtime_rules)
            self._native_cache[frame.filename] = cached
        return not cached

    def slice_for_sample(self, frame: Frame, stack: Sequence[Frame]) -> str:
        """Full attribution: descendant and bottom rules in declaration order."""
        if not self._has_descendant_attributes:
            return self.slice_without_descendant_attributes(frame)
        options = self._options
        for rule in options.attributes:
            frames = stack if rule.descendant else (frame,)
            if any(
                self.matches_predicate(candidate, rule.key, rule.value)
                for candidate in frames
            ):
                return rule.target_slice
        for definition in options.slices:
            if definition.is_default:
                continue
            if definition.matches_frame(frame, options.runtime_rules):
                return definition.name
        return self._default_slice

    def _filter_matches(
        self,
        item: _SliceFilter,
        stack: Sequence[Frame],
        bottom: Frame,
    ) -> bool:
        frames = stack if item.descendant else (bottom,)
        if item.key == "slice" and not item.descendant:
            matched = any(
                self.slice_without_descendant_attributes(frame) == item.value
                for frame in frames
            )
            # Rescue via the sample's EFFECTIVE attribution: the first-match
            # winning rule's target, not the mere existence of any losing
            # descendant rule targeting the slice.
            if not matched and self.slice_for_sample(bottom, stack) == item.value:
                matched = True
            return not matched if item.inverted else matched
        if item.key == "slice":
            matches = (
                self.slice_without_descendant_attributes(frame) == item.value
                for frame in frames
            )
        else:
            matches = (
                self.matches_predicate(frame, item.key, item.value) for frame in frames
            )
        if item.inverted:
            # Negation binds to descendant EXISTENCE: a stack containing the
            # forbidden frame must not pass just because some other frame
            # fails to match. (Bottom filters are single-frame, where the
            # two formulas coincide.)
            return not any(matches)
        return any(matches)

    def filters_match_sample(self, stack: Sequence[Frame], bottom: Frame) -> bool:
        return all(
            self._filter_matches(item, stack, bottom) for item in self._bottom_filters
        ) and (
            not self._descendant_filters
            or any(
                self._filter_matches(item, stack, bottom)
                for item in self._descendant_filters
            )
        )

    def is_collapsed(self, frame: Frame) -> bool:
        for item in self._collapse:
            if item.key == "slice":
                if self.slice_without_descendant_attributes(frame) == item.value:
                    return True
            elif self.matches_predicate(frame, item.key, item.value):
                return True
        return False


def analyze_slices(
    profile: Profile,
    options: SliceAnalysisOptions,
) -> SliceAnalysisResult:
    return analyze_slice_facts(profile.sample_facts(), options)


def validate_slice_definitions(slices: Sequence[SliceDefinition]) -> None:
    """Shared definition validation for every slice-loading surface.

    Scope configs load slice files without ever running slice analysis, so
    the checks must not live only inside `analyze_slice_facts`.
    """
    default_names = [item.name for item in slices if item.is_default]
    if len(default_names) > 1:
        raise ValueError(
            "Slice config declares multiple default slices: "
            f"{', '.join(default_names)}. Exactly one slice may set default."
        )
    seen_names: set[str] = set()
    for item in slices:
        # Duplicate names silently merged with first/last-wins metadata that
        # diverged across implementations; fail closed like multiple defaults.
        if item.name in seen_names:
            raise ValueError(
                f"Slice config declares duplicate slice name: {item.name}. "
                "Each slice name may be defined once."
            )
        seen_names.add(item.name)
        # A user slice under a pseudo name would be attributed and then
        # unconditionally stripped at render, leaving matched time with no
        # owning row.
        if item.name in (GC_PSEUDO_SLICE, UNCOLLAPSIBLE_PSEUDO_SLICE):
            raise ValueError(
                f"Slice config declares reserved pseudo-slice name: {item.name}. "
                "The names (gc) and (uncollapsible) are reserved for analyzer "
                "pseudo-outputs."
            )


def analyze_slice_facts(
    sample_facts: SampleFactsInput,
    options: SliceAnalysisOptions,
) -> SliceAnalysisResult:
    total_time = 0
    matching_time = 0
    gc_time = 0
    stats_by_slice: dict[str, SliceStats] = {}
    uncollapsible_stats = SliceStats(name=UNCOLLAPSIBLE_PSEUDO_SLICE)
    validate_slice_definitions(options.slices)
    default_names = [item.name for item in options.slices if item.is_default]
    default_slice = default_names[0] if default_names else "(all)"
    matcher = _SliceMatcher(options)
    index = ProfileFactIndex.from_input(sample_facts)

    for fact in index.samples():
        value = fact.primary_value
        total_time += value
        stack = fact.stack
        if not stack:
            continue
        leaf = stack[0]

        selection = index.select_bottom_frame(
            fact,
            is_eligible=matcher.is_eligible_bottom,
            is_collapsed=matcher.is_collapsed,
        )
        if selection is None:
            continue
        bottom = selection.bottom
        bottom_is_collapsed = selection.bottom_is_collapsed
        uncollapsible_frame = selection.root_eligible
        if options.filters and not matcher.filters_match_sample(stack, bottom):
            continue
        matching_time += value

        if is_gc_function(leaf.name):
            gc_time += value
            gc_stats = stats_by_slice.setdefault(
                GC_PSEUDO_SLICE,
                SliceStats(name=GC_PSEUDO_SLICE),
            )
            gc_stats.time_ns += value
            frame_key = (leaf.name, leaf.filename)
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

        slice_name = matcher.slice_for_sample(bottom, stack)
        slice_stats = stats_by_slice.setdefault(
            slice_name,
            SliceStats(name=slice_name, is_default=slice_name == default_slice),
        )
        slice_stats.time_ns += value
        frame_key = (bottom.name, bottom.filename)
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
            frame_key = (frame.name, frame.filename)
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
        uncollapsible=uncollapsible_stats if uncollapsible_stats.time_ns != 0 else None,
    )
