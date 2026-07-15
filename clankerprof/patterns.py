"""Path, regex, glob, and dependency-library matching primitives.

Everything here is rule-driven string matching over frame names and paths;
no categorization policy lives at this layer.
"""

from __future__ import annotations

import fnmatch
import re

from dataclasses import dataclass
from typing import Literal, TypeAlias

from clankerprof.model import TimeNs
from clankerprof.rules import RuntimeRuleSet, load_runtime_rules

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


PatternMode: TypeAlias = Literal["auto", "library", "path", "regex"]


LibrarySelector: TypeAlias = str


DEFAULT_LIBRARY_SELECTORS = frozenset(
    {"dependency", "gem", "library", "package", "vendor"}
)


DEFAULT_RUNTIME_RULES = load_runtime_rules("generic")


@dataclass(frozen=True, slots=True)
class LibraryPath:
    name: str
    relative_path: str


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


def is_runtime_stdlib_path(path: str, rules: RuntimeRuleSet) -> bool:
    if not path or path.startswith("<"):
        return False
    return any(marker in path for marker in rules.stdlib_path_markers)


def is_native_path(path: str, rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES) -> bool:
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
        try:
            match = re.search(pattern, normalized)
        except re.error as exc:
            # re.error is not a ValueError; unwrapped it would escape the CLI
            # error envelope as a traceback.
            raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc
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
                # Group 1 names the library when it participated in the match;
                # otherwise (non-participating optional group, or a pattern
                # with no groups) the whole match names it. The relative path
                # always runs from the naming component to the end of the
                # normalized path — identical to the Rust implementation.
                component = match.group(1) if match.groups() else None
                component_start = (
                    match.start(1) if component is not None else match.start(0)
                )
                if component is None:
                    component = match.group(0)
                return LibraryPath(
                    name=_normalize_library_component(component, rules),
                    relative_path=normalized[component_start:],
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
