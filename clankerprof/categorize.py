"""The runtime categorization engine: rule packs applied to frames."""

from __future__ import annotations

import csv
import io

from collections.abc import Callable, Iterable, Sequence
from importlib import resources
from pathlib import Path

from clankerprof.model import Frame
from clankerprof.patterns import (
    DEFAULT_RUNTIME_RULES,
    extract_library_name,
    is_native_path,
    is_runtime_stdlib_path,
    native_path_excluded,
)
from clankerprof.rules import (
    RuntimeRuleSet,
    load_runtime_rules,
    load_runtime_rules_file,
)

# The csv module's default 131072-byte field cap is a Python-side guard, not
# a CSV dialect property; the Rust scanner is unbounded, so fields of any
# length must parse identically. Portable C-long bound for 32-bit platforms.
_CSV_FIELD_LIMIT = 2**31 - 1


def load_ruby_core_classes(path: str | Path) -> frozenset[str]:
    values: set[str] = set()
    old_limit = csv.field_size_limit(_CSV_FIELD_LIMIT)
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if not row:
                    continue
                value = row[0].strip()
                if value and not value.startswith("#"):
                    values.add(value)
    finally:
        csv.field_size_limit(old_limit)
    return frozenset(values)


def load_default_ruby_core_classes() -> frozenset[str]:
    values: set[str] = set()
    resource = resources.files("clankerprof.runtime_rules").joinpath(
        "ruby_core_classes.csv"
    )
    old_limit = csv.field_size_limit(_CSV_FIELD_LIMIT)
    try:
        for row in csv.reader(io.StringIO(resource.read_text(encoding="utf-8"))):
            if not row:
                continue
            value = row[0].strip()
            if value and not value.startswith("#"):
                values.add(value)
    finally:
        csv.field_size_limit(old_limit)
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


def frame_cache_key(frame: Frame) -> tuple[int, str, str]:
    return (frame.function_id, frame.name, frame.filename)


class RuntimeCategoryCache:
    def __init__(self, rules: RuntimeRuleSet) -> None:
        self._rules = rules
        self._cache: dict[tuple[int, str, str], str | None] = {}

    def category_for(self, frame: Frame) -> str | None:
        key = frame_cache_key(frame)
        if key not in self._cache:
            self._cache[key] = categorize_frame(frame, self._rules)
        return self._cache[key]


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
    if is_native_path(path, rules):
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
        # Core-class explicit-native routing keys on pseudo-paths and the
        # pack's native_path_markers, now exclusion-aware (markers previously
        # ignored the exclude keys — a /native/app/ exclusion could not veto
        # a /native/ marker). native_path_patterns deliberately do NOT
        # core-route: they feed the pack's general native detection
        # (runtime-owned checks, native: predicates), where e.g. the ruby
        # pack's version-dir pattern must not reclassify stdlib or vendored
        # interpreter files away from stdlib/semantic routing. A pack that
        # wants pattern-style core routing declares the substring as a
        # marker. <internal:> keeps precedence (core_internal_categories
        # below), and <cfunc> stays the fast path.
        explicit_native_path = path == "<cfunc>" or (
            not path.startswith("<internal:")
            and not native_path_excluded(path, rules)
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


def should_fold_category(
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
        # The caller window spans the next two distinct locations so the
        # outcome is independent of inline expansion of the leaf's location.
        leaf_location_id = stack[0].location_id if stack else 0
        window_location_ids: list[int] = []
        for caller in stack[1:]:
            if caller.location_id == leaf_location_id and not window_location_ids:
                continue
            if caller.location_id not in window_location_ids:
                if len(window_location_ids) == 2:
                    break
                window_location_ids.append(caller.location_id)
            caller_category = categorize_frame(caller, rules)
            if caller_category and not is_internal_category_for_rules(
                caller_category, rules
            ):
                return True
    return False


def is_internal_category_for_rules(
    category: str | None,
    rules: RuntimeRuleSet,
) -> bool:
    if category in rules.always_foldable_categories:
        return True
    return rules.verbose and category in rules.verbose_only_foldable_categories


def first_non_runtime_file_caller(
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


def _has_runtime_categories(rules: RuntimeRuleSet) -> bool:
    return bool(
        rules.semantic_rules
        or rules.native_rules
        or rules.core_classes
        or rules.stdlib_path_markers
        or rules.core_native_categories
    )


def extract_gem_name(path: str) -> str | None:
    return extract_library_name(path, ruby_rules(()), selector="gem")


def categorize_stack(
    stack: Sequence[Frame],
    *,
    rules: RuntimeRuleSet,
    enhanced_runtime_categorization: bool,
    fold_runtime_internals: bool,
    caller_fallback_when_uncategorized: bool,
    runtime_category_for: Callable[[Frame], str | None],
    configured_category_for: Callable[[Frame], str | None],
) -> tuple[str, Frame, bool, str | None]:
    """The shared categorization engine behind targets and scopes.

    Projections differ only in how they look up runtime categories (cached or
    direct) and how configured categories match a frame; both strategies are
    injected. Returns (category, categorized frame, folded, folded category).
    """
    leaf = stack[0]
    category = runtime_category_for(leaf) if enhanced_runtime_categorization else None
    frame_to_categorize = leaf
    folded = False
    folded_category: str | None = None

    if should_fold_category(category, stack, rules, fold_runtime_internals):
        for caller in stack[1:]:
            caller_category = runtime_category_for(caller)
            if is_internal_category_for_rules(
                caller_category,
                rules,
            ) or is_runtime_stdlib_path(caller.filename, rules):
                continue
            frame_to_categorize = caller
            folded = True
            folded_category = category
            category = caller_category
            break

    if category is None and caller_fallback_when_uncategorized:
        should_walk_up = leaf.filename.startswith("<") or (
            is_runtime_stdlib_path(leaf.filename, rules)
            and any(
                leaf.name.startswith(prefix)
                for prefix in rules.caller_fallback_name_prefixes
            )
        )
        if should_walk_up:
            caller = first_non_runtime_file_caller(stack, rules)
            if caller is not None:
                frame_to_categorize = caller

    if category is None and folded and frame_to_categorize.filename.startswith("<"):
        for rule in rules.native_name_category_rules:
            if rule.matches(frame_to_categorize.name, frame_to_categorize.filename):
                category = rule.category
                break

    if category is None:
        category = configured_category_for(frame_to_categorize)

    if category is None:
        category = "Other"

    if category not in rules.main_simplified_categories:
        category = (
            simplify_category(category, verbose=rules.verbose, rules=rules) or "Other"
        )

    return category, frame_to_categorize, folded, folded_category
