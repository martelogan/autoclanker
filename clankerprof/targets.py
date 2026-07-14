"""Target projection: parent-boundary self-time attribution."""

from __future__ import annotations

import json

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clankerprof.categorize import (
    categorize_frame,
    first_non_runtime_file_caller,
    is_internal_category_for_rules,
    should_fold_category,
    simplify_category,
)
from clankerprof.facts import ProfileFactIndex, SampleFactsInput
from clankerprof.model import CategoryStats, Frame, Profile
from clankerprof.patterns import (
    DEFAULT_RUNTIME_RULES,
    is_runtime_stdlib_path,
    match_category_pattern,
)
from clankerprof.rules import RuntimeRuleSet

OutputMode = Literal["text", "csv", "json", "simple-csv"]


@dataclass(frozen=True, slots=True)
class TargetAnalysisOptions:
    runtime_rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES
    enhanced_runtime_categorization: bool = True
    fold_runtime_internals: bool = False
    track_semantic_callers: bool = False
    attributables: Mapping[str, Mapping[str, float]] | None = None
    caller_fallback_when_uncategorized: bool = False
    legacy_no_enhanced_caller_fallback: bool = False


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

    if should_fold_category(
        category,
        stack,
        options.runtime_rules,
        options.fold_runtime_internals,
    ):
        for caller in stack[1:]:
            caller_category = categorize_frame(caller, options.runtime_rules)
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
