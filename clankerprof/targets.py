"""Target projection: parent-boundary self-time attribution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from clankerprof.categorize import RuntimeCategoryCache, categorize_stack
from clankerprof.facts import ProfileFactIndex, SampleFactsInput
from clankerprof.jsonio import parse_strict_json
from clankerprof.model import Frame, Profile
from clankerprof.patterns import (
    DEFAULT_RUNTIME_RULES,
    is_runtime_stdlib_path,
    match_category_pattern,
)
from clankerprof.rules import RuntimeRuleSet
from clankerprof.stats import CategoryStats

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
    payload = parse_strict_json(Path(path).read_text(encoding="utf-8"))
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
    runtime_category_for: Callable[[Frame], str | None],
) -> tuple[str, Frame, bool, str | None]:
    def configured_category_for(frame: Frame) -> str | None:
        for configured_category, pattern in parent_config.items():
            if match_category_pattern(
                pattern,
                frame.filename,
                options.runtime_rules,
            ):
                return configured_category
        return None

    return categorize_stack(
        stack,
        rules=options.runtime_rules,
        enhanced_runtime_categorization=options.enhanced_runtime_categorization,
        fold_runtime_internals=options.fold_runtime_internals,
        caller_fallback_when_uncategorized=(
            options.caller_fallback_when_uncategorized
            or options.legacy_no_enhanced_caller_fallback
        ),
        runtime_category_for=runtime_category_for,
        configured_category_for=configured_category_for,
    )


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
    runtime_cache = RuntimeCategoryCache(resolved_options.runtime_rules)
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
                runtime_cache.category_for,
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
