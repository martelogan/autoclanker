from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from typing import cast

import yaml


@dataclass(frozen=True, slots=True)
class RuntimeMatchRule:
    category: str
    name_contains: tuple[str, ...] = ()
    name_prefixes: tuple[str, ...] = ()
    except_paths: frozenset[str] = frozenset()
    native_category: str | None = None

    def matches(self, name: str, path: str) -> bool:
        if path in self.except_paths:
            return False
        return any(token in name for token in self.name_contains) or name.startswith(
            self.name_prefixes
        )


def _empty_string_map() -> dict[str, str]:
    return {}


def _empty_string_tuple_map() -> dict[str, tuple[str, ...]]:
    return {}


@dataclass(frozen=True, slots=True)
class RuntimeRuleSet:
    name: str
    core_classes: frozenset[str] = frozenset()
    verbose: bool = False
    enabled: bool = False
    semantic_rules: tuple[RuntimeMatchRule, ...] = ()
    native_rules: tuple[RuntimeMatchRule, ...] = ()
    simplification_map: Mapping[str, str] = field(default_factory=_empty_string_map)
    main_simplified_categories: frozenset[str] = frozenset()
    always_foldable_categories: frozenset[str] = frozenset()
    verbose_only_foldable_categories: frozenset[str] = frozenset()
    special_namespace_prefixes: frozenset[str] = frozenset()
    stdlib_path_markers: tuple[str, ...] = ()
    native_path_markers: tuple[str, ...] = ()
    native_path_patterns: tuple[str, ...] = ()
    native_path_exclude_markers: tuple[str, ...] = ()
    native_path_exclude_patterns: tuple[str, ...] = ()
    library_path_patterns: tuple[str, ...] = ()
    library_selector_path_patterns: Mapping[str, tuple[str, ...]] = field(
        default_factory=_empty_string_tuple_map
    )
    library_name_suffix_patterns: tuple[str, ...] = ()
    legacy_caller_fallback_name_prefixes: tuple[str, ...] = ()
    core_native_categories: Mapping[str, str] = field(default_factory=_empty_string_map)
    core_native_default_category: str = "Runtime Core (Native)"
    stdlib_category: str = "Runtime Stdlib"
    internals_category: str = "Runtime Internals"


def _string_tuple(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Runtime rule field {key} must be an array.")
    return tuple(str(item) for item in cast(list[object], value))


def _string_map(payload: Mapping[str, object], key: str) -> dict[str, str]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Runtime rule field {key} must be an object.")
    return {
        str(raw_key): str(raw_value)
        for raw_key, raw_value in cast(dict[object, object], value).items()
    }


def _string_tuple_map(
    payload: Mapping[str, object],
    key: str,
) -> dict[str, tuple[str, ...]]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Runtime rule field {key} must be an object.")
    result: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_patterns in cast(dict[object, object], value).items():
        if raw_patterns is None:
            result[str(raw_key)] = ()
            continue
        if not isinstance(raw_patterns, list):
            raise ValueError(f"Runtime rule field {key}.{raw_key} must be an array.")
        result[str(raw_key)] = tuple(
            str(item) for item in cast(list[object], raw_patterns)
        )
    return result


def _load_match_rules(
    payload: Mapping[str, object], key: str
) -> tuple[RuntimeMatchRule, ...]:
    value = payload.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"Runtime rule field {key} must be an array.")
    rules: list[RuntimeMatchRule] = []
    for item in cast(list[object], value):
        if not isinstance(item, dict):
            raise ValueError(f"Each {key} entry must be an object.")
        raw_item = cast(dict[str, object], item)
        if "category" not in raw_item:
            raise ValueError(f"Each {key} entry must include category.")
        rules.append(
            RuntimeMatchRule(
                category=str(raw_item["category"]),
                native_category=(
                    str(raw_item["native_category"])
                    if raw_item.get("native_category") is not None
                    else None
                ),
                name_contains=_string_tuple(raw_item, "name_contains"),
                name_prefixes=_string_tuple(raw_item, "name_prefixes"),
                except_paths=frozenset(_string_tuple(raw_item, "except_paths")),
            )
        )
    return tuple(rules)


def load_runtime_rules(
    name: str,
    *,
    core_classes: Iterable[str] = (),
    verbose: bool = False,
) -> RuntimeRuleSet:
    resource = resources.files("clankerprof.runtime_rules").joinpath(f"{name}.yml")
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime rule pack {name} must be a YAML object.")
    raw = cast(dict[str, object], payload)
    return RuntimeRuleSet(
        name=str(raw.get("name", name)),
        core_classes=frozenset(core_classes),
        verbose=verbose,
        enabled=True,
        semantic_rules=_load_match_rules(raw, "semantic_rules"),
        native_rules=_load_match_rules(raw, "native_rules"),
        simplification_map=_string_map(raw, "simplification_map"),
        main_simplified_categories=frozenset(
            _string_tuple(raw, "main_simplified_categories")
        ),
        always_foldable_categories=frozenset(
            _string_tuple(raw, "always_foldable_categories")
        ),
        verbose_only_foldable_categories=frozenset(
            _string_tuple(raw, "verbose_only_foldable_categories")
        ),
        special_namespace_prefixes=frozenset(
            _string_tuple(raw, "special_namespace_prefixes")
        ),
        stdlib_path_markers=_string_tuple(raw, "stdlib_path_markers"),
        native_path_markers=_string_tuple(raw, "native_path_markers"),
        native_path_patterns=_string_tuple(raw, "native_path_patterns"),
        native_path_exclude_markers=_string_tuple(raw, "native_path_exclude_markers"),
        native_path_exclude_patterns=_string_tuple(raw, "native_path_exclude_patterns"),
        library_path_patterns=_string_tuple(raw, "library_path_patterns"),
        library_selector_path_patterns=_string_tuple_map(
            raw,
            "library_selector_path_patterns",
        ),
        library_name_suffix_patterns=_string_tuple(raw, "library_name_suffix_patterns"),
        legacy_caller_fallback_name_prefixes=_string_tuple(
            raw,
            "legacy_caller_fallback_name_prefixes",
        ),
        core_native_categories=_string_map(raw, "core_native_categories"),
        core_native_default_category=str(
            raw.get("core_native_default_category", "Runtime Core (Native)")
        ),
        stdlib_category=str(raw.get("stdlib_category", "Runtime Stdlib")),
        internals_category=str(raw.get("internals_category", "Runtime Internals")),
    )
