from __future__ import annotations

import re

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Final, cast

from clankerprof.jsonio import parse_strict_yaml

RUNTIME_RULES_SCHEMA_VERSION: Final = "clankerprof.runtime_rules.v1"

_KNOWN_RULE_PACK_KEYS: Final = frozenset(
    {
        "name",
        "schema_version",
        "semantic_rules",
        "native_rules",
        "simplification_map",
        "main_simplified_categories",
        "always_foldable_categories",
        "verbose_only_foldable_categories",
        "special_namespace_prefixes",
        "stdlib_path_markers",
        "native_path_markers",
        "native_path_patterns",
        "native_path_exclude_markers",
        "native_path_exclude_patterns",
        "library_path_patterns",
        "library_selector_path_patterns",
        "library_name_suffix_patterns",
        "native_name_category_rules",
        "caller_fallback_name_prefixes",
        "legacy_caller_fallback_name_prefixes",
        "core_native_categories",
        "core_semantic_categories",
        "core_stdlib_categories",
        "core_internal_categories",
        "core_native_default_category",
        "stdlib_category",
        "internals_category",
    }
)

_KNOWN_MATCH_RULE_KEYS: Final = frozenset(
    {
        "category",
        "native_category",
        "name_contains",
        "name_prefixes",
        "name_patterns",
        "except_paths",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeMatchRule:
    category: str
    name_contains: tuple[str, ...] = ()
    name_prefixes: tuple[str, ...] = ()
    name_patterns: tuple[str, ...] = ()
    except_paths: frozenset[str] = frozenset()
    native_category: str | None = None

    def matches(self, name: str, path: str) -> bool:
        if path in self.except_paths:
            return False
        return (
            any(token in name for token in self.name_contains)
            or name.startswith(self.name_prefixes)
            or any(_match_name_pattern(pattern, name) for pattern in self.name_patterns)
        )


def _empty_string_map() -> dict[str, str]:
    return {}


def _empty_string_tuple_map() -> dict[str, tuple[str, ...]]:
    return {}


def _match_name_pattern(pattern: str, name: str) -> bool:
    resolved_pattern = pattern.removeprefix("regex:")
    try:
        return bool(re.search(resolved_pattern, name))
    except re.error as exc:
        raise ValueError(
            f"Invalid runtime rule name pattern {pattern!r}: {exc}"
        ) from exc


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
    native_name_category_rules: tuple[RuntimeMatchRule, ...] = ()
    caller_fallback_name_prefixes: tuple[str, ...] = ()
    legacy_caller_fallback_name_prefixes: tuple[str, ...] = ()
    core_native_categories: Mapping[str, str] = field(default_factory=_empty_string_map)
    core_semantic_categories: Mapping[str, str] = field(
        default_factory=_empty_string_map
    )
    core_stdlib_categories: Mapping[str, str] = field(default_factory=_empty_string_map)
    core_internal_categories: Mapping[str, str] = field(
        default_factory=_empty_string_map
    )
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


def _aliased_string_tuple(
    payload: Mapping[str, object],
    key: str,
    legacy_key: str,
) -> tuple[str, ...]:
    value = _string_tuple(payload, key)
    legacy_value = _string_tuple(payload, legacy_key)
    if value and legacy_value and value != legacy_value:
        raise ValueError(
            f"Runtime rule fields {key} and {legacy_key} are aliases; use only one."
        )
    return value or legacy_value


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
        unknown_keys = sorted(set(raw_item) - _KNOWN_MATCH_RULE_KEYS)
        if unknown_keys:
            raise ValueError(f"Unknown {key} entry keys: {', '.join(unknown_keys)}.")
        if "category" not in raw_item:
            raise ValueError(f"Each {key} entry must include category.")
        name_patterns = _string_tuple(raw_item, "name_patterns")
        for pattern in name_patterns:
            # Eager validation at load, matching the Rust port: a bad pattern
            # fails when the pack is read, not when a frame first reaches it.
            try:
                re.compile(pattern.removeprefix("regex:"))
            except re.error as exc:
                raise ValueError(
                    f"Invalid runtime rule name pattern '{pattern}'."
                ) from exc
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
                name_patterns=name_patterns,
                except_paths=frozenset(_string_tuple(raw_item, "except_paths")),
            )
        )
    return tuple(rules)


def runtime_rules_from_mapping(
    payload: Mapping[str, object],
    *,
    name: str = "custom",
    core_classes: Iterable[str] = (),
    verbose: bool = False,
) -> RuntimeRuleSet:
    unknown_keys = sorted(set(payload) - _KNOWN_RULE_PACK_KEYS)
    if unknown_keys:
        raise ValueError(f"Unknown runtime rule pack keys: {', '.join(unknown_keys)}.")
    schema_version = payload.get("schema_version", RUNTIME_RULES_SCHEMA_VERSION)
    if schema_version != RUNTIME_RULES_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported runtime rules schema version: "
            f"{schema_version!r}; expected {RUNTIME_RULES_SCHEMA_VERSION!r}."
        )
    caller_fallback_name_prefixes = _aliased_string_tuple(
        payload,
        "caller_fallback_name_prefixes",
        "legacy_caller_fallback_name_prefixes",
    )
    return RuntimeRuleSet(
        name=str(payload.get("name", name)),
        core_classes=frozenset(core_classes),
        verbose=verbose,
        enabled=True,
        semantic_rules=_load_match_rules(payload, "semantic_rules"),
        native_rules=_load_match_rules(payload, "native_rules"),
        simplification_map=_string_map(payload, "simplification_map"),
        main_simplified_categories=frozenset(
            _string_tuple(payload, "main_simplified_categories")
        ),
        always_foldable_categories=frozenset(
            _string_tuple(payload, "always_foldable_categories")
        ),
        verbose_only_foldable_categories=frozenset(
            _string_tuple(payload, "verbose_only_foldable_categories")
        ),
        special_namespace_prefixes=frozenset(
            _string_tuple(payload, "special_namespace_prefixes")
        ),
        stdlib_path_markers=_string_tuple(payload, "stdlib_path_markers"),
        native_path_markers=_string_tuple(payload, "native_path_markers"),
        native_path_patterns=_string_tuple(payload, "native_path_patterns"),
        native_path_exclude_markers=_string_tuple(
            payload, "native_path_exclude_markers"
        ),
        native_path_exclude_patterns=_string_tuple(
            payload, "native_path_exclude_patterns"
        ),
        library_path_patterns=_string_tuple(payload, "library_path_patterns"),
        library_selector_path_patterns=_string_tuple_map(
            payload,
            "library_selector_path_patterns",
        ),
        library_name_suffix_patterns=_string_tuple(
            payload, "library_name_suffix_patterns"
        ),
        native_name_category_rules=_load_match_rules(
            payload,
            "native_name_category_rules",
        ),
        caller_fallback_name_prefixes=caller_fallback_name_prefixes,
        legacy_caller_fallback_name_prefixes=caller_fallback_name_prefixes,
        core_native_categories=_string_map(payload, "core_native_categories"),
        core_semantic_categories=_string_map(payload, "core_semantic_categories"),
        core_stdlib_categories=_string_map(payload, "core_stdlib_categories"),
        core_internal_categories=_string_map(payload, "core_internal_categories"),
        core_native_default_category=str(
            payload.get("core_native_default_category", "Runtime Core (Native)")
        ),
        stdlib_category=str(payload.get("stdlib_category", "Runtime Stdlib")),
        internals_category=str(payload.get("internals_category", "Runtime Internals")),
    )


def load_runtime_rules(
    name: str,
    *,
    core_classes: Iterable[str] = (),
    verbose: bool = False,
) -> RuntimeRuleSet:
    resource = resources.files("clankerprof.runtime_rules").joinpath(f"{name}.yml")
    payload = parse_strict_yaml(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime rule pack {name} must be a YAML object.")
    return runtime_rules_from_mapping(
        cast(Mapping[str, object], payload),
        name=name,
        core_classes=core_classes,
        verbose=verbose,
    )


def load_runtime_rules_file(
    path: str | Path,
    *,
    core_classes: Iterable[str] = (),
    verbose: bool = False,
) -> RuntimeRuleSet:
    rules_path = Path(path)
    payload = parse_strict_yaml(rules_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime rule file {rules_path} must be a YAML object.")
    return runtime_rules_from_mapping(
        cast(Mapping[str, object], payload),
        name=rules_path.stem,
        core_classes=core_classes,
        verbose=verbose,
    )
