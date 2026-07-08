from __future__ import annotations

import argparse
import json
import sys
import tomllib

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from clankerprof import __version__
from clankerprof.analysis import (
    DEFAULT_LIBRARY_SELECTORS,
    DEFAULT_RUNTIME_RULES,
    AttributionRule,
    BoundaryAnalysisOptions,
    BoundaryCategoryDefinition,
    BoundaryCountMode,
    BoundaryDefinition,
    BoundaryDomainDefinition,
    FramePredicate,
    FramePredicateExpr,
    JsonValue,
    RuntimeRuleSet,
    SliceAnalysisOptions,
    SliceDefinition,
    TargetAnalysisOptions,
    analyze_boundary_facts,
    analyze_slice_facts,
    analyze_target_facts,
    frame_predicate_expr_leaf_predicates,
    load_default_ruby_core_classes,
    load_json_mapping,
    load_ruby_core_classes,
    parse_frame_predicate,
    parse_frame_predicates,
    ruby_rules,
    runtime_rules_from_file,
)
from clankerprof.compare import CompareOptions, compare_json
from clankerprof.facts import (
    SampleFactsInput,
    read_sample_facts,
    sample_facts_to_jsonable,
)
from clankerprof.model import CategoryStats
from clankerprof.proto import load_profile
from clankerprof.render import (
    render_boundary_json,
    render_json_payload,
    render_semantic_callers_csv,
    render_slice_json,
    render_target_csv,
    render_target_json,
    render_target_text,
)


def _load_attributables(path: str | None) -> dict[str, dict[str, float]] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Attributables must be a JSON object.")
    result: dict[str, dict[str, float]] = {}
    for name, values in cast(dict[object, object], payload).items():
        if not isinstance(values, dict):
            raise ValueError(f"Attributable column {name} must be an object.")
        raw_values = cast(dict[object, object], values)
        result[str(name)] = {
            str(key): float(cast(Any, value)) for key, value in raw_values.items()
        }
    return result


def _load_projection_input(
    profile_path: str | None,
    facts_path: str | None,
) -> SampleFactsInput:
    if profile_path is not None and facts_path is not None:
        raise ValueError("--profile and --facts are mutually exclusive.")
    if facts_path is not None:
        return read_sample_facts(facts_path)
    if profile_path is None:
        raise ValueError("--profile or --facts is required.")
    return load_profile(profile_path).to_sample_facts()


def _load_slices_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path)
    if config_path.suffix == ".toml":
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Slice config file must be a YAML object.")
    return cast(dict[str, object], payload)


def _string_array(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{key} in slice config must be an array.")
    return tuple(str(item) for item in cast(list[object], value))


def _optional_bool(payload: dict[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} in slice config must be a boolean.")
    return value


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} in slice config must be an integer.")
    return int(cast(Any, value))


def _optional_by_slice(payload: dict[str, object]) -> str | None:
    value = payload.get("by_slice")
    if value is None:
        return None
    if isinstance(value, bool):
        return "0.1%" if value else None
    if isinstance(value, int | float):
        if isinstance(value, float) and not value.is_integer():
            return f"{value}%"
        return str(int(value))
    return str(value)


def _optional_unattributed_libraries(payload: dict[str, object]) -> int | None:
    values = {
        key: payload[key]
        for key in ("unattributed_gems", "unattributed_libraries")
        if key in payload and payload[key] is not None
    }
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(
            "unattributed_gems and unattributed_libraries are aliases; use only one."
        )
    value = next(iter(values.values()))
    if isinstance(value, bool):
        return 2**63 - 1 if value else None
    return int(cast(Any, value))


def _focus_slices(values: Sequence[str]) -> frozenset[str]:
    result: set[str] = set()
    for value in values:
        result.update(part for part in value.split(",") if part)
    return frozenset(result)


def _merge_single_value(
    cli_value: str | None,
    config_value: object,
    *,
    name: str,
) -> str | None:
    if cli_value is not None and config_value is not None:
        raise ValueError(f"{name} specified both on command line and in config file.")
    if cli_value is not None:
        return cli_value
    if config_value is None:
        return None
    return str(config_value)


def _uses_slice_reference(raw_filter: str) -> bool:
    body = raw_filter
    while body.startswith(("!", "<")):
        body = body[1:]
    return body.startswith("slice:")


def _needs_slices(
    *,
    slices_path: str | None,
    filters: Sequence[str],
    collapse: Sequence[str],
    attributes: Sequence[AttributionRule],
    by_slice: str | None,
) -> bool:
    if slices_path is not None or by_slice is not None or attributes:
        return True
    return any(_uses_slice_reference(item) for item in (*filters, *collapse))


def _filter_body(raw_filter: str) -> str:
    body = raw_filter
    while body.startswith(("!", "<")):
        body = body[1:]
    return body


def _validate_slice_options(options: SliceAnalysisOptions) -> None:
    names = {item.name for item in options.slices}
    valid_filter_keys = _valid_filter_keys(options.runtime_rules)
    if not options.slices:
        if options.by_slice is not None:
            raise ValueError("--by-slice requires --slices=<file>.")
        if options.attributes:
            raise ValueError("--attribute requires --slices=<file>.")
        if any(
            _uses_slice_reference(item)
            for item in (*options.filters, *options.collapse)
        ):
            raise ValueError("slice:... requires --slices=<file>.")
        return
    for raw_filter in (*options.filters, *options.collapse):
        body = _filter_body(raw_filter)
        key, _, value = body.partition(":")
        if not key or not value:
            raise ValueError(f"Filter must be '<key>:<value>': {raw_filter}")
        if key not in valid_filter_keys:
            raise ValueError(f"Unsupported filter key: {key}")
        if key == "slice" and value not in names:
            raise ValueError(f"Unknown slice: {value}")
    for raw_collapse in options.collapse:
        if raw_collapse.startswith(("!", "<")):
            raise ValueError(
                f"Collapse filters do not support prefixes: {raw_collapse}"
            )
    seen_attributes: set[tuple[str, str]] = set()
    valid_attribute_keys = valid_filter_keys - {"slice"}
    for attribute in options.attributes:
        key = (attribute.key, attribute.value)
        if attribute.key not in valid_attribute_keys:
            raise ValueError(f"Unsupported attribute filter key: {attribute.key}")
        if (
            attribute.target_slice not in names
            and not options.allow_virtual_attribute_slices
        ):
            raise ValueError(f"Unknown slice in --attribute: {attribute.target_slice}")
        if key in seen_attributes:
            raise ValueError(
                f"Duplicate attribute rule filter: {attribute.key}:{attribute.value}"
            )
        seen_attributes.add(key)


def _valid_filter_keys(rules: RuntimeRuleSet) -> frozenset[str]:
    return frozenset(
        {
            "name",
            "path",
            "slice",
            *DEFAULT_LIBRARY_SELECTORS,
            *rules.library_selector_path_patterns,
        }
    )


def _runtime_rules(args: argparse.Namespace) -> RuntimeRuleSet:
    runtime = str(getattr(args, "runtime", "generic"))
    no_enhanced = bool(getattr(args, "no_enhanced", False))
    runtime_rules_path = getattr(args, "runtime_rules", None)
    core_classes: frozenset[str] = (
        load_ruby_core_classes(args.ruby_core_classes)
        if args.ruby_core_classes is not None
        else load_default_ruby_core_classes()
        if runtime == "ruby"
        else frozenset[str]()
    )
    if runtime_rules_path is not None:
        return runtime_rules_from_file(
            runtime_rules_path,
            core_classes=core_classes,
            verbose=bool(args.verbose_runtime_internals),
        )
    if runtime == "generic" and not no_enhanced:
        return DEFAULT_RUNTIME_RULES
    if runtime == "generic" and no_enhanced:
        core_classes = load_default_ruby_core_classes()
    elif runtime != "ruby":
        raise ValueError(f"Unsupported runtime: {runtime}")
    return ruby_rules(
        core_classes,
        verbose=bool(args.verbose_runtime_internals),
    )


def _use_compat_target_csv_layout(args: argparse.Namespace) -> bool:
    layout = getattr(args, "target_csv_layout", None)
    legacy_layout = bool(getattr(args, "legacy_target_csv_layout", False))
    if legacy_layout and layout == "standard":
        raise ValueError(
            "--legacy-target-csv-layout conflicts with --target-csv-layout=standard."
        )
    return legacy_layout or layout == "compat"


def _write_legacy_target_csv_artifacts(
    output_path: str,
    results: Mapping[str, Mapping[str, CategoryStats]],
    attributables: dict[str, dict[str, float]] | None,
) -> dict[str, Any]:
    base_name = Path(output_path).name
    output_dir = Path("output")
    verbose_dir = output_dir / "verbose"
    verbose_dir.mkdir(parents=True, exist_ok=True)

    simplified_path = output_dir / base_name
    verbose_path = verbose_dir / base_name
    simplified_path.write_text(
        render_target_csv(
            results,
            attributables=attributables,
            simplified=True,
            legacy_layout=True,
        ),
        encoding="utf-8",
    )
    verbose_path.write_text(
        render_target_csv(
            results,
            attributables=attributables,
            simplified=False,
            legacy_layout=True,
        ),
        encoding="utf-8",
    )
    return {
        "tool": "clankerprof_targets",
        "ok": True,
        "output": str(simplified_path),
        "compat_target_csv_layout": True,
        "legacy_target_csv_layout": True,
        "outputs": {
            "simplified_csv": str(simplified_path),
            "verbose_csv": str(verbose_path),
        },
    }


def _parse_attribute(raw: str) -> AttributionRule:
    filter_part, separator, target = raw.partition(",to:")
    if not separator or not target:
        raise ValueError(f"Attribute rule must be '<filter>,to:<slice>': {raw}")
    if filter_part.startswith("!"):
        raise ValueError(f"Attribute rules do not support '!': {raw}")
    descendant = filter_part.startswith("<")
    body = filter_part[1:] if descendant else filter_part
    if body.startswith("!"):
        raise ValueError(f"Attribute rules do not support '!': {raw}")
    key, separator, value = body.partition(":")
    if not separator:
        raise ValueError(f"Attribute rule filter must be '<key>:<value>': {raw}")
    if key == "slice":
        raise ValueError(f"Attribute rules do not support slice: filters: {raw}")
    return AttributionRule(
        key=key,
        value=value,
        target_slice=target,
        descendant=descendant,
    )


def _load_slices(path: str | None) -> tuple[SliceDefinition, ...]:
    if path is None:
        return ()
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Slices file must be a YAML object.")
    raw_payload = cast(dict[object, object], payload)
    raw_slices = raw_payload.get("slices", [])
    if not isinstance(raw_slices, list):
        raise ValueError("Slices file must contain a slices array.")
    slices: list[SliceDefinition] = []
    for item in cast(list[object], raw_slices):
        if not isinstance(item, dict):
            raise ValueError("Each slice entry must be an object.")
        raw_item = cast(dict[object, object], item)
        paths = raw_item.get("paths", [])
        if paths is None:
            paths = []
        if not isinstance(paths, list):
            raise ValueError("Slice paths must be an array.")
        if "name" not in raw_item:
            raise ValueError("Each slice entry must include a name.")
        metadata = _slice_metadata(raw_item)
        slices.append(
            SliceDefinition(
                name=str(raw_item["name"]),
                path_patterns=tuple(str(path) for path in cast(list[object], paths)),
                is_default=bool(raw_item.get("default", False)),
                metadata=metadata,
            )
        )
    return tuple(slices)


def _load_config_payload(path: str | Path, *, description: str) -> dict[str, object]:
    config_path = Path(path)
    if config_path.suffix == ".toml":
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} config file must be an object.")
    return cast(dict[str, object], payload)


def _aliased_config_value(
    payload: dict[str, object],
    *names: str,
    description: str,
) -> tuple[str, object | None]:
    present = [name for name in names if name in payload]
    if len(present) > 1:
        formatted = " or ".join(f"[{name}]" for name in names)
        raise ValueError(f"Use only one of {formatted} for {description}.")
    if not present:
        return names[0], None
    name = present[0]
    return name, payload[name]


def _aliased_mapping_value(
    payload: Mapping[object, object],
    *names: str,
    description: str,
) -> tuple[str, object | None]:
    present = [name for name in names if name in payload]
    if len(present) > 1:
        formatted = " or ".join(names)
        raise ValueError(f"Use only one of {formatted} for {description}.")
    if not present:
        return names[0], None
    name = present[0]
    return name, payload[name]


def _string_values(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in cast(list[object], value))
    raise ValueError(f"{field_name} must be a string or array of strings.")


def _predicate_expr_children(
    value: object,
    *,
    field_name: str,
    default_key: str,
) -> tuple[FramePredicateExpr, ...]:
    if isinstance(value, str):
        return (
            FramePredicateExpr(
                predicates=(parse_frame_predicate(value, default_key=default_key),)
            ),
        )
    if isinstance(value, list):
        children: list[FramePredicateExpr] = []
        for index, item in enumerate(cast(list[object], value)):
            if isinstance(item, str):
                children.append(
                    FramePredicateExpr(
                        predicates=(
                            parse_frame_predicate(item, default_key=default_key),
                        )
                    )
                )
                continue
            if isinstance(item, dict):
                children.append(
                    _predicate_expr(
                        cast(dict[object, object], item),
                        field_name=f"{field_name}[{index}]",
                        default_key=default_key,
                    )
                )
                continue
            raise ValueError(
                f"{field_name}[{index}] must be a string or predicate table."
            )
        return tuple(children)
    if isinstance(value, dict):
        return (
            _predicate_expr(
                cast(dict[object, object], value),
                field_name=field_name,
                default_key=default_key,
            ),
        )
    raise ValueError(f"{field_name} must be a string, array, or predicate table.")


def _predicate_expr(
    value: object,
    *,
    field_name: str,
    default_key: str,
) -> FramePredicateExpr:
    if isinstance(value, str):
        return FramePredicateExpr(
            predicates=(parse_frame_predicate(value, default_key=default_key),)
        )
    if isinstance(value, list):
        raw_list = cast(list[object], value)
        return FramePredicateExpr(
            predicates=parse_frame_predicates(
                _string_values(raw_list, field_name=field_name),
                default_key=default_key,
            )
        )
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a string, array, or predicate table.")

    raw_value = cast(dict[object, object], value)
    allowed_keys = {"patterns", "match", "selector", "any", "all", "not"}
    unknown_keys = sorted(str(key) for key in raw_value if key not in allowed_keys)
    if unknown_keys:
        raise ValueError(
            f"{field_name} contains unsupported predicate keys: "
            f"{', '.join(unknown_keys)}."
        )
    selector_keys = [
        key for key in ("patterns", "match", "selector") if key in raw_value
    ]
    if len(selector_keys) > 1:
        raise ValueError(
            f"{field_name} must use only one of patterns, match, or selector."
        )

    predicates: tuple[FramePredicate, ...] = ()
    if "patterns" in raw_value:
        predicates = parse_frame_predicates(
            _string_values(raw_value["patterns"], field_name=f"{field_name}.patterns"),
            default_key=default_key,
        )
    if "match" in raw_value:
        predicates = parse_frame_predicates(
            _string_values(raw_value["match"], field_name=f"{field_name}.match"),
            default_key=default_key,
        )
    if "selector" in raw_value:
        predicates = parse_frame_predicates(
            _string_values(
                raw_value["selector"],
                field_name=f"{field_name}.selector",
            ),
            default_key=default_key,
        )
    any_expr = (
        _predicate_expr_children(
            raw_value["any"],
            field_name=f"{field_name}.any",
            default_key=default_key,
        )
        if "any" in raw_value
        else ()
    )
    all_expr = (
        _predicate_expr_children(
            raw_value["all"],
            field_name=f"{field_name}.all",
            default_key=default_key,
        )
        if "all" in raw_value
        else ()
    )
    not_expr = (
        _predicate_expr_children(
            raw_value["not"],
            field_name=f"{field_name}.not",
            default_key=default_key,
        )
        if "not" in raw_value
        else ()
    )
    if not (predicates or any_expr or all_expr or not_expr):
        raise ValueError(f"{field_name} predicate table cannot be empty.")
    return FramePredicateExpr(
        predicates=predicates,
        any=any_expr,
        all=all_expr,
        not_=not_expr,
    )


def _load_boundary_categories(
    raw_table: object,
    *,
    section_name: str = "category",
) -> tuple[BoundaryCategoryDefinition, ...]:
    if raw_table is None:
        return ()
    if not isinstance(raw_table, dict):
        raise ValueError(f"[{section_name}] must be an object.")
    categories: list[BoundaryCategoryDefinition] = []
    seen: set[str] = set()
    for raw_name, raw_value in cast(dict[object, object], raw_table).items():
        name = str(raw_name)
        if name in seen:
            raise ValueError(f"Duplicate category label: {name}")
        seen.add(name)
        categories.append(
            BoundaryCategoryDefinition(
                name=name,
                predicates=_predicate_expr(
                    raw_value,
                    field_name=f"{section_name} {name}",
                    default_key="path",
                ),
            )
        )
    return tuple(categories)


def _load_boundary_domains(
    raw_table: object,
    *,
    section_name: str = "domain",
) -> tuple[BoundaryDomainDefinition, ...]:
    if raw_table is None:
        return ()
    if not isinstance(raw_table, dict):
        raise ValueError(f"[{section_name}] must be an object.")
    domains: list[BoundaryDomainDefinition] = []
    seen: set[str] = set()
    for raw_name, raw_value in cast(dict[object, object], raw_table).items():
        name = str(raw_name)
        if name in seen:
            raise ValueError(f"Duplicate domain label: {name}")
        seen.add(name)
        fallback = False
        if isinstance(raw_value, dict):
            raw_domain_mapping = cast(dict[object, object], raw_value)
            fallback = bool(raw_domain_mapping.get("fallback", False))
            raw_domain_predicate = {
                key: value
                for key, value in raw_domain_mapping.items()
                if key != "fallback"
            }
            expression = _predicate_expr(
                raw_domain_predicate,
                field_name=f"{section_name} {name}",
                default_key="path",
            )
        else:
            expression = _predicate_expr(
                raw_value,
                field_name=f"{section_name} {name}",
                default_key="path",
            )
        domains.append(
            BoundaryDomainDefinition(
                name=name,
                predicates=expression,
                fallback=fallback,
            )
        )
    return tuple(domains)


def _load_boundary_bucket(
    raw_value: object,
    *,
    field_name: str = "boundary.bucket",
) -> dict[str, tuple[str, ...]]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"{field_name} must be an object.")
    buckets: dict[str, tuple[str, ...]] = {}
    category_to_bucket: dict[str, str] = {}
    for raw_name, raw_categories in cast(dict[object, object], raw_value).items():
        name = str(raw_name)
        categories = _string_values(
            raw_categories,
            field_name=f"{field_name} {name}",
        )
        for category in categories:
            if category in category_to_bucket:
                raise ValueError(
                    f"Category {category} appears in both "
                    f"{category_to_bucket[category]} and {name} buckets."
                )
            category_to_bucket[category] = name
        buckets[name] = categories
    return buckets


def _load_boundary_attributables(
    raw_value: object,
    *,
    field_name: str = "boundary.attributables",
) -> dict[str, float]:
    if raw_value is None:
        return {}
    if not isinstance(raw_value, dict):
        raise ValueError(f"{field_name} must be an object.")
    result: dict[str, float] = {}
    for raw_name, raw_metric in cast(dict[object, object], raw_value).items():
        metric = float(cast(Any, raw_metric))
        if metric < 0:
            raise ValueError(f"Boundary attributable {raw_name} cannot be negative.")
        result[str(raw_name)] = metric
    return result


def _boundary_predicate_value(raw_boundary: dict[object, object]) -> object:
    if "selector" in raw_boundary:
        return raw_boundary["selector"]
    if "matcher" in raw_boundary:
        return raw_boundary["matcher"]
    if "match" in raw_boundary:
        return raw_boundary["match"]
    if "function" in raw_boundary:
        raw_function = raw_boundary["function"]
        if isinstance(raw_function, list):
            return [f"name_eq:{item}" for item in cast(list[object], raw_function)]
        return f"name_eq:{raw_function}"
    raise ValueError("Each scope must define selector, matcher, match, or function.")


def _boundary_label_fallback(raw_predicates: object) -> str:
    if isinstance(raw_predicates, str):
        return raw_predicates
    if isinstance(raw_predicates, list):
        raw_values = cast(list[object], raw_predicates)
        if raw_values:
            return str(raw_values[0])
    return "boundary"


def _load_boundaries(
    raw_value: object,
    *,
    section_name: str = "boundary",
) -> tuple[BoundaryDefinition, ...]:
    if not isinstance(raw_value, list):
        raise ValueError(f"Scope config must contain a {section_name} array.")
    boundaries: list[BoundaryDefinition] = []
    seen_names: set[str] = set()
    for item in cast(list[object], raw_value):
        if not isinstance(item, dict):
            raise ValueError("Each boundary entry must be an object.")
        raw_boundary = cast(dict[object, object], item)
        raw_predicates = _boundary_predicate_value(raw_boundary)
        label = str(
            raw_boundary.get("label")
            or raw_boundary.get("name")
            or raw_boundary.get("function")
            or _boundary_label_fallback(raw_predicates)
        )
        if label in seen_names:
            raise ValueError(f"Duplicate boundary label: {label}")
        seen_names.add(label)
        count = str(raw_boundary.get("count", "occurrence"))
        if count not in {"occurrence", "once_per_sample"}:
            raise ValueError(
                f"{section_name}.count must be occurrence or once_per_sample."
            )
        rollup_name, raw_rollup = _aliased_mapping_value(
            raw_boundary,
            "rollup",
            "bucket",
            description=f"{section_name} {label} rollup",
        )
        boundaries.append(
            BoundaryDefinition(
                name=label,
                predicates=_predicate_expr(
                    raw_predicates,
                    field_name=f"{section_name} {label}",
                    default_key="name_eq",
                ),
                buckets=_load_boundary_bucket(
                    raw_rollup,
                    field_name=f"{section_name}.{rollup_name}",
                ),
                attributables=_load_boundary_attributables(
                    raw_boundary.get("attributables"),
                    field_name=f"{section_name}.attributables",
                ),
                exclude_descendants=_predicate_expr(
                    raw_boundary.get("exclude_descendants", []),
                    field_name=f"{section_name} {label} exclude_descendants",
                    default_key="name_eq",
                ),
                count=cast(BoundaryCountMode, count),
            )
        )
    return tuple(boundaries)


def _uses_slice_predicate_in_boundary_options(
    options: BoundaryAnalysisOptions,
) -> bool:
    predicates = [
        predicate
        for category in options.categories
        for predicate in frame_predicate_expr_leaf_predicates(category.predicates)
    ]
    predicates.extend(
        predicate
        for domain in options.domains
        for predicate in frame_predicate_expr_leaf_predicates(domain.predicates)
    )
    for boundary in options.boundaries:
        predicates.extend(frame_predicate_expr_leaf_predicates(boundary.predicates))
        predicates.extend(
            frame_predicate_expr_leaf_predicates(boundary.exclude_descendants)
        )
    return any(predicate.key == "slice" for predicate in predicates)


def _load_boundary_options(
    path: str | Path,
    *,
    runtime_rules: RuntimeRuleSet,
) -> BoundaryAnalysisOptions:
    payload = _load_config_payload(path, description="Boundary")
    slices_path = payload.get("slices")
    slices: tuple[SliceDefinition, ...] = ()
    if slices_path is not None:
        raw_slices_path = Path(str(slices_path))
        if not raw_slices_path.is_absolute():
            raw_slices_path = Path(path).parent / raw_slices_path
        slices = _load_slices(str(raw_slices_path))
    category_name, raw_categories = _aliased_config_value(
        payload,
        "cost_kind",
        "category",
        description="cost-kind definitions",
    )
    domain_name, raw_domains = _aliased_config_value(
        payload,
        "owner",
        "domain",
        description="owner definitions",
    )
    scope_name, raw_scopes = _aliased_config_value(
        payload,
        "scope",
        "boundary",
        description="scope definitions",
    )
    options = BoundaryAnalysisOptions(
        runtime_rules=runtime_rules,
        categories=_load_boundary_categories(
            raw_categories,
            section_name=category_name,
        ),
        domains=_load_boundary_domains(raw_domains, section_name=domain_name),
        boundaries=_load_boundaries(raw_scopes, section_name=scope_name),
        slices=slices,
    )
    if _uses_slice_predicate_in_boundary_options(options) and not options.slices:
        raise ValueError("slice: predicates in boundary config require slices.")
    return options


def _json_compatible(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_compatible(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in cast(dict[object, object], value).items()
        }
    return str(value)


def _slice_metadata(raw_item: dict[object, object]) -> dict[str, JsonValue]:
    reserved = {"name", "paths", "default"}
    metadata: dict[str, JsonValue] = {}
    for key, value in raw_item.items():
        key_text = str(key)
        if key_text in reserved:
            continue
        if key_text == "metadata" and isinstance(value, dict):
            for nested_key, nested_value in cast(dict[object, object], value).items():
                metadata[str(nested_key)] = _json_compatible(nested_value)
            continue
        metadata[key_text] = _json_compatible(value)
    return metadata


def run_targets(args: argparse.Namespace) -> dict[str, Any]:
    sample_facts = _load_projection_input(args.profile, args.facts)
    config = load_json_mapping(args.config) if args.config is not None else {}
    for target in args.targets:
        config.setdefault(target, {})
    if not config:
        raise ValueError("--config or --target is required.")
    attributables = _load_attributables(args.cpu_attributables)
    runtime_rules = _runtime_rules(args)
    compat_target_csv_layout = _use_compat_target_csv_layout(args)
    results = analyze_target_facts(
        sample_facts,
        config,
        TargetAnalysisOptions(
            runtime_rules=runtime_rules,
            enhanced_runtime_categorization=not bool(args.no_enhanced),
            fold_runtime_internals=bool(args.fold_runtime_internals),
            track_semantic_callers=bool(args.track_semantic_callers),
            attributables=attributables,
            caller_fallback_when_uncategorized=bool(args.no_enhanced),
        ),
    )
    if args.semantic_callers_csv:
        if not args.track_semantic_callers:
            raise ValueError(
                "--semantic-callers-csv requires --track-semantic-callers."
            )
        Path(args.semantic_callers_csv).write_text(
            render_semantic_callers_csv(
                results,
                runtime_rules=runtime_rules,
                dependency_prefix=("gems" if compat_target_csv_layout else "deps"),
                legacy_layout=compat_target_csv_layout,
            )
            + ("" if compat_target_csv_layout else "\n"),
            encoding="utf-8",
        )
    if args.format == "json":
        return cast(dict[str, Any], render_target_json(results))
    if compat_target_csv_layout:
        if args.format != "csv" or not args.output:
            raise ValueError(
                "--target-csv-layout=compat requires --format csv and --output."
            )
        return _write_legacy_target_csv_artifacts(
            args.output,
            results,
            attributables,
        )
    rendered = (
        render_target_csv(
            results,
            attributables=attributables,
            simplified=args.format == "simple-csv",
        )
        if args.format in {"csv", "simple-csv"}
        else render_target_text(
            results,
            show_folded=bool(args.fold_runtime_internals),
            show_semantic_callers=bool(args.track_semantic_callers),
        )
    )
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        return {"tool": "clankerprof_targets", "ok": True, "output": args.output}
    print(rendered)
    return {"tool": "clankerprof_targets", "ok": True}


def run_slices(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_slices_config(args.config)
    raw_profile = _merge_single_value(
        args.profile, config.get("profile"), name="profile"
    )
    raw_facts = _merge_single_value(args.facts, config.get("facts"), name="facts")
    sample_facts = _load_projection_input(raw_profile, raw_facts)
    raw_slices = _merge_single_value(args.slices, config.get("slices"), name="slices")
    raw_top = _optional_int(config, "top")
    if args.top is not None and raw_top is not None:
        raise ValueError("top specified both on command line and in config file.")
    top = args.top if args.top is not None else raw_top
    raw_by_slice = _optional_by_slice(config)
    if args.by_slice is not None and raw_by_slice is not None:
        raise ValueError("by_slice specified both on command line and in config file.")
    by_slice = args.by_slice if args.by_slice is not None else raw_by_slice
    raw_show_paths = _optional_bool(config, "show_paths")
    raw_no_collapse_native = _optional_bool(config, "no_collapse_native")
    if args.show_paths and raw_show_paths is not None:
        raise ValueError(
            "show_paths specified both on command line and in config file."
        )
    show_paths = args.show_paths if args.show_paths else bool(raw_show_paths)
    if args.no_collapse_native and raw_no_collapse_native is not None:
        raise ValueError(
            "no_collapse_native specified both on command line and in config file."
        )
    no_collapse_native = (
        args.no_collapse_native
        if args.no_collapse_native
        else bool(raw_no_collapse_native)
    )
    raw_unattributed_libraries = _optional_unattributed_libraries(config)
    if (
        args.unattributed_libraries is not None
        and raw_unattributed_libraries is not None
    ):
        raise ValueError(
            "unattributed_libraries specified both on command line and in config file "
            "(--unattributed-gems is a compatibility alias)."
        )
    unattributed_libraries = (
        args.unattributed_libraries
        if args.unattributed_libraries is not None
        else raw_unattributed_libraries
    )
    raw_filters = (
        *_string_array(config, "filters"),
        *_string_array(config, "filter"),
        *tuple(args.filters),
    )
    raw_collapse = (*_string_array(config, "collapse"), *tuple(args.collapse))
    attributes = tuple(
        _parse_attribute(raw)
        for raw in (*_string_array(config, "attribute"), *tuple(args.attribute))
    )
    if raw_slices is None and _needs_slices(
        slices_path=None,
        filters=raw_filters,
        collapse=raw_collapse,
        attributes=attributes,
        by_slice=by_slice,
    ):
        default_slices = Path("slices.yml")
        if default_slices.exists():
            raw_slices = str(default_slices)
    options = SliceAnalysisOptions(
        runtime_rules=_runtime_rules(args),
        slices=_load_slices(raw_slices),
        filters=raw_filters,
        collapse=raw_collapse,
        attributes=attributes,
        top=top,
        by_slice=by_slice,
        show_paths=show_paths,
        no_collapse_native=no_collapse_native,
        unattributed_gems=unattributed_libraries,
        unattributed_libraries=unattributed_libraries,
        allow_virtual_attribute_slices=bool(args.allow_virtual_attribute_slices),
    )
    _validate_slice_options(options)
    result = analyze_slice_facts(
        sample_facts,
        options,
    )
    payload = cast(dict[str, Any], render_slice_json(result, options))
    if args.output:
        Path(args.output).write_text(
            render_json_payload(payload) + "\n", encoding="utf-8"
        )
        return {"tool": "clankerprof_slices", "ok": True, "output": args.output}
    return payload


def run_boundaries(args: argparse.Namespace) -> dict[str, Any]:
    sample_facts = _load_projection_input(args.profile, args.facts)
    runtime_rules = _runtime_rules(args)
    options = _load_boundary_options(args.config, runtime_rules=runtime_rules)
    options = BoundaryAnalysisOptions(
        boundaries=options.boundaries,
        categories=options.categories,
        domains=options.domains,
        slices=options.slices,
        runtime_rules=options.runtime_rules,
        enhanced_runtime_categorization=not bool(args.no_enhanced),
        fold_runtime_internals=bool(args.fold_runtime_internals),
        caller_fallback_when_uncategorized=bool(args.no_enhanced),
        legacy_no_enhanced_caller_fallback=bool(args.no_enhanced),
    )
    payload = cast(
        dict[str, Any],
        render_boundary_json(
            analyze_boundary_facts(sample_facts, options),
            top=args.top,
        ),
    )
    if args.output:
        Path(args.output).write_text(
            render_json_payload(payload) + "\n",
            encoding="utf-8",
        )
        return {"tool": "clankerprof_boundaries", "ok": True, "output": args.output}
    return payload


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Compare inputs must be JSON objects.")
    return compare_json(
        cast(dict[str, Any], before),
        cast(dict[str, Any], after),
        CompareOptions(
            threshold_abs=float(args.threshold_abs),
            threshold_rel=float(args.threshold_rel),
            focus_slices=_focus_slices(args.focus_slices),
            focus_boundaries=_focus_slices(args.focus_boundaries),
        ),
    )


def run_facts(args: argparse.Namespace) -> dict[str, Any]:
    profile = load_profile(args.profile)
    payload = sample_facts_to_jsonable(profile.to_sample_facts())
    if args.output:
        Path(args.output).write_text(
            render_json_payload(cast(dict[str, Any], payload)) + "\n",
            encoding="utf-8",
        )
        return {
            "tool": "clankerprof_facts",
            "ok": True,
            "output": args.output,
            "schema_version": payload["schema_version"],
            "summary": payload["summary"],
        }
    return cast(dict[str, Any], payload)


def register_pprof_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "pprof",
        help="Analyze pprof CPU profiles with clankerprof.",
    )
    register_commands(parser.add_subparsers(dest="pprof_command", required=True))


def register_commands(subparsers: Any) -> None:
    targets = subparsers.add_parser(
        "targets",
        help="Attribute target-function self time to configured categories.",
    )
    targets.add_argument("--profile")
    targets.add_argument(
        "--facts",
        help="Read versioned sample-facts JSON instead of a pprof profile.",
    )
    targets.add_argument(
        "--config",
        help=(
            "JSON target config mapping parent functions to category path "
            "patterns or explicit regex: patterns. Use --target for the "
            "minimal no-config path."
        ),
    )
    targets.add_argument(
        "--target",
        dest="targets",
        action="append",
        default=[],
        help=(
            "Parent function to explain without a config file. Repeat for "
            "multiple targets; uncategorized cost is reported as Other."
        ),
    )
    targets.add_argument("--output")
    targets.add_argument(
        "--format",
        choices=("text", "csv", "simple-csv", "json"),
        default="json",
    )
    targets.add_argument("--runtime", choices=("generic", "ruby"), default="generic")
    targets.add_argument(
        "--runtime-rules",
        help=(
            "Load a YAML runtime rule pack from disk. This keeps runtime or "
            "domain-specific labels outside the package while preserving the "
            "same categorization and folding machinery."
        ),
    )
    targets.add_argument(
        "--ruby-core-classes", "--core-classes", dest="ruby_core_classes"
    )
    targets.add_argument(
        "--no-enhanced",
        action="store_true",
        help=(
            "Disable runtime semantic categorization and use configured "
            "native/delegated caller fallbacks before category matching."
        ),
    )
    targets.add_argument(
        "--fold-runtime-internals",
        "--fold-ruby-internals",
        action="store_true",
        dest="fold_runtime_internals",
    )
    targets.add_argument(
        "--verbose-runtime-internals",
        "--verbose-ruby-internals",
        action="store_true",
        dest="verbose_runtime_internals",
    )
    targets.add_argument("--track-semantic-callers", action="store_true")
    targets.add_argument("--cpu-attributables", "--attributables")
    targets.add_argument("--semantic-callers-csv")
    targets.add_argument(
        "--target-csv-layout",
        choices=("standard", "compat"),
        default=None,
        help=(
            "Use standard single-file CSV output or compat two-file output under "
            "output/ and output/verbose/."
        ),
    )
    targets.add_argument(
        "--legacy-target-csv-layout",
        action="store_true",
        dest="legacy_target_csv_layout",
        help=(
            "With --format csv --output, write output/<name> and "
            "output/verbose/<name> for compatibility with older target reports."
        ),
    )
    targets.set_defaults(handler=run_targets)

    boundary_help = "Run scope/cost-kind/rollup/owner decomposition over a profile."

    def add_boundary_like_command(name: str) -> None:
        parser = subparsers.add_parser(name, help=boundary_help)
        parser.add_argument("--profile")
        parser.add_argument(
            "--facts",
            help="Read versioned sample-facts JSON instead of a pprof profile.",
        )
        parser.add_argument(
            "--config",
            required=True,
            help=(
                "TOML/YAML config with [cost_kind], optional [owner], and "
                "[[scope]] sections. Legacy [category], [domain], and "
                "[[boundary]] sections remain accepted."
            ),
        )
        parser.add_argument("--output")
        parser.add_argument("--top", type=int)
        parser.add_argument("--runtime", choices=("generic", "ruby"), default="generic")
        parser.add_argument(
            "--runtime-rules",
            help=(
                "Load a YAML runtime rule pack from disk for native/core labels, "
                "library extraction, and folding categories."
            ),
        )
        parser.add_argument(
            "--ruby-core-classes", "--core-classes", dest="ruby_core_classes"
        )
        parser.add_argument(
            "--no-enhanced",
            action="store_true",
            help=(
                "Disable runtime semantic categorization and use configured "
                "native/delegated caller fallbacks before category matching."
            ),
        )
        parser.add_argument(
            "--fold-runtime-internals",
            "--fold-ruby-internals",
            action="store_true",
            dest="fold_runtime_internals",
        )
        parser.add_argument(
            "--verbose-runtime-internals",
            "--verbose-ruby-internals",
            action="store_true",
            dest="verbose_runtime_internals",
        )
        parser.set_defaults(handler=run_boundaries)

    add_boundary_like_command("boundaries")
    add_boundary_like_command("scopes")

    slices = subparsers.add_parser(
        "slices",
        help="Run slice attribution over a pprof profile.",
    )
    slices.add_argument("--profile")
    slices.add_argument(
        "--facts",
        help="Read versioned sample-facts JSON instead of a pprof profile.",
    )
    slices.add_argument("--config")
    slices.add_argument("--slices")
    slices.add_argument("--filter", dest="filters", action="append", default=[])
    slices.add_argument("--collapse", action="append", default=[])
    slices.add_argument("--attribute", action="append", default=[])
    slices.add_argument("--top", type=int)
    slices.add_argument("--by-slice", nargs="?", const="0.1%")
    slices.add_argument("--show-paths", action="store_true")
    slices.add_argument("--no-collapse-native", action="store_true")
    slices.add_argument(
        "--unattributed-gems",
        "--unattributed-libraries",
        dest="unattributed_libraries",
        nargs="?",
        const=2**63 - 1,
        type=int,
    )
    slices.add_argument("--runtime", choices=("generic", "ruby"), default="generic")
    slices.add_argument(
        "--runtime-rules",
        help=(
            "Load a YAML runtime rule pack from disk for native/core labels, "
            "library extraction, and folding categories."
        ),
    )
    slices.add_argument(
        "--ruby-core-classes", "--core-classes", dest="ruby_core_classes"
    )
    slices.add_argument(
        "--verbose-runtime-internals",
        "--verbose-ruby-internals",
        action="store_true",
        dest="verbose_runtime_internals",
    )
    slices.add_argument("--allow-virtual-attribute-slices", action="store_true")
    slices.add_argument("--output")
    slices.set_defaults(handler=run_slices)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two clankerprof JSON slice or boundary outputs.",
    )
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--threshold-abs", type=float, default=2.0)
    compare.add_argument("--threshold-rel", type=float, default=15.0)
    compare.add_argument("--focus-slices", nargs="*", default=[])
    compare.add_argument(
        "--focus-boundaries",
        "--focus-scopes",
        dest="focus_boundaries",
        nargs="*",
        default=[],
    )
    compare.set_defaults(handler=run_compare)

    facts = subparsers.add_parser(
        "facts",
        help="Export decoded pprof sample facts as stable JSON.",
    )
    facts.add_argument("--profile", required=True)
    facts.add_argument("--output")
    facts.set_defaults(handler=run_facts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clankerprof",
        description="Language-agnostic pprof analyzer with runtime-specific rule packs.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument("--output")
    register_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def _emit_json(payload: dict[str, Any], output_path: str | None) -> None:
    rendered = render_json_payload(payload)
    if output_path:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        payload = cast(dict[str, Any], args.handler(args))
        _emit_json(payload, None if payload.get("output") else args.output)
        if payload.get("tool") == "clankerprof_compare" and payload.get(
            "has_regression"
        ):
            return 2
        return 0
    except ValueError as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
