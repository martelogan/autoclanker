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
    AttributionRule,
    JsonValue,
    RuntimeRuleSet,
    SliceAnalysisOptions,
    SliceDefinition,
    TargetAnalysisOptions,
    analyze_slices,
    analyze_targets,
    load_default_ruby_core_classes,
    load_json_mapping,
    load_ruby_core_classes,
    ruby_rules,
)
from clankerprof.compare import CompareOptions, compare_slice_json
from clankerprof.model import CategoryStats
from clankerprof.proto import load_profile
from clankerprof.render import (
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


def _optional_unattributed_gems(payload: dict[str, object]) -> int | None:
    value = payload.get("unattributed_gems")
    if value is None:
        return None
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
    valid_filter_keys = {"name", "path", "gem", "slice"}
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
    for attribute in options.attributes:
        key = (attribute.key, attribute.value)
        if attribute.key not in {"name", "path", "gem"}:
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


def _runtime_rules(args: argparse.Namespace) -> RuntimeRuleSet:
    runtime = str(getattr(args, "runtime", "generic"))
    no_enhanced = bool(getattr(args, "no_enhanced", False))
    if runtime == "generic" and not no_enhanced:
        return RuntimeRuleSet(name="generic")
    if runtime not in {"generic", "ruby"}:
        raise ValueError(f"Unsupported runtime: {runtime}")
    core_classes = (
        load_ruby_core_classes(args.ruby_core_classes)
        if args.ruby_core_classes is not None
        else load_default_ruby_core_classes()
    )
    return ruby_rules(
        core_classes,
        verbose=bool(args.verbose_runtime_internals),
    )


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
        render_target_csv(results, attributables=attributables, simplified=True),
        encoding="utf-8",
    )
    verbose_path.write_text(
        render_target_csv(results, attributables=attributables, simplified=False),
        encoding="utf-8",
    )
    return {
        "tool": "clankerprof_targets",
        "ok": True,
        "output": str(simplified_path),
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
    profile = load_profile(args.profile)
    config = load_json_mapping(args.config)
    attributables = _load_attributables(args.cpu_attributables)
    results = analyze_targets(
        profile,
        config,
        TargetAnalysisOptions(
            runtime_rules=_runtime_rules(args),
            enhanced_runtime_categorization=not bool(args.no_enhanced),
            fold_runtime_internals=bool(args.fold_runtime_internals),
            track_semantic_callers=bool(args.track_semantic_callers),
            attributables=attributables,
            legacy_no_enhanced_caller_fallback=bool(args.no_enhanced),
        ),
    )
    if args.semantic_callers_csv:
        if not args.track_semantic_callers:
            raise ValueError(
                "--semantic-callers-csv requires --track-semantic-callers."
            )
        Path(args.semantic_callers_csv).write_text(
            render_semantic_callers_csv(results) + "\n",
            encoding="utf-8",
        )
    if args.format == "json":
        return cast(dict[str, Any], render_target_json(results))
    if args.legacy_target_csv_layout:
        if args.format != "csv" or not args.output:
            raise ValueError(
                "--legacy-target-csv-layout requires --format csv and --output."
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
    if raw_profile is None:
        raise ValueError("--profile is required.")
    profile = load_profile(raw_profile)
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
    raw_unattributed_gems = _optional_unattributed_gems(config)
    if args.unattributed_gems is not None and raw_unattributed_gems is not None:
        raise ValueError(
            "unattributed_gems specified both on command line and in config file."
        )
    unattributed_gems = (
        args.unattributed_gems
        if args.unattributed_gems is not None
        else raw_unattributed_gems
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
        slices=_load_slices(raw_slices),
        filters=raw_filters,
        collapse=raw_collapse,
        attributes=attributes,
        top=top,
        by_slice=by_slice,
        show_paths=show_paths,
        no_collapse_native=no_collapse_native,
        unattributed_gems=unattributed_gems,
        allow_virtual_attribute_slices=bool(args.allow_virtual_attribute_slices),
    )
    _validate_slice_options(options)
    result = analyze_slices(
        profile,
        options,
    )
    payload = cast(dict[str, Any], render_slice_json(result, options))
    if args.output:
        Path(args.output).write_text(
            render_json_payload(payload) + "\n", encoding="utf-8"
        )
        return {"tool": "clankerprof_slices", "ok": True, "output": args.output}
    return payload


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError("Compare inputs must be JSON objects.")
    return compare_slice_json(
        cast(dict[str, Any], before),
        cast(dict[str, Any], after),
        CompareOptions(
            threshold_abs=float(args.threshold_abs),
            threshold_rel=float(args.threshold_rel),
            focus_slices=_focus_slices(args.focus_slices),
        ),
    )


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
    targets.add_argument("--profile", required=True)
    targets.add_argument("--config", required=True)
    targets.add_argument("--output")
    targets.add_argument(
        "--format",
        choices=("text", "csv", "simple-csv", "json"),
        default="json",
    )
    targets.add_argument("--runtime", choices=("generic", "ruby"), default="generic")
    targets.add_argument("--ruby-core-classes")
    targets.add_argument(
        "--no-enhanced",
        action="store_true",
        help=(
            "Disable runtime semantic categorization and use the legacy "
            "Ruby native-caller fallback before regex matching."
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
    targets.add_argument("--cpu-attributables")
    targets.add_argument("--semantic-callers-csv")
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

    slices = subparsers.add_parser(
        "slices",
        help="Run slice attribution over a pprof profile.",
    )
    slices.add_argument("--profile")
    slices.add_argument("--config")
    slices.add_argument("--slices")
    slices.add_argument("--filter", dest="filters", action="append", default=[])
    slices.add_argument("--collapse", action="append", default=[])
    slices.add_argument("--attribute", action="append", default=[])
    slices.add_argument("--top", type=int)
    slices.add_argument("--by-slice", nargs="?", const="0.1%")
    slices.add_argument("--show-paths", action="store_true")
    slices.add_argument("--no-collapse-native", action="store_true")
    slices.add_argument("--unattributed-gems", nargs="?", const=2**63 - 1, type=int)
    slices.add_argument("--allow-virtual-attribute-slices", action="store_true")
    slices.add_argument("--output")
    slices.set_defaults(handler=run_slices)

    compare = subparsers.add_parser(
        "compare",
        help="Compare two clankerprof JSON slice outputs.",
    )
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--threshold-abs", type=float, default=2.0)
    compare.add_argument("--threshold-rel", type=float, default=15.0)
    compare.add_argument("--focus-slices", nargs="*", default=[])
    compare.set_defaults(handler=run_compare)


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
