from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer import load_serialized_payload, validate_adapter_config
from autoclanker.bayes_layer.adapters import available_adapter_kinds, load_adapter
from autoclanker.bayes_layer.types import JsonValue, ValidAdapterConfig, to_json_value


def _fixture_config() -> ValidAdapterConfig:
    return ValidAdapterConfig(
        kind="fixture",
        mode="fixture",
        session_root=".autoclanker",
        allow_missing=False,
    )


def handle_validate_config(args: argparse.Namespace) -> dict[str, JsonValue]:
    path = Path(args.input)
    config = validate_adapter_config(
        load_serialized_payload(path), base_dir=path.parent
    )
    return cast(dict[str, JsonValue], to_json_value(config))


def handle_list(_args: argparse.Namespace) -> dict[str, JsonValue]:
    adapters = [
        {
            "kind": "fixture",
            "supported_modes": ["fixture"],
            "description": "Self-contained deterministic adapter for tests and local development.",
        },
        {
            "kind": "autoresearch",
            "supported_modes": [
                "auto",
                "local_repo_path",
                "installed_module",
                "subprocess_cli",
            ],
            "description": "First-party adapter that can resolve a checkout path, installed adapter module, or runnable adapter command.",
        },
        {
            "kind": "cevolve",
            "supported_modes": [
                "auto",
                "local_repo_path",
                "installed_module",
                "subprocess_cli",
            ],
            "description": "First-party adapter that can resolve a checkout path, installed adapter module, or runnable adapter command.",
        },
        {
            "kind": "python_module",
            "supported_modes": ["installed_module"],
            "description": "Generic external module integration path.",
        },
        {
            "kind": "subprocess",
            "supported_modes": ["subprocess_cli"],
            "description": "Generic external subprocess integration path.",
        },
    ]
    return {
        "adapters": cast(list[JsonValue], adapters),
        "known_kinds": list(available_adapter_kinds()),
    }


def handle_probe(args: argparse.Namespace) -> dict[str, JsonValue]:
    path = Path(args.input)
    config = validate_adapter_config(
        load_serialized_payload(path), base_dir=path.parent
    )
    adapter = load_adapter(config)
    return cast(dict[str, JsonValue], to_json_value(adapter.probe()))


def handle_registry(args: argparse.Namespace) -> dict[str, JsonValue]:
    if args.input is None:
        config = _fixture_config()
    else:
        path = Path(args.input)
        config = validate_adapter_config(
            load_serialized_payload(path), base_dir=path.parent
        )
    adapter = load_adapter(config)
    registry = adapter.build_registry()
    default_genotype = [
        {"gene_id": ref.gene_id, "state_id": ref.state_id}
        for ref in registry.default_genotype()
    ]
    return {
        "kind": config.kind,
        "mode": config.mode,
        "registry": cast(dict[str, JsonValue], to_json_value(registry.to_dict())),
        "default_genotype": cast(list[JsonValue], to_json_value(default_genotype)),
    }


def handle_surface(args: argparse.Namespace) -> dict[str, JsonValue]:
    if args.input is None:
        config = _fixture_config()
    else:
        path = Path(args.input)
        config = validate_adapter_config(
            load_serialized_payload(path), base_dir=path.parent
        )
    adapter = load_adapter(config)
    registry = adapter.build_registry()
    return {
        "kind": config.kind,
        "mode": config.mode,
        "surface_summary": cast(
            dict[str, JsonValue], to_json_value(registry.surface_summary())
        ),
        "surface": cast(dict[str, JsonValue], to_json_value(registry.to_dict())),
    }


def register_adapter_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "adapter", help="Inspect adapter configs and availability."
    )
    adapter_subparsers = parser.add_subparsers(dest="adapter_command", required=True)

    validate_parser = adapter_subparsers.add_parser(
        "validate-config",
        help="Validate an adapter config file.",
    )
    validate_parser.add_argument(
        "--input", required=True, help="Path to an adapter config file."
    )
    validate_parser.set_defaults(handler=handle_validate_config)

    list_parser = adapter_subparsers.add_parser(
        "list", help="List builtin and generic adapter kinds."
    )
    list_parser.set_defaults(handler=handle_list)

    probe_parser = adapter_subparsers.add_parser(
        "probe", help="Probe one adapter configuration."
    )
    probe_parser.add_argument(
        "--input", required=True, help="Path to an adapter config file."
    )
    probe_parser.set_defaults(handler=handle_probe)

    registry_parser = adapter_subparsers.add_parser(
        "registry",
        help="Emit the adapter gene/state registry as machine-readable JSON.",
    )
    registry_parser.add_argument(
        "--input",
        help="Optional path to an adapter config file. Defaults to the fixture registry.",
    )
    registry_parser.set_defaults(handler=handle_registry)

    surface_parser = adapter_subparsers.add_parser(
        "surface",
        help="Emit the richer optimization surface, including semantic metadata.",
    )
    surface_parser.add_argument(
        "--input",
        help="Optional path to an adapter config file. Defaults to the fixture registry.",
    )
    surface_parser.set_defaults(handler=handle_surface)
