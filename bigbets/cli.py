from __future__ import annotations

import argparse
import json
import re
import sys

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bigbets.core import (
    JsonValue,
    ValidationFailure,
    load_bigbets_registry,
    normalize_bigbets_registry,
    registry_to_input_payload,
    render_artifact_metadata_json,
    render_bigbets,
    write_bigbets_artifacts,
)
from bigbets.site import list_storage_adapters, site_result_payload, write_static_site
from bigbets.version import (
    BIGBETS_ARTIFACT_SCHEMA_VERSION,
    BIGBETS_REGISTRY_SCHEMA_VERSION,
    BIGBETS_VERSION,
    generator_metadata,
)

_FORMATS = {
    "metadata": "big_bets.artifact_metadata.json",
    "json": "big_bets.registry.json",
    "csv": "big_bets.rankings.csv",
    "markdown": "big_bets.md",
    "mermaid": "big_bets.mmd",
    "excalidraw": "big_bets.excalidraw",
    "svg": "big_bets.svg",
    "html": "index.html",
}


def handle_validate(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = load_bigbets_registry(Path(args.input))
    normalized = normalize_bigbets_registry(registry)
    summary = cast(dict[str, JsonValue], normalized["summary"])
    return {
        "ok": True,
        "tool": "bigbets_validate",
        "input": args.input,
        "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
        "artifact_schema_version": BIGBETS_ARTIFACT_SCHEMA_VERSION,
        "generator": cast(dict[str, JsonValue], generator_metadata()),
        "summary": summary,
    }


def handle_emit(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = load_bigbets_registry(Path(args.input))
    rendered = render_bigbets(registry)
    content = {
        "metadata": render_artifact_metadata_json(),
        "json": rendered.registry_json,
        "csv": rendered.rankings_csv,
        "markdown": rendered.markdown,
        "mermaid": rendered.mermaid,
        "excalidraw": rendered.excalidraw,
        "svg": rendered.svg,
        "html": rendered.html,
    }[args.format]
    if args.output:
        Path(args.output).write_text(content, encoding="utf-8")
    else:
        print(content, end="" if content.endswith("\n") else "\n")
    return {
        "ok": True,
        "tool": "bigbets_emit",
        "input": args.input,
        "format": args.format,
        "output": args.output,
        "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
        "artifact_schema_version": BIGBETS_ARTIFACT_SCHEMA_VERSION,
        "generator": cast(dict[str, JsonValue], generator_metadata()),
        "bytes": len(content.encode("utf-8")),
    }


def handle_render(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = load_bigbets_registry(Path(args.input))
    output_dir = Path(args.output_dir)
    artifacts = write_bigbets_artifacts(registry, output_dir)
    return {
        "ok": True,
        "tool": "bigbets_render",
        "input": args.input,
        "output_dir": str(output_dir),
        "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
        "artifact_schema_version": BIGBETS_ARTIFACT_SCHEMA_VERSION,
        "generator": cast(dict[str, JsonValue], generator_metadata()),
        "artifacts": [
            {"format": _format_for_path(path), "path": str(path)}
            for path in artifacts
        ],
    }


def handle_site_scaffold(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = load_bigbets_registry(Path(args.input))
    scaffold = write_static_site(
        registry,
        Path(args.output_dir),
        app_id=args.app_id,
        storage_adapter=args.storage_adapter,
        storage_adapter_file=(
            Path(args.storage_adapter_file) if args.storage_adapter_file else None
        ),
        overwrite_storage_adapter=args.overwrite_storage_adapter,
    )
    return {
        "ok": True,
        "tool": "bigbets_site_scaffold",
        "input": args.input,
        **site_result_payload(scaffold),
    }


def handle_site_adapters(_args: argparse.Namespace) -> dict[str, JsonValue]:
    return {
        "ok": True,
        "tool": "bigbets_site_adapters",
        "adapters": [
            {
                "name": adapter.name,
                "description": adapter.description,
                "source": adapter.source,
            }
            for adapter in list_storage_adapters()
        ],
    }


def handle_snapshot_create(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = load_bigbets_registry(Path(args.input))
    output_dir = Path(args.output_dir)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    name = args.name or f"Plan {created_at[:10]}"
    snapshot_id = f"{created_at.replace('-', '').replace(':', '').replace('Z', 'Z')}-{_slug(name)}"
    snapshot_path = snapshots_dir / f"{snapshot_id}.registry.json"
    payload = registry_to_input_payload(registry)
    snapshot_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    index_path = snapshots_dir / "index.json"
    snapshots = _read_snapshot_index(index_path)
    snapshots = [item for item in snapshots if item.get("id") != snapshot_id]
    snapshots.insert(
        0,
        {
            "id": snapshot_id,
            "name": name,
            "created_at": created_at,
            "path": str(snapshot_path.relative_to(output_dir)),
            "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
            "generator": cast(dict[str, JsonValue], generator_metadata()),
        },
    )
    index_path.write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "ok": True,
        "tool": "bigbets_snapshot_create",
        "input": args.input,
        "output_dir": str(output_dir),
        "snapshot": snapshots[0],
        "snapshot_path": str(snapshot_path),
        "index_path": str(index_path),
    }


def handle_snapshot_list(args: argparse.Namespace) -> dict[str, JsonValue]:
    output_dir = Path(args.output_dir)
    index_path = output_dir / "snapshots" / "index.json"
    snapshots = _read_snapshot_index(index_path)
    return {
        "ok": True,
        "tool": "bigbets_snapshot_list",
        "output_dir": str(output_dir),
        "index_path": str(index_path),
        "snapshots": cast(list[JsonValue], snapshots),
    }


def register_bigbets_leaf_commands(subparsers: Any) -> None:
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a bigbets registry."
    )
    validate_parser.add_argument("--input", required=True)
    validate_parser.set_defaults(handler=handle_validate)

    emit_parser = subparsers.add_parser(
        "emit",
        help="Emit one generated bigbets artifact to stdout or --output.",
    )
    emit_parser.add_argument("--input", required=True)
    emit_parser.add_argument(
        "--format",
        required=True,
        choices=sorted(_FORMATS),
        help="Artifact format to emit.",
    )
    emit_parser.add_argument("--output", help="Optional file path for the artifact.")
    emit_parser.set_defaults(handler=handle_emit)

    render_parser = subparsers.add_parser(
        "render",
        help="Render JSON, CSV, Markdown, Mermaid, SVG, and static HTML.",
    )
    render_parser.add_argument("--input", required=True)
    render_parser.add_argument("--output-dir", required=True)
    render_parser.set_defaults(handler=handle_render)

    site_parser = subparsers.add_parser(
        "site",
        help="Generate an editable host-neutral static site.",
    )
    site_subparsers = site_parser.add_subparsers(dest="site_command", required=True)
    scaffold_parser = site_subparsers.add_parser(
        "scaffold",
        help="Write an editable static site seeded from a bigbets registry.",
    )
    scaffold_parser.add_argument("--input", required=True)
    scaffold_parser.add_argument("--output-dir", required=True)
    scaffold_parser.add_argument("--app-id", default="bigbets-portfolio")
    scaffold_parser.add_argument(
        "--storage-adapter",
        default="local-storage",
        choices=[adapter.name for adapter in list_storage_adapters()],
        help="Built-in storage adapter to write beside the generated site.",
    )
    scaffold_parser.add_argument(
        "--storage-adapter-file",
        help="Copy a host-specific storage-adapter.js from this path.",
    )
    scaffold_parser.add_argument(
        "--overwrite-storage-adapter",
        action="store_true",
        help="Overwrite an existing storage-adapter.js in the output directory.",
    )
    scaffold_parser.set_defaults(handler=handle_site_scaffold)

    adapters_parser = site_subparsers.add_parser(
        "adapters",
        help="List built-in static-site storage adapters.",
    )
    adapters_parser.set_defaults(handler=handle_site_adapters)

    snapshot_parser = subparsers.add_parser(
        "snapshot",
        help="Create and list dated registry snapshots beside rendered artifacts.",
    )
    snapshot_subparsers = snapshot_parser.add_subparsers(
        dest="snapshot_command", required=True
    )
    snapshot_create_parser = snapshot_subparsers.add_parser(
        "create",
        help="Write a dated registry snapshot and update snapshots/index.json.",
    )
    snapshot_create_parser.add_argument("--input", required=True)
    snapshot_create_parser.add_argument("--output-dir", required=True)
    snapshot_create_parser.add_argument("--name", help="Human label for the snapshot.")
    snapshot_create_parser.set_defaults(handler=handle_snapshot_create)

    snapshot_list_parser = snapshot_subparsers.add_parser(
        "list",
        help="List snapshots in an output directory.",
    )
    snapshot_list_parser.add_argument("--output-dir", required=True)
    snapshot_list_parser.set_defaults(handler=handle_snapshot_list)


def register_bigbets_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "bigbets",
        help="Validate and render a ranked big-bet portfolio registry.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {BIGBETS_VERSION}",
    )
    bigbets_subparsers = parser.add_subparsers(
        dest="bigbets_command", required=True
    )
    register_bigbets_leaf_commands(bigbets_subparsers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bigbets",
        description="Validate and render ranked big-bet portfolio registries.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {BIGBETS_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_bigbets_leaf_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
        payload = cast(dict[str, JsonValue], args.handler(args))
        if args.command == "emit" and not args.output:
            return 0
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except ValidationFailure as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def _format_for_path(path: Path) -> str:
    for format_name, filename in _FORMATS.items():
        if path.name == filename:
            return format_name
    return path.suffix.lstrip(".") or "unknown"


def _read_snapshot_index(path: Path) -> list[dict[str, JsonValue]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValidationFailure(f"{path} must contain a JSON list.")
    return cast(list[dict[str, JsonValue]], payload)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "plan"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
