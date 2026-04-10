from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer import (
    EraState,
    build_fixture_registry,
    compile_beliefs,
    load_inline_ideas_payload,
    load_serialized_payload,
    load_serialized_payload_from_text,
    preview_compiled_beliefs,
    validate_adapter_config,
)
from autoclanker.bayes_layer.adapters import load_adapter
from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationMode,
    CanonicalizationOutcome,
    canonicalize_belief_input,
    default_canonicalization_mode,
    load_canonicalization_model,
)
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    JsonValue,
    SessionContext,
    UserProfile,
    to_json_value,
)


def _load_input_payload(input_path: str | None) -> dict[str, object]:
    if input_path and input_path != "-":
        return load_serialized_payload(Path(input_path))
    return load_serialized_payload_from_text(sys.stdin.read())


def _load_belief_payload(args: argparse.Namespace) -> dict[str, object]:
    ideas_json = cast(str | None, getattr(args, "ideas_json", None))
    if ideas_json is not None:
        if args.input is not None:
            raise ValueError("Use either --input or --ideas-json, not both.")
        return load_inline_ideas_payload(ideas_json)
    return _load_input_payload(args.input)


def _fallback_session_context(args: argparse.Namespace) -> SessionContext | None:
    era_id = cast(str | None, getattr(args, "era_id", None))
    session_id = cast(str | None, getattr(args, "session_id", None))
    author = cast(str | None, getattr(args, "author", None))
    user_profile = cast(UserProfile | None, getattr(args, "user_profile", None))
    if (
        era_id is None
        and session_id is None
        and author is None
        and user_profile is None
    ):
        return None
    if era_id is None:
        raise ValueError("--era-id is required when overriding beginner idea context.")
    return SessionContext(
        era_id=era_id,
        session_id=session_id,
        author=author,
        user_profile=user_profile,
    )


def _resolve_registry(adapter_config_path: str | None) -> GeneRegistry:
    if adapter_config_path is None:
        return build_fixture_registry()
    path = Path(adapter_config_path)
    config = validate_adapter_config(
        load_serialized_payload(path), base_dir=path.parent
    )
    return load_adapter(config).build_registry()


def _canonicalization_mode(args: argparse.Namespace) -> CanonicalizationMode | None:
    raw_mode = cast(str | None, getattr(args, "canonicalization_mode", None))
    return cast(CanonicalizationMode | None, raw_mode)


def _canonicalize_input(
    args: argparse.Namespace,
    *,
    registry: GeneRegistry,
) -> CanonicalizationOutcome:
    payload = _load_belief_payload(args)
    model = load_canonicalization_model(
        cast(str | None, getattr(args, "canonicalization_model", None))
    )
    return canonicalize_belief_input(
        payload,
        fallback_session_context=_fallback_session_context(args),
        registry=registry,
        mode=default_canonicalization_mode(
            requested_mode=_canonicalization_mode(args),
            model=model,
        ),
        model=model,
    )


def handle_validate(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = _resolve_registry(getattr(args, "adapter_config", None))
    outcome = _canonicalize_input(args, registry=registry)
    payload = cast(dict[str, JsonValue], to_json_value(outcome.beliefs))
    payload["belief_count"] = len(outcome.beliefs.beliefs)
    if outcome.summary is not None:
        payload["canonicalization_summary"] = to_json_value(outcome.summary)
    if outcome.surface_overlay_payload is not None:
        payload["surface_overlay"] = outcome.surface_overlay_payload
    return payload


def handle_preview(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = _resolve_registry(args.adapter_config)
    outcome = _canonicalize_input(args, registry=registry)
    preview = preview_compiled_beliefs(
        outcome.beliefs,
        outcome.registry,
        EraState(era_id=outcome.beliefs.session_context.era_id),
    )
    payload = cast(dict[str, JsonValue], to_json_value(preview))
    if outcome.summary is not None:
        payload["canonicalization_summary"] = to_json_value(outcome.summary)
    if outcome.surface_overlay_payload is not None:
        payload["surface_overlay"] = outcome.surface_overlay_payload
    return payload


def handle_compile(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = _resolve_registry(args.adapter_config)
    outcome = _canonicalize_input(args, registry=registry)
    bundle = compile_beliefs(
        outcome.beliefs,
        outcome.registry,
        EraState(era_id=outcome.beliefs.session_context.era_id),
    )
    payload = cast(dict[str, JsonValue], to_json_value(bundle))
    if outcome.summary is not None:
        payload["canonicalization_summary"] = to_json_value(outcome.summary)
    if outcome.surface_overlay_payload is not None:
        payload["surface_overlay"] = outcome.surface_overlay_payload
    return payload


def handle_expand_ideas(args: argparse.Namespace) -> dict[str, JsonValue]:
    registry = _resolve_registry(getattr(args, "adapter_config", None))
    outcome = _canonicalize_input(args, registry=registry)
    payload = cast(
        dict[str, JsonValue], to_json_value(outcome.beliefs.canonical_payload)
    )
    payload["belief_count"] = len(outcome.beliefs.beliefs)
    if outcome.summary is not None:
        payload["canonicalization_summary"] = to_json_value(outcome.summary)
    if outcome.surface_overlay_payload is not None:
        payload["surface_overlay"] = outcome.surface_overlay_payload
    return payload


def _add_beginner_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ideas-json",
        help="Inline beginner ideas as JSON. Accepts one string idea, one idea object, a list of idea strings or objects, or an object with top-level 'ideas'. String ideas and omitted confidence values default to confidence 2.",
    )
    parser.add_argument(
        "--era-id",
        help="Optional era id for beginner idea files that omit session_context.",
    )
    parser.add_argument(
        "--session-id",
        help="Optional session id for beginner idea files that omit session_context.",
    )
    parser.add_argument(
        "--author",
        help="Optional author for beginner idea files that omit session_context.",
    )
    parser.add_argument(
        "--user-profile",
        choices=("basic", "expert"),
        help="Optional user profile for beginner idea files that omit session_context.",
    )
    parser.add_argument(
        "--canonicalization-mode",
        choices=("deterministic", "hybrid", "llm"),
        help="Override the beginner canonicalization pipeline mode. deterministic uses only registry semantics, hybrid uses deterministic resolution first and then the model for unresolved ideas, and llm uses the model first with deterministic fallback when the model returns no typed belief. Defaults to hybrid only when a canonicalization model is configured, otherwise deterministic.",
    )
    parser.add_argument(
        "--canonicalization-model",
        help="Optional provider-agnostic canonicalization model identifier. Use 'stub' for the built-in test model, 'anthropic' for the bundled Anthropic provider, or an import path exposing build_autoclanker_canonicalization_model().",
    )


def register_belief_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "beliefs", help="Validate and compile human belief files."
    )
    belief_subparsers = parser.add_subparsers(dest="belief_command", required=True)

    validate_parser = belief_subparsers.add_parser(
        "validate", help="Validate a belief batch."
    )
    validate_parser.add_argument(
        "--input",
        help="Path to a JSON or YAML belief file. Use '-' to read from stdin.",
    )
    validate_parser.add_argument(
        "--adapter-config",
        help="Optional adapter config used to resolve a registry for auto-canonicalization.",
    )
    _add_beginner_context_arguments(validate_parser)
    validate_parser.set_defaults(handler=handle_validate)

    preview_parser = belief_subparsers.add_parser(
        "preview",
        help="Compile a preview of belief priors against a registry.",
    )
    preview_parser.add_argument(
        "--input",
        help="Path to a JSON or YAML belief file. Use '-' to read from stdin.",
    )
    preview_parser.add_argument(
        "--adapter-config",
        help="Optional adapter config used to resolve a registry.",
    )
    _add_beginner_context_arguments(preview_parser)
    preview_parser.set_defaults(handler=handle_preview)

    compile_parser = belief_subparsers.add_parser(
        "compile",
        help="Compile a belief batch into priors and hints.",
    )
    compile_parser.add_argument(
        "--input",
        help="Path to a JSON or YAML belief file. Use '-' to read from stdin.",
    )
    compile_parser.add_argument(
        "--adapter-config",
        help="Optional adapter config used to resolve a registry.",
    )
    _add_beginner_context_arguments(compile_parser)
    compile_parser.set_defaults(handler=handle_compile)

    expand_parser = belief_subparsers.add_parser(
        "expand-ideas",
        help="Normalize a beginner idea file into the full typed belief schema.",
    )
    expand_parser.add_argument(
        "--input",
        help="Path to a JSON or YAML belief file. Use '-' to read from stdin.",
    )
    expand_parser.add_argument(
        "--adapter-config",
        help="Optional adapter config used to resolve a registry for auto-canonicalization.",
    )
    _add_beginner_context_arguments(expand_parser)
    expand_parser.set_defaults(handler=handle_expand_ideas)

    canonicalize_parser = belief_subparsers.add_parser(
        "canonicalize-ideas",
        help="Resolve high-level beginner ideas against the active registry. This currently emits the same normalized typed payload as expand-ideas, with canonicalization summary metadata.",
    )
    canonicalize_parser.add_argument(
        "--input",
        help="Path to a JSON or YAML belief file. Use '-' to read from stdin.",
    )
    canonicalize_parser.add_argument(
        "--adapter-config",
        help="Optional adapter config used to resolve a registry for auto-canonicalization.",
    )
    _add_beginner_context_arguments(canonicalize_parser)
    canonicalize_parser.set_defaults(handler=handle_expand_ideas)
