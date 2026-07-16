from __future__ import annotations

import argparse
import json
import sys

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from autoclanker import __version__
from autoclanker.bayes_layer.cli_adapter import register_adapter_commands
from autoclanker.bayes_layer.cli_beliefs import register_belief_commands
from autoclanker.bayes_layer.cli_eval import register_eval_commands
from autoclanker.bayes_layer.cli_session import register_session_commands
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    JsonValue,
    SessionFailure,
    ValidationFailure,
)
from autoclanker.cli_graph import register_graph_commands
from autoclanker.issue_seeder import register_issue_seed_commands
from bigbets.cli import register_bigbets_commands
from clankerprof.cli import register_pprof_commands
from goalloop.cli import register_goalloop_commands

EXIT_VALIDATION_ERROR = 2
EXIT_SESSION_ERROR = 3
EXIT_ADAPTER_ERROR = 4

_LOCAL_OUTPUT_COMMAND_PREFIXES = (
    ("bigbets", "emit"),
    ("bigbets", "issues", "merge"),
    ("pprof", "boundaries"),
    ("pprof", "facts"),
    ("pprof", "scopes"),
    ("pprof", "slices"),
    ("pprof", "targets"),
)


def _is_local_output_arg(argv: Sequence[str], index: int) -> bool:
    command_parts = tuple(item for item in argv[:index] if not item.startswith("-"))
    return any(
        command_parts[: len(prefix)] == prefix
        for prefix in _LOCAL_OUTPUT_COMMAND_PREFIXES
    )


def _normalize_global_output_position(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    extracted: list[str] = []
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item == "--output":
            if index + 1 >= len(normalized):
                raise ValueError("--output requires a path argument.")
            if _is_local_output_arg(normalized, index):
                index += 2
                continue
            extracted = [item, normalized[index + 1]]
            del normalized[index : index + 2]
            continue
        if item.startswith("--output="):
            if _is_local_output_arg(normalized, index):
                index += 1
                continue
            extracted = [item]
            del normalized[index]
            continue
        index += 1
    return extracted + normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoclanker",
        description="Non-interactive CLI for the autoclanker Bayesian guidance layer.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--output",
        help="Optional path for a copy of the JSON result payload.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    register_belief_commands(subparsers)
    register_eval_commands(subparsers)
    register_adapter_commands(subparsers)
    register_session_commands(subparsers)
    register_graph_commands(subparsers)
    register_issue_seed_commands(subparsers)
    register_bigbets_commands(subparsers)
    register_pprof_commands(subparsers)
    _register_goalloop_family(subparsers)
    return parser


def _emit_json(payload: dict[str, JsonValue], output_path: str | None) -> None:
    raw_output = payload.pop("raw_output", None)
    if raw_output is not None:
        print(raw_output)
        return
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is not None:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _emit_error(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, sort_keys=True), file=sys.stderr)


def _register_goalloop_family(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "goalloop",
        help="Deterministic goal loops for agent harnesses.",
    )

    def _exit_with_code(
        handler: Callable[[argparse.Namespace], int],
    ) -> Callable[[argparse.Namespace], int]:
        def wrapped(args: argparse.Namespace) -> int:
            if getattr(args, "output", None):
                raise ValueError(
                    "goalloop commands write JSON directly to stdout and do not "
                    "support the global --output flag; use shell redirection."
                )
            raise SystemExit(handler(args))

        return wrapped

    register_goalloop_commands(
        parser.add_subparsers(dest="goalloop_command", required=True),
        wrap=_exit_with_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        normalized_argv = (
            _normalize_global_output_position(list(sys.argv[1:]))
            if argv is None
            else _normalize_global_output_position(argv)
        )
        args = parser.parse_args(normalized_argv)
        handler = args.handler
        payload = cast(dict[str, JsonValue], handler(args))
        _emit_json(payload, None if payload.get("output") else args.output)
        if payload.get("tool") == "clankerprof_compare" and payload.get(
            "has_regression"
        ):
            return EXIT_VALIDATION_ERROR
        return 0
    except ValidationFailure as exc:
        _emit_error(str(exc))
        return EXIT_VALIDATION_ERROR
    except SessionFailure as exc:
        _emit_error(str(exc))
        return EXIT_SESSION_ERROR
    except AdapterFailure as exc:
        _emit_error(str(exc))
        return EXIT_ADAPTER_ERROR
    except ValueError as exc:
        _emit_error(str(exc))
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
