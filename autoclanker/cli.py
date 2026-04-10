from __future__ import annotations

import argparse
import json
import sys

from collections.abc import Sequence
from pathlib import Path
from typing import cast

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

EXIT_VALIDATION_ERROR = 2
EXIT_SESSION_ERROR = 3
EXIT_ADAPTER_ERROR = 4


def _normalize_global_output_position(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    extracted: list[str] = []
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item == "--output":
            if index + 1 >= len(normalized):
                raise ValueError("--output requires a path argument.")
            extracted = [item, normalized[index + 1]]
            del normalized[index : index + 2]
            continue
        if item.startswith("--output="):
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
    return parser


def _emit_json(payload: dict[str, JsonValue], output_path: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output_path is not None:
        Path(output_path).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def _emit_error(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, sort_keys=True), file=sys.stderr)


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
        _emit_json(payload, args.output)
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
