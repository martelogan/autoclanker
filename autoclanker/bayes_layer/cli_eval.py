from __future__ import annotations

import argparse
import sys

from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer import (
    load_serialized_payload,
    load_serialized_payload_from_text,
    validate_eval_result,
)
from autoclanker.bayes_layer.types import JsonValue, to_json_value


def _load_payload(input_path: str | None) -> dict[str, object]:
    if input_path and input_path != "-":
        return load_serialized_payload(Path(input_path))
    return load_serialized_payload_from_text(sys.stdin.read())


def handle_validate(args: argparse.Namespace) -> dict[str, JsonValue]:
    result = validate_eval_result(_load_payload(args.input))
    return cast(dict[str, JsonValue], to_json_value(result))


def register_eval_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser("eval", help="Validate eval results.")
    eval_subparsers = parser.add_subparsers(dest="eval_command", required=True)
    validate_parser = eval_subparsers.add_parser(
        "validate", help="Validate one eval result payload."
    )
    validate_parser.add_argument(
        "--input",
        help="Path to a JSON eval result file. Use '-' to read from stdin.",
    )
    validate_parser.set_defaults(handler=handle_validate)
