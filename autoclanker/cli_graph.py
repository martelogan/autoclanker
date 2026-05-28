from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer.types import JsonValue
from autoclanker.clankergraph import (
    load_clankergraph_document,
    summarize_clankergraph_document,
)


def handle_validate(args: argparse.Namespace) -> dict[str, JsonValue]:
    document = load_clankergraph_document(Path(cast(str, args.input)))
    summary = summarize_clankergraph_document(document)
    return {
        "ok": True,
        "summary": summary,
    }


def handle_summarize(args: argparse.Namespace) -> dict[str, JsonValue]:
    document = load_clankergraph_document(Path(cast(str, args.input)))
    return summarize_clankergraph_document(document)


def register_graph_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "graph",
        help="Validate and summarize generic Clankergraph artifacts.",
    )
    graph_subparsers = parser.add_subparsers(dest="graph_command", required=True)

    validate_parser = graph_subparsers.add_parser(
        "validate",
        help="Validate a clankergraph.v1 JSON artifact and emit a compact summary.",
    )
    validate_parser.add_argument(
        "--input",
        required=True,
        help="Path to a clankergraph.v1 JSON artifact.",
    )
    validate_parser.set_defaults(handler=handle_validate)

    summarize_parser = graph_subparsers.add_parser(
        "summarize",
        help="Summarize a clankergraph.v1 JSON artifact.",
    )
    summarize_parser.add_argument(
        "--input",
        required=True,
        help="Path to a clankergraph.v1 JSON artifact.",
    )
    summarize_parser.set_defaults(handler=handle_summarize)
