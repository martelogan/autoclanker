from __future__ import annotations

import argparse

from typing import cast

from autoclanker import __version__
from autoclanker.cli import build_parser
from tests.compliance import covers


def _subparser_choices(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        raw_choices = getattr(action, "choices", None)
        if isinstance(raw_choices, dict):
            choices: dict[str, argparse.ArgumentParser] = {}
            for name, choice in cast(dict[object, object], raw_choices).items():
                if isinstance(name, str) and isinstance(
                    choice, argparse.ArgumentParser
                ):
                    choices[name] = choice
            if choices:
                return choices
    raise AssertionError("Expected parser to expose subcommands.")


@covers("M0-001")
def test_package_exposes_version() -> None:
    assert __version__


@covers("M0-002")
def test_cli_parser_exposes_required_command_tree() -> None:
    parser = build_parser()
    root_commands = _subparser_choices(parser)

    assert {"beliefs", "eval", "adapter", "session"} <= set(root_commands)
    assert {
        "validate",
        "preview",
        "compile",
        "expand-ideas",
        "canonicalize-ideas",
    } <= set(_subparser_choices(root_commands["beliefs"]))
    assert {"validate"} <= set(_subparser_choices(root_commands["eval"]))
    assert {"list", "probe", "validate-config", "registry", "surface"} <= set(
        _subparser_choices(root_commands["adapter"])
    )
    assert {
        "init",
        "apply-beliefs",
        "ingest-eval",
        "fit",
        "suggest",
        "recommend-commit",
        "status",
    } <= set(_subparser_choices(root_commands["session"]))
