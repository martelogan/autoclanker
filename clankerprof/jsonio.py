"""Strict JSON/YAML input parsing shared by clankerprof input surfaces.

Python's ``json`` module accepts the non-standard ``Infinity``/``-Infinity``/
``NaN`` tokens that RFC 8259 forbids and that the Rust port (serde_json)
rejects at parse time, and PyYAML silently keeps the last value for duplicate
mapping keys where serde_yaml errors. Every clankerprof JSON/YAML input goes
through these loaders so both implementations fail closed on the same inputs.
"""

from __future__ import annotations

import json

from collections.abc import Callable
from typing import Any, cast

import yaml


def _reject_constant(token: str) -> Any:
    raise ValueError(f"JSON input must not contain the non-finite token {token!r}.")


def parse_strict_json(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


YAML_KEY_MESSAGE = "YAML mapping keys must be strings."


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate and non-string mapping keys.

    serde_yaml (YAML 1.2 core schema) has no timestamp type, so the 1.1
    timestamp resolver is removed: `2026-01-01` stays a plain string in both
    implementations instead of becoming a Python date (which would then be
    rejected as a non-string key that Rust cannot even observe).
    """

    def construct_mapping(
        self, node: yaml.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        seen: set[Any] = set()
        # PyYAML ships no complete type stubs for construct_object.
        construct_object = cast(
            "Callable[..., Any]",
            self.construct_object,  # pyright: ignore[reportUnknownMemberType]
        )
        for key_node, _value_node in node.value:
            key: Any = construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                # Rust walks the parsed tree with the same message; PyYAML's
                # bool/None/number key objects have no shared spelling with
                # serde_yaml, so neither implementation coerces them.
                raise ValueError(YAML_KEY_MESSAGE)
            if key in seen:
                # Matches serde_yaml's duplicate-key error text so the two
                # implementations emit the same envelope.
                raise ValueError(f'duplicate entry with key "{key}"')
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_StrictYamlLoader.yaml_implicit_resolvers = {
    prefix: [
        (tag, regexp)
        for tag, regexp in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for prefix, resolvers in _StrictYamlLoader.yaml_implicit_resolvers.items()
}


def parse_strict_yaml(text: str) -> Any:
    """`yaml.safe_load` with duplicate or non-string mapping keys rejected."""
    return yaml.load(text, Loader=_StrictYamlLoader)
