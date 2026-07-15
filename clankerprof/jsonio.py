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


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of last-wins."""

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
            try:
                duplicate = key in seen
            except TypeError:
                # Unhashable keys fail construct_mapping's own validation.
                continue
            if duplicate:
                # Matches serde_yaml's duplicate-key error text so the two
                # implementations emit the same envelope.
                raise ValueError(f'duplicate entry with key "{key}"')
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_strict_yaml(text: str) -> Any:
    """`yaml.safe_load` with duplicate mapping keys rejected."""
    return yaml.load(text, Loader=_StrictYamlLoader)
