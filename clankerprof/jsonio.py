"""Strict JSON/YAML input parsing shared by clankerprof input surfaces.

Python's ``json`` module accepts the non-standard ``Infinity``/``-Infinity``/
``NaN`` tokens that RFC 8259 forbids and that the Rust port (serde_json)
rejects at parse time, and PyYAML silently keeps the last value for duplicate
mapping keys where serde_yaml errors. Every clankerprof JSON/YAML input goes
through these loaders so both implementations fail closed on the same inputs.
"""

from __future__ import annotations

import json
import math
import re

from collections.abc import Callable
from typing import Any, cast

import yaml


def _reject_constant(token: str) -> Any:
    raise ValueError(f"JSON input must not contain the non-finite token {token!r}.")


def parse_strict_json(text: str) -> Any:
    # Integer literals outside [i64::MIN, u64::MAX] need no guard here:
    # serde_json parses them as f64, and Python's unbounded int coerces to
    # the identical f64 in float-domain fields while integer-domain fields
    # reject both representations with the same messages (pinned by the
    # parity suite).
    return json.loads(text, parse_constant=_reject_constant)


YAML_KEY_MESSAGE = "YAML mapping keys must be strings."


class _StrictYamlLoader(yaml.SafeLoader):
    """SafeLoader with serde_yaml's scalar typing and strict mapping keys.

    Rejects duplicate and non-string mapping keys, and replaces PyYAML's
    YAML 1.1 implicit scalar resolution with the exact typing serde_yaml 0.9
    applies (empirical table pinned by tests in both languages and by
    crates/clankerprof-core/tests/yaml_scalar_semantics.rs):

    - bools are only ``true/True/TRUE/false/False/FALSE`` (never
      ``yes/no/on/off``);
    - ints are decimal without leading zeros, underscores, or sexagesimal
      forms, plus signed ``0x``/``0o``/``0b`` prefixed forms; values outside
      ``[-2^63, 2^64-1]`` are a parse error exactly where serde_yaml fails;
    - floats require a dot or exponent (unsigned exponents allowed, unlike
      YAML 1.1), never contain underscores, and overflow to a plain string
      rather than infinity; ``.inf``/``.nan`` spellings keep working, with
      the NaN forms unsigned-only;
    - the 1.1 timestamp resolver is removed (`2026-01-01` stays a string)
      and so is the ``=`` value special.
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


_REPLACED_RESOLVER_TAGS = frozenset(
    f"tag:yaml.org,2002:{name}"
    for name in ("timestamp", "value", "bool", "int", "float")
)

_StrictYamlLoader.yaml_implicit_resolvers = {
    prefix: [
        (tag, regexp) for tag, regexp in resolvers if tag not in _REPLACED_RESOLVER_TAGS
    ]
    for prefix, resolvers in _StrictYamlLoader.yaml_implicit_resolvers.items()
}

# PyYAML ships no complete type stubs for the resolver registration API.
_add_implicit_resolver = cast(
    "Callable[..., None]",
    _StrictYamlLoader.add_implicit_resolver,  # pyright: ignore[reportUnknownMemberType]
)

_add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)

_INT_MIN = -(2**63)
_UINT_MAX = 2**64 - 1

_add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0x[0-9a-fA-F]+|0o[0-7]+|0b[01]+|0|[1-9][0-9]*)$", re.ASCII),
    list("-+0123456789"),
)

_FLOAT_PATTERN = re.compile(
    r"""^[-+]?(?:
        \.[0-9]+(?:[eE][-+]?[0-9]+)?
        |[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?
        |[0-9]+[eE][-+]?[0-9]+
    )$""",
    re.ASCII | re.VERBOSE,
)

_add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        _FLOAT_PATTERN.pattern + r"|^[-+]?\.(?:inf|Inf|INF)$|^\.(?:nan|NaN|NAN)$",
        re.ASCII | re.VERBOSE,
    ),
    list("-+0123456789."),
)


def _construct_serde_int(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> int:
    scalar = str(loader.construct_scalar(node))
    text = scalar
    sign = 1
    if text and text[0] in "+-":
        sign = -1 if text[0] == "-" else 1
        text = text[1:]
    if text.startswith("0x"):
        value = sign * int(text[2:], 16)
    elif text.startswith("0o"):
        value = sign * int(text[2:], 8)
    elif text.startswith("0b"):
        value = sign * int(text[2:], 2)
    else:
        value = sign * int(text)
    if not _INT_MIN <= value <= _UINT_MAX:
        # serde_yaml fails the whole parse here; the location suffix it
        # appends ("at line N column M") is engine-specific detail.
        width = "i128" if value < 0 else "u128"
        raise ValueError(
            f"invalid type: integer `{value}` as {width}, expected any YAML value"
        )
    return value


def _construct_serde_float(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> Any:
    scalar = str(loader.construct_scalar(node))
    lowered = scalar.lstrip("+-").lower()
    if lowered == ".inf":
        return math.inf if not scalar.startswith("-") else -math.inf
    if lowered == ".nan":
        return math.nan
    value = float(scalar)
    if math.isinf(value):
        # serde_yaml keeps overflowing literals (e.g. `1e309`) as strings
        # rather than resolving them to infinity.
        return scalar
    return value


_StrictYamlLoader.add_constructor("tag:yaml.org,2002:int", _construct_serde_int)
_StrictYamlLoader.add_constructor("tag:yaml.org,2002:float", _construct_serde_float)


def parse_strict_yaml(text: str) -> Any:
    """`yaml.safe_load` with duplicate or non-string mapping keys rejected."""
    return yaml.load(text, Loader=_StrictYamlLoader)
