"""Strict JSON/YAML input parsing shared by clankerprof input surfaces.

Python's ``json`` module accepts the non-standard ``Infinity``/``-Infinity``/
``NaN`` tokens that RFC 8259 forbids and that the Rust port (serde_json)
rejects at parse time, and both stdlib parsers silently keep the last value
for duplicate object members / YAML mapping keys. Every clankerprof JSON/YAML
input goes through these loaders so both implementations fail closed on the
same inputs.
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


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Duplicate object member names are validation errors, never last-wins:
    the same multiset of members must not change meaning with ordering (the
    JSON-member-level twin of the duplicate-row and YAML duplicate-key rules).
    Fast path: dict() is C-speed; only a length mismatch triggers the scan."""
    obj = dict(pairs)
    if len(obj) != len(pairs):
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f'duplicate entry with key "{key}"')
            seen.add(key)
    return obj


JSON_DEPTH_LIMIT = 128

_JSON_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_JSON_NON_BRACKET_RE = re.compile(r"[^\[\]{}]+")


def _check_json_depth(text: str) -> None:
    """Mirror serde_json's fixed 128-level parse recursion limit.

    Python's own JSON recursion limit is interpreter-stack dependent
    (~1000), so without this the 129..~1000 nesting band would parse here
    and error in Rust. Detection runs over a C-speed bracket projection with
    string literals stripped; the slow positional rescan only runs on
    violating input.
    """
    brackets = _JSON_NON_BRACKET_RE.sub("", _JSON_STRING_RE.sub("", text))
    if len(brackets) <= JSON_DEPTH_LIMIT:
        return
    depth = 0
    for item in brackets:
        if item in "[{":
            depth += 1
            if depth > JSON_DEPTH_LIMIT:
                raise ValueError(_json_depth_message(text))
        elif depth:
            depth -= 1


def _json_depth_message(text: str) -> str:
    # serde_json reports the position just before the offending bracket;
    # the "at line N column M" suffix is engine-aligned detail.
    depth = 0
    line = 1
    column = 0
    in_string = False
    escaped = False
    for ch in text:
        if ch == "\n" and not in_string:
            line += 1
            column = 0
            continue
        column += 1
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if depth > JSON_DEPTH_LIMIT:
                return f"recursion limit exceeded at line {line} column {column - 1}"
        elif ch in "]}" and depth:
            depth -= 1
    return "recursion limit exceeded"


def parse_strict_json(text: str) -> Any:
    # Integer literals outside [i64::MIN, u64::MAX] need no guard here:
    # serde_json parses them as f64, and Python's unbounded int coerces to
    # the identical f64 in float-domain fields while integer-domain fields
    # reject both representations with the same messages (pinned by the
    # parity suite).
    _check_json_depth(text)
    return json.loads(
        text,
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object_pairs,
    )


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
      and so is the ``=`` value special;
    - explicit tags follow serde_yaml: typed core tags (``!!int``,
      ``!!float``, ``!!bool``, ``!!null``) apply the strict scalar grammars
      above (mismatches error with serde's "invalid value" core), every
      other global tag is ignored in favor of the underlying node
      (``!!binary`` stays the base64 string; ``!!set`` stays the mapping —
      deterministic, unlike a constructed Python set), and local ``!name``
      tags are rejected with the shared message both engines emit.
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

_BOOL_SCALAR_RE = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")

_add_implicit_resolver("tag:yaml.org,2002:bool", _BOOL_SCALAR_RE, list("tTfF"))

_INT_MIN = -(2**63)
_UINT_MAX = 2**64 - 1

_INT_SCALAR_RE = re.compile(
    r"^[-+]?(?:0x[0-9a-fA-F]+|0o[0-7]+|0b[01]+|0|[1-9][0-9]*)$", re.ASCII
)

_add_implicit_resolver("tag:yaml.org,2002:int", _INT_SCALAR_RE, list("-+0123456789"))

_FLOAT_PATTERN = re.compile(
    r"""^[-+]?(?:
        \.[0-9]+(?:[eE][-+]?[0-9]+)?
        |[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?
        |[0-9]+[eE][-+]?[0-9]+
    )$""",
    re.ASCII | re.VERBOSE,
)

_FLOAT_SCALAR_RE = re.compile(
    _FLOAT_PATTERN.pattern + r"|^[-+]?\.(?:inf|Inf|INF)$|^\.(?:nan|NaN|NAN)$",
    re.ASCII | re.VERBOSE,
)

# Explicit `!!float` accepts serde_yaml's wider grammar: a plain integer
# spelling is a legal explicit float even though the implicit resolver
# requires a dot or exponent.
_EXPLICIT_FLOAT_RE = re.compile(r"^[-+]?[0-9]+$", re.ASCII)

_add_implicit_resolver(
    "tag:yaml.org,2002:float", _FLOAT_SCALAR_RE, list("-+0123456789.")
)


YAML_LOCAL_TAG_MESSAGE = "YAML local tags are not supported in clankerprof inputs."


def _construct_ignoring_global_tag(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    """Mirror serde_yaml: a global (URI) tag it does not type-check is
    ignored in favor of the underlying node — `!!binary SGVsbG8=` is the
    string "SGVsbG8=", `!!set {a: null}` is the mapping (deterministic,
    unlike a Python set), `!!str [1, 2]` is the sequence. The strict mapping
    checks (string keys, duplicates) still apply.
    """
    if isinstance(node, yaml.ScalarNode):
        return str(loader.construct_scalar(node))
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_mapping(cast(yaml.MappingNode, node), deep=True)


def _construct_serde_int(loader: yaml.SafeLoader, node: yaml.Node) -> int | Any:
    if not isinstance(node, yaml.ScalarNode):
        # serde_yaml ignores typed tags on non-scalar nodes.
        return _construct_ignoring_global_tag(loader, node)
    scalar = str(loader.construct_scalar(node))
    if _INT_SCALAR_RE.fullmatch(scalar) is None:
        # Explicit `!!int` spellings outside the implicit grammar (e.g.
        # `!!int 1_0`) fail in serde_yaml with this message core.
        raise ValueError(f'invalid value: string "{scalar}", expected an integer')
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


def _construct_serde_float(loader: yaml.SafeLoader, node: yaml.Node) -> Any | Any:
    if not isinstance(node, yaml.ScalarNode):
        # serde_yaml ignores typed tags on non-scalar nodes.
        return _construct_ignoring_global_tag(loader, node)
    scalar = str(loader.construct_scalar(node))
    if (
        _FLOAT_SCALAR_RE.fullmatch(scalar) is None
        and _EXPLICIT_FLOAT_RE.fullmatch(scalar) is None
    ):
        # Explicit `!!float` spellings outside serde_yaml's float grammar
        # (e.g. `!!float 1_0`, `!!float inf`) fail with this message core;
        # plain integers (`!!float 5`) are legal explicit floats there.
        raise ValueError(f'invalid value: string "{scalar}", expected a float')
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


def _construct_serde_bool(loader: yaml.SafeLoader, node: yaml.Node) -> bool | Any:
    if not isinstance(node, yaml.ScalarNode):
        # serde_yaml ignores typed tags on non-scalar nodes.
        return _construct_ignoring_global_tag(loader, node)
    scalar = str(loader.construct_scalar(node))
    if _BOOL_SCALAR_RE.fullmatch(scalar) is None:
        # Explicit `!!bool yes` etc. fail in serde_yaml (its boolean grammar
        # is true/True/TRUE/false/False/FALSE only).
        raise ValueError(f'invalid value: string "{scalar}", expected a boolean')
    return scalar[0] in "tT"


def _construct_serde_null(loader: yaml.SafeLoader, node: yaml.Node) -> None | Any:
    if not isinstance(node, yaml.ScalarNode):
        # serde_yaml ignores typed tags on non-scalar nodes.
        return _construct_ignoring_global_tag(loader, node)
    scalar = str(loader.construct_scalar(node))
    if scalar not in ("", "~", "null", "Null", "NULL"):
        raise ValueError(f'invalid value: string "{scalar}", expected null')
    return None


def _construct_multi_global_tag(
    loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node
) -> Any:
    return _construct_ignoring_global_tag(loader, node)


def _reject_local_tag(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    # serde_yaml represents only local (`!name`) tags as Value::Tagged; the
    # Rust strict walk rejects those with this same message.
    raise ValueError(YAML_LOCAL_TAG_MESSAGE)


_StrictYamlLoader.add_constructor("tag:yaml.org,2002:int", _construct_serde_int)
_StrictYamlLoader.add_constructor("tag:yaml.org,2002:float", _construct_serde_float)
_StrictYamlLoader.add_constructor("tag:yaml.org,2002:bool", _construct_serde_bool)
_StrictYamlLoader.add_constructor("tag:yaml.org,2002:null", _construct_serde_null)
# NOT map/seq: their default constructors already match serde for their own
# node kinds and are two-phase generators, which alias cycles need to reach
# the aligned recursion-limit guard instead of PyYAML's own
# "unconstructable recursive node" error.
for _generic_tag_name in (
    "binary", "set", "omap", "pairs", "timestamp", "value", "str"
):
    _StrictYamlLoader.add_constructor(
        f"tag:yaml.org,2002:{_generic_tag_name}", _construct_ignoring_global_tag
    )
# PyYAML ships no complete type stubs for the multi-constructor API.
_add_multi_constructor = cast(
    "Callable[..., None]",
    _StrictYamlLoader.add_multi_constructor,  # pyright: ignore[reportUnknownMemberType]
)
# Every other global URI tag (tag:yaml.org,2002:python/..., tag:example.com,...)
# is ignored like serde_yaml ignores it; exact registrations above win first.
_add_multi_constructor("tag:", _construct_multi_global_tag)
# Anything left is a local tag; both engines reject with the shared message.
# PyYAML's stubs type the fallback-constructor tag as str, but the runtime
# API accepts None for the catch-all slot.
_add_constructor_untyped = cast(
    "Callable[..., None]",
    _StrictYamlLoader.add_constructor,  # pyright: ignore[reportUnknownMemberType]
)
_add_constructor_untyped(None, _reject_local_tag)


def parse_strict_yaml(text: str) -> Any:
    """`yaml.safe_load` with duplicate or non-string mapping keys rejected."""
    return yaml.load(text, Loader=_StrictYamlLoader)
