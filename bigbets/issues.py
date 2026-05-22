from __future__ import annotations

import json
import re

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import yaml

from bigbets.core import (
    BigBetsRegistry,
    JsonValue,
    ValidationFailure,
    registry_to_input_payload,
    validate_bigbets_registry,
)
from bigbets.version import BIGBETS_REGISTRY_SCHEMA_VERSION, generator_metadata

_FENCE_RE = re.compile(r"```(?P<info>[^\n`]*)\n(?P<body>.*?)```", re.DOTALL)
_COMMENT_RE = re.compile(
    r"<!--\s*bigbets:idea-family\s*(?P<body>.*?)\s*-->",
    re.DOTALL | re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_PRIORITY_RE = re.compile(r"^P[0-9]+$")
_STATUS_VALUES = {
    "active",
    "candidate",
    "parked",
    "blocked",
    "shipped",
    "rejected",
    "superseded",
    "closed",
}
_ROLE_VALUES = {
    "ideas-lane",
    "wip",
    "evidence",
    "proof",
    "follow-up",
    "blocked",
    "shipped",
}
_LINK_KIND_VALUES = {
    "artifact",
    "board",
    "doc",
    "evidence",
    "issue",
    "project",
    "pr",
    "tracker",
}


def load_issue_family_payloads(path: Path) -> list[dict[str, JsonValue]]:
    text = path.read_text(encoding="utf-8")
    payload = _load_payload(text, source_name=str(path))
    return [_idea_family_from_issue_object(item) for item in _issue_objects(payload)]


def issue_family_patch(families: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
            "generator": cast(dict[str, JsonValue], generator_metadata()),
            "idea_families": [dict(family) for family in families],
        },
    )


def merge_issue_families(
    registry: BigBetsRegistry,
    families: Sequence[Mapping[str, JsonValue]],
) -> BigBetsRegistry:
    payload = registry_to_input_payload(registry)
    big_bets = cast(list[dict[str, JsonValue]], payload["big_bets"])
    existing_families = cast(list[dict[str, JsonValue]], payload["idea_families"])
    bet_ids = {str(bet["id"]) for bet in big_bets}
    by_issue = {
        _json_positive_int(family.get("issue"), "existing family issue"): dict(family)
        for family in existing_families
        if isinstance(family.get("issue"), int)
    }
    next_rank_by_bet = _next_rank_by_bet(existing_families)

    for family in families:
        normalized = dict(family)
        big_bet = str(normalized.get("big_bet") or "")
        if big_bet not in bet_ids:
            raise ValidationFailure(
                f"Cannot merge issue #{normalized.get('issue')}: unknown big_bet {big_bet!r}."
            )
        if normalized.get("rank") is None:
            normalized["rank"] = next_rank_by_bet.get(big_bet, 1)
            next_rank_by_bet[big_bet] = (
                _json_positive_int(normalized.get("rank"), "family rank") + 1
            )
        by_issue[_json_positive_int(normalized.get("issue"), "family issue")] = normalized

    payload["idea_families"] = cast(list[JsonValue], list(by_issue.values()))
    return validate_bigbets_registry(cast(dict[str, object], payload))


def _next_rank_by_bet(
    families: Sequence[Mapping[str, JsonValue]],
) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for family in families:
        big_bet = str(family.get("big_bet") or "")
        rank = family.get("rank")
        if isinstance(rank, int):
            ranks[big_bet] = max(ranks.get(big_bet, 0), rank)
    return {big_bet: rank + 1 for big_bet, rank in ranks.items()}


def _load_payload(text: str, *, source_name: str) -> object:
    stripped = text.lstrip()
    if not stripped:
        raise ValidationFailure(f"{source_name} was empty.")
    if "bigbets:idea-family" in text or "bigbets-idea-family" in text:
        return _load_markdown_issue(text, source_name=source_name)
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            return json.loads(text)
        return yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationFailure(f"Failed to parse {source_name}: {exc}") from exc


def _load_markdown_issue(text: str, *, source_name: str) -> dict[str, JsonValue]:
    metadata = _embedded_metadata(text, source_name=source_name)
    title = _first_markdown_heading(text)
    if title and "title" not in metadata:
        metadata["title"] = title
    return metadata


def _embedded_metadata(text: str, *, source_name: str) -> dict[str, JsonValue]:
    for match in _COMMENT_RE.finditer(text):
        return _load_metadata_block(match.group("body"), source_name=source_name)
    for match in _FENCE_RE.finditer(text):
        if "bigbets-idea-family" in match.group("info"):
            return _load_metadata_block(match.group("body"), source_name=source_name)
    raise ValidationFailure(
        f"{source_name} does not contain a bigbets idea-family metadata block."
    )


def _load_metadata_block(text: str, *, source_name: str) -> dict[str, JsonValue]:
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationFailure(
            f"Failed to parse bigbets metadata in {source_name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(
            f"Expected bigbets metadata in {source_name} to be a mapping."
        )
    return cast(dict[str, JsonValue], payload)


def _issue_objects(payload: object) -> list[Mapping[str, object]]:
    if isinstance(payload, list):
        sequence = cast(list[object], payload)
        return [_mapping(item, "issue entry") for item in sequence]
    if isinstance(payload, dict):
        mapping = cast(dict[str, object], payload)
        issues = mapping.get("issues")
        if isinstance(issues, list):
            return [_mapping(item, "issues entry") for item in cast(list[object], issues)]
        return [_mapping(mapping, "issue entry")]
    raise ValidationFailure("Issue import input must be a mapping or list.")


def _idea_family_from_issue_object(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    embedded = _body_metadata(raw)
    source: dict[str, object] = {**dict(raw), **embedded}
    artifact = _optional_string_any(source, "artifact", "ideas_json", "seed")
    role = _optional_string_any(source, "role")
    if role is None:
        role = "ideas-lane" if artifact and ".ideas.json" in artifact else "follow-up"
    role = _choice(role, "role", _ROLE_VALUES)
    status = _choice(
        _optional_string_any(source, "status") or "candidate",
        "status",
        _STATUS_VALUES,
    )
    family = {
        "issue": _issue_number(source),
        "slug": _optional_identifier_any(source, "slug"),
        "title": _required_string_any(source, "idea_family_title", "title"),
        "big_bet": _required_identifier_any(source, "big_bet"),
        "priority": _required_priority_any(source, "priority"),
        "rank": _optional_positive_int_any(source, "rank"),
        "status": status,
        "role": role,
        "next_action": _optional_string_any(source, "next_action"),
        "artifact": artifact,
        "url": _optional_string_any(source, "url", "html_url"),
        "links": _links(source.get("links")),
    }
    return cast(
        dict[str, JsonValue],
        {key: value for key, value in family.items() if value not in (None, [])},
    )


def _body_metadata(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    body = raw.get("body")
    if not isinstance(body, str):
        return {}
    if "bigbets:idea-family" not in body and "bigbets-idea-family" not in body:
        return {}
    return _embedded_metadata(body, source_name="issue body")


def _issue_number(source: Mapping[str, object]) -> int:
    value = source.get("issue", source.get("number"))
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
        return int(value)
    raise ValidationFailure("Imported idea family must include a positive issue number.")


def _required_string_any(source: Mapping[str, object], *keys: str) -> str:
    value = _optional_string_any(source, *keys)
    if value is None:
        raise ValidationFailure(f"Imported idea family is missing {keys[0]!r}.")
    return value


def _optional_string_any(source: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValidationFailure(f"Expected {key!r} to be a string.")
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def _required_identifier_any(source: Mapping[str, object], *keys: str) -> str:
    value = _required_string_any(source, *keys)
    if _IDENTIFIER_RE.match(value) is None:
        raise ValidationFailure(f"Expected {keys[0]!r} to be a bigbets identifier.")
    return value


def _optional_identifier_any(source: Mapping[str, object], *keys: str) -> str | None:
    value = _optional_string_any(source, *keys)
    if value is None:
        return None
    if _IDENTIFIER_RE.match(value) is None:
        raise ValidationFailure(f"Expected {keys[0]!r} to be a bigbets identifier.")
    return value


def _required_priority_any(source: Mapping[str, object], *keys: str) -> str:
    value = _required_string_any(source, *keys)
    if _PRIORITY_RE.match(value) is None:
        raise ValidationFailure(f"Expected {keys[0]!r} to look like P0, P1, ...")
    return value


def _optional_positive_int_any(
    source: Mapping[str, object], *keys: str
) -> int | None:
    for key in keys:
        value = source.get(key)
        if value is None:
            continue
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit() and int(value) > 0:
            return int(value)
        raise ValidationFailure(f"Expected {key!r} to be a positive integer.")
    return None


def _links(value: object) -> list[dict[str, JsonValue]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValidationFailure("Expected 'links' to be a list.")
    links: list[dict[str, JsonValue]] = []
    for index, item in enumerate(cast(list[object], value)):
        mapping = _mapping(item, f"links[{index}]")
        link = {
            "label": _required_string_any(mapping, "label"),
            "url": _required_string_any(mapping, "url"),
            "kind": _optional_choice_any(mapping, _LINK_KIND_VALUES, "kind"),
        }
        links.append(cast(dict[str, JsonValue], {k: v for k, v in link.items() if v}))
    return links


def _json_positive_int(value: JsonValue | None, label: str) -> int:
    if isinstance(value, int) and value > 0:
        return value
    raise ValidationFailure(f"Expected {label} to be a positive integer.")


def _optional_choice_any(
    source: Mapping[str, object],
    choices: set[str],
    *keys: str,
) -> str | None:
    value = _optional_string_any(source, *keys)
    if value is None:
        return None
    return _choice(value, keys[0], choices)


def _choice(value: str, label: str, choices: set[str]) -> str:
    if value not in choices:
        raise ValidationFailure(
            f"Expected {label!r} to be one of: {', '.join(sorted(choices))}."
        )
    return value


def _first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValidationFailure(f"Expected {label} to be a mapping.")
    return cast(Mapping[str, Any], value)


__all__ = [
    "issue_family_patch",
    "load_issue_family_payloads",
    "merge_issue_families",
]
