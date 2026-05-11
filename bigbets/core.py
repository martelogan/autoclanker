from __future__ import annotations

import csv
import hashlib
import html
import json
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from io import StringIO
from pathlib import Path
from typing import Any, TypeAlias, cast

import yaml

from bigbets.version import (
    BIGBETS_ARTIFACT_SCHEMA_VERSION,
    BIGBETS_REGISTRY_SCHEMA_VERSION,
    BIGBETS_VERSION,
    generator_metadata,
)

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ValidationFailure(ValueError):
    """Raised when a bigbets registry is invalid or cannot be loaded."""


_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_PRIORITY_RE = re.compile(r"^P([0-9]+)$")
_KNOWN_STATUSES = {
    "active",
    "candidate",
    "parked",
    "blocked",
    "shipped",
    "rejected",
    "superseded",
    "closed",
}
_KNOWN_ROLES = {
    "ideas-lane",
    "wip",
    "evidence",
    "proof",
    "follow-up",
    "blocked",
    "shipped",
}
_CARD_WIDTH = 350
_CARD_HEIGHT = 156
_CARD_GAP = 46
_WAVE_GAP = 116
_MARGIN_X = 78
_MARGIN_Y = 132


@dataclass(frozen=True, slots=True)
class BigBet:
    id: str
    title: str
    priority: str
    rank: int | None
    wave: int
    status: str
    narrative: str
    near_term_win: str
    long_term_unlock: str
    next_action: str | None
    confidence: str | None
    risk: str | None
    depends_on: tuple[str, ...]
    unlocks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdeaFamily:
    issue: int
    title: str
    big_bet: str
    priority: str
    rank: int | None
    status: str
    role: str | None
    next_action: str | None
    artifact: str | None
    url: str | None


@dataclass(frozen=True, slots=True)
class BigBetsRegistry:
    title: str
    description: str | None
    updated_at: str | None
    big_bets: tuple[BigBet, ...]
    idea_families: tuple[IdeaFamily, ...]
    max_p0_big_bets: int


@dataclass(frozen=True, slots=True)
class RenderedBigBets:
    registry_json: str
    rankings_csv: str
    markdown: str
    mermaid: str
    excalidraw: str
    svg: str
    html: str


def load_serialized_payload_from_text(
    text: str,
    *,
    source_name: str = "<stdin>",
) -> dict[str, object]:
    stripped = text.lstrip()
    if not stripped:
        raise ValidationFailure(f"{source_name} was empty.")
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationFailure(f"Failed to parse {source_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(f"{source_name} must contain a top-level mapping.")
    return cast(dict[str, object], payload)


def load_serialized_payload(path: Path) -> dict[str, object]:
    return load_serialized_payload_from_text(
        path.read_text(encoding="utf-8"),
        source_name=str(path),
    )


def to_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if is_dataclass(value):
        result: dict[str, JsonValue] = {}
        for field in fields(value):
            item = to_json_value(getattr(value, field.name))
            if item is not None:
                result[field.name] = item
        return result

    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            converted = to_json_value(item)
            if converted is not None:
                result[str(key)] = converted
        return result

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        sequence = cast(Sequence[Any], value)
        return [to_json_value(item) for item in sequence]

    raise TypeError(f"Unsupported JSON conversion for value: {type(value)!r}")


def load_bigbets_registry(path: Path) -> BigBetsRegistry:
    return validate_bigbets_registry(
        load_serialized_payload(path), source_name=str(path)
    )


def validate_bigbets_registry(
    payload: dict[str, object], *, source_name: str = "<registry>"
) -> BigBetsRegistry:
    schema_version = payload.get("schema_version")
    if schema_version is not None and schema_version != BIGBETS_REGISTRY_SCHEMA_VERSION:
        raise ValidationFailure(
            f"{source_name} has unsupported schema_version {schema_version!r}; "
            f"expected {BIGBETS_REGISTRY_SCHEMA_VERSION!r}."
        )

    metadata = _optional_mapping(payload.get("metadata"), "metadata") or {}
    title = _optional_string(metadata, "title") or "Big Bets Portfolio"
    description = _optional_string(metadata, "description")
    updated_at = _optional_string(metadata, "updated_at")
    max_p0_big_bets = _optional_positive_int(metadata, "max_p0_big_bets") or 3

    big_bets = tuple(
        _parse_big_bet(item, index)
        for index, item in enumerate(
            _required_sequence(payload.get("big_bets"), "big_bets")
        )
    )
    idea_families = tuple(
        _parse_idea_family(item, index)
        for index, item in enumerate(
            _required_sequence(payload.get("idea_families"), "idea_families")
        )
    )
    registry = BigBetsRegistry(
        title=title,
        description=description,
        updated_at=updated_at,
        big_bets=tuple(sorted(big_bets, key=_big_bet_sort_key)),
        idea_families=tuple(sorted(idea_families, key=_idea_family_sort_key)),
        max_p0_big_bets=max_p0_big_bets,
    )
    _validate_registry_semantics(registry, source_name=source_name)
    return registry


def normalize_bigbets_registry(registry: BigBetsRegistry) -> dict[str, JsonValue]:
    issues_by_bet: dict[str, list[dict[str, JsonValue]]] = {}
    for family in registry.idea_families:
        issues_by_bet.setdefault(family.big_bet, []).append(
            cast(dict[str, JsonValue], to_json_value(family))
        )

    big_bets: list[dict[str, JsonValue]] = []
    for bet in registry.big_bets:
        families = issues_by_bet.get(bet.id, [])
        bet_payload = cast(dict[str, JsonValue], to_json_value(bet))
        bet_payload["idea_family_issues"] = [
            cast(int, family["issue"]) for family in families
        ]
        bet_payload["idea_family_count"] = len(families)
        big_bets.append(bet_payload)

    edges = _normalized_edges(registry)
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
            "artifact_schema_version": BIGBETS_ARTIFACT_SCHEMA_VERSION,
            "generator": cast(dict[str, JsonValue], generator_metadata()),
            "metadata": {
                "title": registry.title,
                "description": registry.description,
                "updated_at": registry.updated_at,
                "max_p0_big_bets": registry.max_p0_big_bets,
            },
            "summary": {
                "big_bet_count": len(registry.big_bets),
                "idea_family_count": len(registry.idea_families),
                "edge_count": len(edges),
                "p0_big_bet_count": sum(
                    1 for bet in registry.big_bets if bet.priority == "P0"
                ),
            },
            "big_bets": big_bets,
            "idea_families": cast(
                list[JsonValue], to_json_value(list(registry.idea_families))
            ),
            "edges": cast(list[JsonValue], to_json_value(edges)),
        },
    )


def registry_to_input_payload(registry: BigBetsRegistry) -> dict[str, JsonValue]:
    big_bets: list[JsonValue] = []
    for bet in registry.big_bets:
        payload = cast(dict[str, JsonValue], to_json_value(bet))
        if not bet.depends_on:
            payload.pop("depends_on", None)
        if not bet.unlocks:
            payload.pop("unlocks", None)
        big_bets.append(payload)

    idea_families: list[JsonValue] = [
        to_json_value(family) for family in registry.idea_families
    ]
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
            "metadata": {
                "title": registry.title,
                "description": registry.description,
                "updated_at": registry.updated_at,
                "max_p0_big_bets": registry.max_p0_big_bets,
            },
            "big_bets": big_bets,
            "idea_families": idea_families,
        },
    )


def render_bigbets(registry: BigBetsRegistry) -> RenderedBigBets:
    normalized = normalize_bigbets_registry(registry)
    mermaid = render_mermaid(registry)
    excalidraw = render_excalidraw(registry)
    svg = render_svg(registry)
    return RenderedBigBets(
        registry_json=json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        rankings_csv=render_rankings_csv(registry),
        markdown=render_markdown(registry, mermaid),
        mermaid=mermaid,
        excalidraw=excalidraw,
        svg=svg,
        html=render_html(registry, svg, normalized),
    )


def _artifact_metadata(
    *, site_schema_version: str | None = None
) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "schema_version": BIGBETS_REGISTRY_SCHEMA_VERSION,
        "artifact_schema_version": BIGBETS_ARTIFACT_SCHEMA_VERSION,
        "generator": cast(dict[str, JsonValue], generator_metadata()),
    }
    if site_schema_version is not None:
        metadata["site_schema_version"] = site_schema_version
    return metadata


def write_bigbets_artifacts(registry: BigBetsRegistry, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = render_bigbets(registry)
    artifacts = {
        "big_bets.artifact_metadata.json": render_artifact_metadata_json(),
        "big_bets.registry.json": rendered.registry_json,
        "big_bets.rankings.csv": rendered.rankings_csv,
        "big_bets.md": rendered.markdown,
        "big_bets.mmd": rendered.mermaid,
        "big_bets.excalidraw": rendered.excalidraw,
        "big_bets.svg": rendered.svg,
        "index.html": rendered.html,
    }
    written: list[Path] = []
    for name, content in artifacts.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def render_artifact_metadata_json(*, site_schema_version: str | None = None) -> str:
    return (
        json.dumps(
            _artifact_metadata(site_schema_version=site_schema_version),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_rankings_csv(registry: BigBetsRegistry) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "kind",
            "big_bet_priority",
            "big_bet_order",
            "wave",
            "big_bet_id",
            "big_bet_title",
            "big_bet_status",
            "lane_priority",
            "lane_order",
            "issue",
            "idea_family_title",
            "idea_family_status",
            "role",
            "next_action",
            "artifact",
            "url",
            "schema_version",
            "generator_version",
        ]
    )
    families_by_bet = _families_by_bet(registry)
    for bet in registry.big_bets:
        families = families_by_bet.get(bet.id, ())
        if not families:
            writer.writerow(
                [
                    "bet",
                    bet.priority,
                    bet.rank or "",
                    bet.wave,
                    bet.id,
                    bet.title,
                    bet.status,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    bet.next_action or "",
                    "",
                    "",
                    BIGBETS_REGISTRY_SCHEMA_VERSION,
                    BIGBETS_VERSION,
                ]
            )
        for family in families:
            writer.writerow(
                [
                    "family",
                    bet.priority,
                    bet.rank or "",
                    bet.wave,
                    bet.id,
                    bet.title,
                    bet.status,
                    family.priority,
                    family.rank or "",
                    family.issue,
                    family.title,
                    family.status,
                    family.role or "",
                    family.next_action or bet.next_action or "",
                    family.artifact or "",
                    family.url or "",
                    BIGBETS_REGISTRY_SCHEMA_VERSION,
                    BIGBETS_VERSION,
                ]
            )
    return buffer.getvalue()


def render_mermaid(registry: BigBetsRegistry) -> str:
    lines = [
        f"%% schema_version={BIGBETS_REGISTRY_SCHEMA_VERSION} generator=bigbets@{BIGBETS_VERSION}",
        "flowchart TD",
    ]
    by_wave = _big_bets_by_wave(registry)
    for wave, bets in by_wave.items():
        lines.append(f"  subgraph wave_{wave}[{_priority_for_wave(wave)}]")
        for bet in bets:
            label = _mermaid_label(bet, registry)
            lines.append(f'    {_node_id(bet.id)}["{label}"]')
        lines.append("  end")
    for edge in _normalized_edges(registry):
        lines.append(f"  {_node_id(edge['from'])} --> {_node_id(edge['to'])}")
    return "\n".join(lines) + "\n"


def render_markdown(registry: BigBetsRegistry, mermaid: str | None = None) -> str:
    lines = [
        f"# {registry.title}",
        "",
        f"Generated by `bigbets {BIGBETS_VERSION}`.",
        "",
        f"Registry schema: `{BIGBETS_REGISTRY_SCHEMA_VERSION}`.",
        "",
    ]
    if registry.description:
        lines.extend([registry.description, ""])
    if registry.updated_at:
        lines.extend([f"Updated: `{registry.updated_at}`", ""])
    lines.extend(
        [
            "## Priority Queue",
            "",
            "| P layer | Big bet | Status | Idea families | Next action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    families_by_bet = _families_by_bet(registry)
    for bet in registry.big_bets:
        family_links = ", ".join(
            _markdown_issue_link(family) for family in families_by_bet.get(bet.id, ())
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    bet.priority,
                    _escape_markdown_table(bet.title),
                    bet.status,
                    family_links or "-",
                    _escape_markdown_table(bet.next_action or "-"),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Big Bets", ""])
    for bet in registry.big_bets:
        lines.extend(
            [
                f"### {bet.priority}: {bet.title}",
                "",
                bet.narrative,
                "",
                f"- **Near-term win:** {bet.near_term_win}",
                f"- **Long-term unlock:** {bet.long_term_unlock}",
                f"- **Status:** {bet.status}",
                f"- **Next action:** {bet.next_action or '-'}",
                f"- **Confidence:** {bet.confidence or '-'}",
                f"- **Risk:** {bet.risk or '-'}",
                "",
            ]
        )
    if mermaid is not None:
        lines.extend(["## Graph", "", "```mermaid", mermaid.rstrip(), "```", ""])
    return "\n".join(lines)


def render_svg(registry: BigBetsRegistry) -> str:
    by_wave = _big_bets_by_wave(registry)
    max_cards = max((len(items) for items in by_wave.values()), default=1)
    width = max(
        1040,
        (_MARGIN_X * 2) + max_cards * _CARD_WIDTH + max(0, max_cards - 1) * _CARD_GAP,
    )
    height = max(
        520,
        (_MARGIN_Y * 2)
        + len(by_wave) * _CARD_HEIGHT
        + max(0, len(by_wave) - 1) * _WAVE_GAP,
    )
    positions: dict[str, tuple[int, int]] = {}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{_xml(registry.title)}">',
        f"<metadata>{_xml(json.dumps(_artifact_metadata(), sort_keys=True))}</metadata>",
        "<defs>",
        '<pattern id="grid" width="56" height="56" patternUnits="userSpaceOnUse"><path d="M 56 0 L 0 0 0 56" fill="none" stroke="#ece5d8" stroke-width="0.9"/></pattern>',
        '<filter id="shadow" x="-8%" y="-8%" width="116%" height="132%"><feDropShadow dx="0" dy="8" stdDeviation="6" flood-color="#1e1e1e" flood-opacity="0.105"/></filter>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="8.7" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1e1e1e"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="#fdfbf4"/>',
        '<circle cx="190" cy="135" r="230" fill="#dbeafe" opacity="0.22"/>',
        f'<circle cx="{width - 170}" cy="145" r="250" fill="#d8f3dc" opacity="0.21"/>',
        '<rect width="100%" height="100%" fill="url(#grid)" opacity="0.66"/>',
        f'<text x="{_MARGIN_X}" y="52" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="31" font-weight="400" fill="#1e1e1e">{_xml(registry.title)}</text>',
        f'<text x="{_MARGIN_X}" y="82" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="16" fill="#4b5563">Priority layers flow top-down. Dragging changes layer placement; arrows are explicit dependencies.</text>',
    ]
    for wave_index, (wave, bets) in enumerate(by_wave.items()):
        y = _MARGIN_Y + wave_index * (_CARD_HEIGHT + _WAVE_GAP)
        parts.append(
            f'<text x="{_MARGIN_X}" y="{y - 22}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="18" font-weight="400" fill="#4b5563">{_xml(_priority_for_wave(wave))}</text>'
        )
        for card_index, bet in enumerate(bets):
            x = _MARGIN_X + card_index * (_CARD_WIDTH + _CARD_GAP)
            positions[bet.id] = (x, y)
    for edge in _normalized_edges(registry):
        source = positions.get(edge["from"])
        target = positions.get(edge["to"])
        if source is None or target is None:
            continue
        sx = source[0] + _CARD_WIDTH / 2
        sy = source[1] + _CARD_HEIGHT
        tx = target[0] + _CARD_WIDTH / 2
        ty = target[1]
        bend = max(46, abs(ty - sy) / 2)
        dash = ' stroke-dasharray="8 8"' if edge["kind"] == "depends_on" else ""
        parts.extend(_svg_edge(edge["from"], edge["to"], sx, sy, tx, ty, bend, dash))
    families_by_bet = _families_by_bet(registry)
    for bet in registry.big_bets:
        x, y = positions[bet.id]
        color = _priority_color(bet.priority)
        parts.extend(_svg_card(bet, families_by_bet.get(bet.id, ()), x, y, color))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_excalidraw(registry: BigBetsRegistry) -> str:
    by_wave = _big_bets_by_wave(registry)
    positions: dict[str, tuple[int, int]] = {}
    elements: list[dict[str, JsonValue]] = []
    families_by_bet = _families_by_bet(registry)
    for wave_index, (_wave, bets) in enumerate(by_wave.items()):
        y = _MARGIN_Y + wave_index * (_CARD_HEIGHT + _WAVE_GAP)
        for card_index, bet in enumerate(bets):
            x = _MARGIN_X + card_index * (_CARD_WIDTH + _CARD_GAP)
            positions[bet.id] = (x, y)
            fill = _priority_fill(bet.priority)
            element_id = _excalidraw_id(f"rect:{bet.id}")
            text_id = _excalidraw_id(f"text:{bet.id}")
            elements.append(
                {
                    "id": element_id,
                    "type": "rectangle",
                    "x": x,
                    "y": y,
                    "width": _CARD_WIDTH,
                    "height": _CARD_HEIGHT,
                    "angle": 0,
                    "strokeColor": "#1e1e1e",
                    "backgroundColor": fill,
                    "fillStyle": "solid",
                    "strokeWidth": 1,
                    "strokeStyle": "solid",
                    "roughness": 2,
                    "opacity": 100,
                    "groupIds": [],
                    "frameId": None,
                    "roundness": {"type": 3},
                    "seed": _stable_seed(element_id),
                    "version": 1,
                    "versionNonce": _stable_seed(f"nonce:{element_id}"),
                    "isDeleted": False,
                    "boundElements": [{"type": "text", "id": text_id}],
                    "updated": 1,
                    "link": None,
                    "locked": False,
                }
            )
            issue_label = " ".join(
                f"#{family.issue}" for family in families_by_bet.get(bet.id, ())
            )
            text = f"{bet.priority} / {bet.title}\n{bet.status}"
            if issue_label:
                text = f"{text}\n{issue_label}"
            elements.append(
                {
                    "id": text_id,
                    "type": "text",
                    "x": x + 22,
                    "y": y + 22,
                    "width": _CARD_WIDTH - 44,
                    "height": 90,
                    "angle": 0,
                    "strokeColor": "#1e1e1e",
                    "backgroundColor": "transparent",
                    "fillStyle": "solid",
                    "strokeWidth": 1,
                    "strokeStyle": "solid",
                    "roughness": 1,
                    "opacity": 100,
                    "groupIds": [],
                    "frameId": None,
                    "roundness": None,
                    "seed": _stable_seed(text_id),
                    "version": 1,
                    "versionNonce": _stable_seed(f"nonce:{text_id}"),
                    "isDeleted": False,
                    "boundElements": None,
                    "updated": 1,
                    "link": None,
                    "locked": False,
                    "fontSize": 16,
                    "fontFamily": 5,
                    "text": text,
                    "rawText": text,
                    "textAlign": "left",
                    "verticalAlign": "top",
                    "containerId": element_id,
                    "originalText": text,
                    "lineHeight": 1.25,
                }
            )
    for edge in _normalized_edges(registry):
        source = positions.get(edge["from"])
        target = positions.get(edge["to"])
        if source is None or target is None:
            continue
        sx = source[0] + _CARD_WIDTH / 2
        sy = source[1] + _CARD_HEIGHT
        tx = target[0] + _CARD_WIDTH / 2
        ty = target[1]
        arrow_id = _excalidraw_id(f"arrow:{edge['from']}:{edge['to']}")
        elements.append(
            {
                "id": arrow_id,
                "type": "arrow",
                "x": sx,
                "y": sy,
                "width": tx - sx,
                "height": ty - sy,
                "angle": 0,
                "strokeColor": "#1e1e1e" if edge["kind"] == "unlocks" else "#5d6c84",
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "strokeWidth": 1,
                "strokeStyle": "solid" if edge["kind"] == "unlocks" else "dashed",
                "roughness": 2,
                "opacity": 100,
                "groupIds": [],
                "frameId": None,
                "roundness": {"type": 2},
                "seed": _stable_seed(arrow_id),
                "version": 1,
                "versionNonce": _stable_seed(f"nonce:{arrow_id}"),
                "isDeleted": False,
                "boundElements": None,
                "updated": 1,
                "link": None,
                "locked": False,
                "points": [[0, 0], [tx - sx, ty - sy]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
            }
        )
    payload = {
        "type": "excalidraw",
        "version": 2,
        "source": f"bigbets {BIGBETS_VERSION}",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#fdfbf4",
            "gridSize": 56,
            "currentItemFontFamily": 5,
        },
        "files": {},
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def render_html(
    registry: BigBetsRegistry, svg: str, normalized: dict[str, JsonValue]
) -> str:
    data_json = json.dumps(normalized, sort_keys=True)
    cards: list[str] = []
    families_by_bet = _families_by_bet(registry)
    for bet in registry.big_bets:
        families = "".join(
            f'<li><a href="{_attr(family.url or "#")}">#{family.issue}</a> {_xml(family.title)}</li>'
            for family in families_by_bet.get(bet.id, ())
        )
        cards.append(
            f"""
<section class="card priority-{_attr(bet.priority.lower())}">
  <div class="card-meta">{_xml(bet.priority)} · {_xml(bet.status)}</div>
  <h2>{_xml(bet.title)}</h2>
  <p>{_xml(bet.narrative)}</p>
  <dl>
    <dt>Near-term win</dt><dd>{_xml(bet.near_term_win)}</dd>
    <dt>Long-term unlock</dt><dd>{_xml(bet.long_term_unlock)}</dd>
    <dt>Next action</dt><dd>{_xml(bet.next_action or "-")}</dd>
  </dl>
  <ul>{families}</ul>
</section>""".strip()
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="generator" content="bigbets {BIGBETS_VERSION}">
  <meta name="bigbets-registry-schema-version" content="{BIGBETS_REGISTRY_SCHEMA_VERSION}">
  <meta name="bigbets-artifact-schema-version" content="{BIGBETS_ARTIFACT_SCHEMA_VERSION}">
  <title>{_xml(registry.title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #cbd5e1;
      --paper: #f8fafc;
      --card: #ffffff;
      --accent: #0f766e;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at 15% 10%, #ccfbf1 0, transparent 28rem), var(--paper);
      color: var(--ink);
      font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header, main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    header h1 {{ font-size: clamp(2rem, 5vw, 4.5rem); line-height: 0.95; letter-spacing: -0.06em; margin: 24px 0 12px; }}
    header p {{ max-width: 760px; color: var(--muted); font-size: 1.05rem; }}
    .graph {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 24px; background: rgba(255,255,255,0.72); padding: 10px; box-shadow: 0 16px 60px rgba(15,23,42,0.08); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 18px; margin-top: 28px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 22px; padding: 18px; box-shadow: 0 10px 30px rgba(15,23,42,0.07); }}
    .card-meta {{ color: var(--accent); font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }}
    h2 {{ margin: 8px 0; letter-spacing: -0.03em; line-height: 1.05; }}
    dl {{ display: grid; grid-template-columns: 8rem 1fr; gap: 6px 12px; }}
    dt {{ color: var(--muted); font-weight: 700; }}
    dd {{ margin: 0; }}
    a {{ color: #0369a1; text-decoration-thickness: 0.08em; }}
    code {{ background: #e2e8f0; padding: 0.1rem 0.25rem; border-radius: 0.3rem; }}
  </style>
</head>
<body>
  <header>
    <p>Generated by <code>bigbets {BIGBETS_VERSION}</code>. Schema <code>{BIGBETS_REGISTRY_SCHEMA_VERSION}</code>.</p>
    <h1>{_xml(registry.title)}</h1>
    <p>{_xml(registry.description or "Ranked big-bet portfolio generated from a structured registry.")}</p>
  </header>
  <main>
    <section class="graph">{svg}</section>
    <section class="grid">{"".join(cards)}</section>
  </main>
  <script type="application/json" id="bigbets-registry">{html.escape(data_json)}</script>
</body>
</html>
"""


def _parse_big_bet(item: object, index: int) -> BigBet:
    mapping = _required_mapping(item, f"big_bets[{index}]")
    bet_id = _required_identifier(mapping, "id", f"big_bets[{index}]")
    return BigBet(
        id=bet_id,
        title=_required_string(mapping, "title", f"big_bets[{index}]"),
        priority=_required_priority(mapping, "priority", f"big_bets[{index}]"),
        rank=_optional_positive_int(mapping, "rank"),
        wave=_required_positive_int(mapping, "wave", f"big_bets[{index}]"),
        status=_required_status(mapping, "status", f"big_bets[{index}]"),
        narrative=_required_string(mapping, "narrative", f"big_bets[{index}]"),
        near_term_win=_required_string(mapping, "near_term_win", f"big_bets[{index}]"),
        long_term_unlock=_required_string(
            mapping, "long_term_unlock", f"big_bets[{index}]"
        ),
        next_action=_optional_string(mapping, "next_action"),
        confidence=_optional_string(mapping, "confidence"),
        risk=_optional_string(mapping, "risk"),
        depends_on=tuple(
            _required_identifier_value(value, f"big_bets[{index}].depends_on")
            for value in _optional_sequence(mapping.get("depends_on"), "depends_on")
        ),
        unlocks=tuple(
            _required_identifier_value(value, f"big_bets[{index}].unlocks")
            for value in _optional_sequence(mapping.get("unlocks"), "unlocks")
        ),
    )


def _parse_idea_family(item: object, index: int) -> IdeaFamily:
    mapping = _required_mapping(item, f"idea_families[{index}]")
    return IdeaFamily(
        issue=_required_positive_int(mapping, "issue", f"idea_families[{index}]"),
        title=_required_string(mapping, "title", f"idea_families[{index}]"),
        big_bet=_required_identifier(mapping, "big_bet", f"idea_families[{index}]"),
        priority=_required_priority(mapping, "priority", f"idea_families[{index}]"),
        rank=_optional_positive_int(mapping, "rank"),
        status=_required_status(mapping, "status", f"idea_families[{index}]"),
        role=_optional_string(mapping, "role"),
        next_action=_optional_string(mapping, "next_action"),
        artifact=_optional_string(mapping, "artifact"),
        url=_optional_string(mapping, "url"),
    )


def _validate_registry_semantics(
    registry: BigBetsRegistry, *, source_name: str
) -> None:
    bet_ids = [bet.id for bet in registry.big_bets]
    duplicate_bets = sorted({bet_id for bet_id in bet_ids if bet_ids.count(bet_id) > 1})
    if duplicate_bets:
        raise ValidationFailure(
            f"{source_name} has duplicate big_bets ids: {', '.join(duplicate_bets)}"
        )
    bet_id_set = set(bet_ids)
    issues = [family.issue for family in registry.idea_families]
    duplicate_issues = sorted({issue for issue in issues if issues.count(issue) > 1})
    if duplicate_issues:
        raise ValidationFailure(
            f"{source_name} maps issue(s) more than once: {duplicate_issues}"
        )
    unknown_bets = sorted(
        {
            family.big_bet
            for family in registry.idea_families
            if family.big_bet not in bet_id_set
        }
    )
    if unknown_bets:
        raise ValidationFailure(
            f"{source_name} idea_families reference unknown big_bets: {', '.join(unknown_bets)}"
        )
    bad_edges = sorted(
        {
            target
            for bet in registry.big_bets
            for target in (*bet.depends_on, *bet.unlocks)
            if target not in bet_id_set
        }
    )
    if bad_edges:
        raise ValidationFailure(
            f"{source_name} big_bet edges reference unknown ids: {', '.join(bad_edges)}"
        )
    p0_count = sum(1 for bet in registry.big_bets if bet.priority == "P0")
    if p0_count > registry.max_p0_big_bets:
        raise ValidationFailure(
            f"{source_name} has {p0_count} P0 big bets; max_p0_big_bets is {registry.max_p0_big_bets}."
        )
    for bet in registry.big_bets:
        priority_value = _priority_rank(bet.priority)
        if priority_value <= 1 and not bet.next_action:
            raise ValidationFailure(
                f"{source_name} big_bet {bet.id!r} is {bet.priority} and must set next_action."
            )
        if bet.id in bet.depends_on or bet.id in bet.unlocks:
            raise ValidationFailure(
                f"{source_name} big_bet {bet.id!r} links to itself."
            )
        expected_priority = _priority_for_wave(bet.wave)
        if bet.priority != expected_priority:
            raise ValidationFailure(
                f"{source_name} big_bet {bet.id!r} has priority {bet.priority!r} "
                f"but wave {bet.wave} requires {expected_priority!r}."
            )
    for family in registry.idea_families:
        if family.role is not None and family.role not in _KNOWN_ROLES:
            raise ValidationFailure(
                f"{source_name} idea_family #{family.issue} has unsupported role "
                f"{family.role!r}; expected one of: {', '.join(sorted(_KNOWN_ROLES))}."
            )
    families_by_bet = _families_by_bet(registry)
    empty_active = [
        bet.id
        for bet in registry.big_bets
        if bet.status not in {"candidate", "parked", "superseded", "closed"}
        and not families_by_bet.get(bet.id)
    ]
    if empty_active:
        raise ValidationFailure(
            f"{source_name} active big_bets need at least one idea family: {', '.join(empty_active)}"
        )


def _normalized_edges(registry: BigBetsRegistry) -> list[dict[str, str]]:
    edges: dict[tuple[str, str], str] = {}
    for bet in registry.big_bets:
        for target in bet.unlocks:
            edges[(bet.id, target)] = "unlocks"
        for source in bet.depends_on:
            edges.setdefault((source, bet.id), "depends_on")
    return [
        {"from": source, "to": target, "kind": kind}
        for (source, target), kind in sorted(edges.items())
    ]


def _families_by_bet(registry: BigBetsRegistry) -> dict[str, tuple[IdeaFamily, ...]]:
    grouped: dict[str, list[IdeaFamily]] = {}
    for family in registry.idea_families:
        grouped.setdefault(family.big_bet, []).append(family)
    return {
        key: tuple(sorted(value, key=_idea_family_sort_key))
        for key, value in grouped.items()
    }


def _big_bets_by_wave(registry: BigBetsRegistry) -> dict[int, tuple[BigBet, ...]]:
    waves: dict[int, list[BigBet]] = {}
    for bet in registry.big_bets:
        waves.setdefault(bet.wave, []).append(bet)
    return {
        wave: tuple(sorted(items, key=_big_bet_sort_key))
        for wave, items in sorted(waves.items())
    }


def _big_bet_sort_key(bet: BigBet) -> tuple[int, int, str]:
    return (bet.wave, bet.rank or 9999, bet.title.lower())


def _idea_family_sort_key(family: IdeaFamily) -> tuple[int, int, int]:
    return (_priority_rank(family.priority), family.rank or 9999, family.issue)


def _priority_rank(priority: str) -> int:
    match = _PRIORITY_RE.match(priority)
    if match is None:
        return 999
    return int(match.group(1))


def _priority_for_wave(wave: int) -> str:
    return f"P{max(0, wave - 1)}"


def _priority_color(priority: str) -> str:
    return {
        "P0": "#0f766e",
        "P1": "#2563eb",
        "P2": "#b45309",
        "P3": "#7c3aed",
    }.get(priority, "#475569")


def _priority_fill(priority: str) -> str:
    return {
        "P0": "#d8f3dc",
        "P1": "#dbeafe",
        "P2": "#fff0bf",
        "P3": "#eee6ff",
    }.get(priority, "#e9edf2")


def _svg_card(
    bet: BigBet, families: tuple[IdeaFamily, ...], x: int, y: int, color: str
) -> list[str]:
    title_lines = _wrap_text(bet.title, 33, max_lines=3)
    family_label = ", ".join(f"#{family.issue}" for family in families[:4])
    if len(families) > 4:
        family_label = f"{family_label}, +{len(families) - 4}"
    parts = [
        f'<g data-bet-id="{_attr(bet.id)}" tabindex="0" role="button" aria-label="{_attr(bet.title)}">',
        f'<path d="{_rounded_rect_path(x, y, _CARD_WIDTH, _CARD_HEIGHT, 27)}" fill="{_priority_fill(bet.priority)}" stroke="none" filter="url(#shadow)"/>',
        *_rough_rect_paths(x, y, _CARD_WIDTH, _CARD_HEIGHT, 27, _stable_seed(bet.id)),
        f'<path d="{_rough_rect_path(x + 13, y + 13, _CARD_WIDTH - 26, _CARD_HEIGHT - 26, 20, _stable_seed(f"inner:{bet.id}"))}" fill="none" stroke="{color}" stroke-width="1.05" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"/>',
        f'<text x="{x + 23}" y="{y + 32}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="12" font-weight="400" fill="{color}">{_xml(bet.priority)} / {_xml(bet.status)}</text>',
    ]
    for index, line in enumerate(title_lines):
        parts.append(
            f'<text x="{x + 23}" y="{y + 61 + index * 19}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="16" font-weight="400" fill="#1e1e1e">{_xml(line)}</text>'
        )
    parts.append(
        f'<text x="{x + 23}" y="{y + _CARD_HEIGHT - 20}" font-family="Excalifont, Virgil, Comic Sans MS, Marker Felt, sans-serif" font-size="12" fill="#4b5563">{_xml(family_label or "No mapped idea families")}</text>'
    )
    parts.append("</g>")
    return parts


def _svg_edge(
    source: str,
    target: str,
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    bend: float,
    dash: str,
) -> list[str]:
    seed = _stable_seed(f"edge:{source}:{target}")
    first = _edge_path(sx, sy, tx, ty, bend, seed)
    second = _edge_path(sx, sy, tx, ty, bend, seed + 17)
    return [
        f'<path d="{first}" fill="none" stroke="#1e1e1e" stroke-width="1.42" stroke-linecap="round" stroke-linejoin="round"{dash} marker-end="url(#arrow)"/>',
        f'<path d="{second}" fill="none" stroke="#1e1e1e" stroke-width="0.72" stroke-linecap="round" stroke-linejoin="round" opacity="0.28"{dash}/>',
    ]


def _edge_path(
    sx: float,
    sy: float,
    tx: float,
    ty: float,
    bend: float,
    seed: int,
) -> str:
    j = _jitter
    return (
        f"M {sx + j(seed, 1):.1f} {sy + j(seed, 2):.1f} "
        f"C {sx + j(seed, 3):.1f} {sy + bend + j(seed, 4):.1f}, "
        f"{tx + j(seed, 5):.1f} {ty - bend + j(seed, 6):.1f}, "
        f"{tx + j(seed, 7):.1f} {ty + j(seed, 8):.1f}"
    )


def _rough_rect_paths(
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    seed: int,
) -> list[str]:
    return [
        f'<path class="card-outline" d="{_rough_rect_path(x, y, width, height, radius, seed)}" fill="none" stroke="#1e1e1e" stroke-width="1.62" stroke-linecap="round" stroke-linejoin="round"/>',
        f'<path d="{_rough_rect_path(x + 1, y - 1, width - 2, height + 1, radius, seed + 31)}" fill="none" stroke="#1e1e1e" stroke-width="0.75" stroke-linecap="round" stroke-linejoin="round" opacity="0.34"/>',
    ]


def _rough_rect_path(
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    seed: int,
) -> str:
    points = [
        (x + radius, y),
        (x + width - radius, y),
        (x + width, y + radius),
        (x + width, y + height - radius),
        (x + width - radius, y + height),
        (x + radius, y + height),
        (x, y + height - radius),
        (x, y + radius),
    ]
    jittered = [
        (px + _jitter(seed, index * 2), py + _jitter(seed, index * 2 + 1))
        for index, (px, py) in enumerate(points)
    ]
    return (
        f"M {jittered[0][0]:.1f} {jittered[0][1]:.1f} "
        f"L {jittered[1][0]:.1f} {jittered[1][1]:.1f} "
        f"Q {x + width + _jitter(seed, 20):.1f} {y + _jitter(seed, 21):.1f} {jittered[2][0]:.1f} {jittered[2][1]:.1f} "
        f"L {jittered[3][0]:.1f} {jittered[3][1]:.1f} "
        f"Q {x + width + _jitter(seed, 22):.1f} {y + height + _jitter(seed, 23):.1f} {jittered[4][0]:.1f} {jittered[4][1]:.1f} "
        f"L {jittered[5][0]:.1f} {jittered[5][1]:.1f} "
        f"Q {x + _jitter(seed, 24):.1f} {y + height + _jitter(seed, 25):.1f} {jittered[6][0]:.1f} {jittered[6][1]:.1f} "
        f"L {jittered[7][0]:.1f} {jittered[7][1]:.1f} "
        f"Q {x + _jitter(seed, 26):.1f} {y + _jitter(seed, 27):.1f} {jittered[0][0]:.1f} {jittered[0][1]:.1f} Z"
    )


def _rounded_rect_path(x: int, y: int, width: int, height: int, radius: int) -> str:
    return (
        f"M {x + radius} {y} H {x + width - radius} "
        f"Q {x + width} {y} {x + width} {y + radius} "
        f"V {y + height - radius} Q {x + width} {y + height} {x + width - radius} {y + height} "
        f"H {x + radius} Q {x} {y + height} {x} {y + height - radius} "
        f"V {y + radius} Q {x} {y} {x + radius} {y} Z"
    )


def _jitter(seed: int, salt: int, amount: float = 0.9) -> float:
    value = (seed ^ (salt * 0x9E3779B1)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    return ((value % 1000) / 999 - 0.5) * amount * 2


def _excalidraw_id(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return f"bb_{digest[:18]}"


def _stable_seed(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def _wrap_text(text: str, width: int, *, max_lines: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return lines or [text[:width]]


def _mermaid_label(bet: BigBet, registry: BigBetsRegistry) -> str:
    families = _families_by_bet(registry).get(bet.id, ())
    issue_label = " ".join(f"#{family.issue}" for family in families)
    return _escape_mermaid(f"{bet.priority} · {bet.title}\\n{issue_label}")


def _node_id(value: str) -> str:
    return "bb_" + re.sub(r"[^A-Za-z0-9_]", "_", value)


def _markdown_issue_link(family: IdeaFamily) -> str:
    label = f"#{family.issue}"
    if family.url:
        return f"[{label}]({family.url})"
    return label


def _escape_markdown_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _escape_mermaid(value: str) -> str:
    return value.replace('"', '\\"')


def _xml(value: str) -> str:
    return html.escape(value, quote=False)


def _attr(value: str) -> str:
    return html.escape(value, quote=True)


def _required_sequence(value: object, label: str) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValidationFailure(f"Expected {label!r} to be a list.")
    sequence = cast(list[object] | tuple[object, ...], value)
    return tuple(sequence)


def _optional_sequence(value: object, label: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValidationFailure(f"Expected {label!r} to be a list.")
    sequence = cast(list[object] | tuple[object, ...], value)
    return tuple(sequence)


def _required_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValidationFailure(f"Expected {label!r} to be a mapping.")
    return cast(dict[str, object], value)


def _optional_mapping(value: object, label: str) -> dict[str, object] | None:
    if value is None:
        return None
    return _required_mapping(value, label)


def _required_string(mapping: dict[str, object], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"Expected {label}.{key} to be a non-empty string.")
    return value.strip()


def _optional_string(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationFailure(f"Expected {key!r} to be a string.")
    normalized = value.strip()
    return normalized or None


def _required_identifier(mapping: dict[str, object], key: str, label: str) -> str:
    value = _required_string(mapping, key, label)
    return _validate_identifier_value(value, f"{label}.{key}")


def _required_identifier_value(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"Expected {label} entries to be non-empty strings.")
    return _validate_identifier_value(value.strip(), label)


def _validate_identifier_value(value: str, label: str) -> str:
    if _IDENTIFIER_RE.match(value) is None:
        raise ValidationFailure(
            f"Expected {label} to match {_IDENTIFIER_RE.pattern!r}; got {value!r}."
        )
    return value


def _required_priority(mapping: dict[str, object], key: str, label: str) -> str:
    value = _required_string(mapping, key, label)
    if _PRIORITY_RE.match(value) is None:
        raise ValidationFailure(f"Expected {label}.{key} to look like P0, P1, ...")
    return value


def _required_status(mapping: dict[str, object], key: str, label: str) -> str:
    value = _required_string(mapping, key, label)
    if value not in _KNOWN_STATUSES:
        raise ValidationFailure(
            f"Expected {label}.{key} to be one of {sorted(_KNOWN_STATUSES)}."
        )
    return value


def _required_positive_int(mapping: dict[str, object], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValidationFailure(f"Expected {label}.{key} to be a positive integer.")
    return value


def _optional_positive_int(mapping: dict[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValidationFailure(f"Expected {key!r} to be a positive integer.")
    return value


__all__ = [
    "BigBet",
    "BigBetsRegistry",
    "IdeaFamily",
    "RenderedBigBets",
    "load_bigbets_registry",
    "normalize_bigbets_registry",
    "registry_to_input_payload",
    "render_artifact_metadata_json",
    "render_bigbets",
    "render_excalidraw",
    "render_html",
    "render_markdown",
    "render_mermaid",
    "render_rankings_csv",
    "render_svg",
    "validate_bigbets_registry",
    "write_bigbets_artifacts",
]
