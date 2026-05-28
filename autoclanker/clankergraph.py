from __future__ import annotations

import json
import re

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer.types import JsonValue, ValidationFailure

CLANKERGRAPH_SCHEMA_VERSION = "clankergraph.v1"
GRAPH_ROLES = ("evidence", "belief", "benchmark", "context")
CONFIDENCE_LEVELS = ("low", "medium", "high")
CONFIDENCE_BASES = ("measured", "inferred", "operator", "llm", "mixed")
STATUSES = (
    "proposed",
    "inconclusive",
    "likely",
    "confirmed",
    "ruled_out",
    "superseded",
    "blocked",
    "implemented",
    "validated",
    "rejected",
)
NODE_KINDS = (
    "case",
    "inquiry",
    "artifact",
    "observation",
    "explanation",
    "subclaim",
    "caveat",
    "action",
    "open_question",
    "source_location",
    "prior_work",
    "belief",
    "gene",
    "state",
    "candidate",
    "frontier",
    "relation",
    "constraint",
    "preference",
    "risk",
    "query",
    "corpus",
    "cohort",
    "stratum",
    "recording",
    "metric",
    "threshold",
    "eval_run",
    "compare",
    "spec",
    "environment",
    "promotion_gate",
    "design_decision",
    "code_symbol",
    "file",
    "section",
    "test_spec",
    "issue",
    "pull_request",
    "owner",
    "pattern",
    "anti_pattern",
    "project",
)
EDGE_KINDS = (
    "produced",
    "observed_in",
    "supports",
    "contradicts",
    "contextualizes",
    "decomposes",
    "supersedes",
    "recommends",
    "validates",
    "blocks",
    "related_to",
    "compiled_from",
    "influences",
    "depends_on",
    "excludes",
    "synergizes",
    "conflicts",
    "prefers",
    "asks_about",
    "selected_for_eval",
    "weakened_by",
    "strengthened_by",
    "locks",
    "covers",
    "materializes",
    "measures",
    "compares_to",
    "passes",
    "fails",
    "keeps",
    "rejects",
    "regresses",
    "invalidates",
    "implements",
    "tests",
    "owned_by",
)
STRENGTHS = ("weak", "medium", "strong")
PROVENANCE_SOURCES = ("measured", "manual", "compiled", "inferred", "extracted")

_EXTENSION_PATTERN = re.compile(r"^x\.[A-Za-z0-9_.-]+$")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _required_string(mapping: Mapping[str, object], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValidationFailure(f"{label}.{key} must be a non-empty string.")
    return value


def _optional_string(mapping: Mapping[str, object], key: str, label: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValidationFailure(f"{label}.{key} must be a non-empty string.")
    return value


def _optional_bool(mapping: Mapping[str, object], key: str, label: str) -> bool | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValidationFailure(f"{label}.{key} must be a boolean.")
    return value


def _optional_mapping(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> dict[str, JsonValue] | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label}.{key} must be a JSON object.")
    return cast(dict[str, JsonValue], dict(cast(Mapping[str, object], value)))


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValidationFailure(f"{label} must be a list.")
    items = cast(list[object], value)
    result: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise ValidationFailure(f"{label}[{index}] must be a non-empty string.")
        result.append(item)
    return result


def _optional_string_list(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> list[str] | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _string_list(value, f"{label}.{key}")


def _enum_or_extension(value: object, allowed: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationFailure(f"{label} must be a non-empty string.")
    if value in allowed or _EXTENSION_PATTERN.match(value):
        return value
    raise ValidationFailure(f"{label} must be a known value or x.* extension.")


def _enum(value: object, allowed: Sequence[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValidationFailure(f"{label} must be one of {', '.join(allowed)}.")
    return value


def _confidence(value: object, label: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, label)
    level = _enum(mapping.get("level"), CONFIDENCE_LEVELS, f"{label}.level")
    basis = mapping.get("basis")
    if basis is not None:
        _enum(basis, CONFIDENCE_BASES, f"{label}.basis")
    payload: dict[str, JsonValue] = {"level": level}
    if isinstance(basis, str):
        payload["basis"] = basis
    return payload


def _provenance(value: object, label: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, label)
    payload: dict[str, object] = {}
    source = mapping.get("source")
    if source is not None:
        payload["source"] = _enum(source, PROVENANCE_SOURCES, f"{label}.source")
    source_refs = _optional_string_list(mapping, "source_refs", label)
    if source_refs is not None:
        payload["source_refs"] = source_refs
    return cast(dict[str, JsonValue], payload)


def _node(value: object, label: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, label)
    payload: dict[str, JsonValue] = {
        "id": _required_string(mapping, "id", label),
        "kind": _enum_or_extension(mapping.get("kind"), NODE_KINDS, f"{label}.kind"),
    }
    for key in ("title", "body"):
        item = _optional_string(mapping, key, label)
        if item is not None:
            payload[key] = item
    status = mapping.get("status")
    if status is not None:
        payload["status"] = _enum_or_extension(status, STATUSES, f"{label}.status")
    confidence = mapping.get("confidence")
    if confidence is not None:
        payload["confidence"] = _confidence(confidence, f"{label}.confidence")
    properties = _optional_mapping(mapping, "properties", label)
    if properties is not None:
        payload["properties"] = properties
    provenance = mapping.get("provenance")
    if provenance is not None:
        payload["provenance"] = _provenance(provenance, f"{label}.provenance")
    return payload


def _edge(value: object, label: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, label)
    payload: dict[str, JsonValue] = {
        "id": _required_string(mapping, "id", label),
        "from": _required_string(mapping, "from", label),
        "to": _required_string(mapping, "to", label),
        "kind": _enum_or_extension(mapping.get("kind"), EDGE_KINDS, f"{label}.kind"),
    }
    strength = mapping.get("strength")
    if strength is not None:
        payload["strength"] = _enum(strength, STRENGTHS, f"{label}.strength")
    confidence = mapping.get("confidence")
    if confidence is not None:
        payload["confidence"] = _enum(
            confidence, CONFIDENCE_LEVELS, f"{label}.confidence"
        )
    properties = _optional_mapping(mapping, "properties", label)
    if properties is not None:
        payload["properties"] = properties
    provenance = mapping.get("provenance")
    if provenance is not None:
        payload["provenance"] = _provenance(provenance, f"{label}.provenance")
    return payload


def _derivation(value: object, label: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, label)
    payload: dict[str, object] = {
        "id": _required_string(mapping, "id", label),
        "from_graph_role": _enum(
            mapping.get("from_graph_role"), GRAPH_ROLES, f"{label}.from_graph_role"
        ),
        "from_node_ids": _string_list(
            mapping.get("from_node_ids"), f"{label}.from_node_ids"
        ),
        "to_graph_role": _enum(
            mapping.get("to_graph_role"), GRAPH_ROLES, f"{label}.to_graph_role"
        ),
        "to_node_ids": _string_list(mapping.get("to_node_ids"), f"{label}.to_node_ids"),
        "transform": _required_string(mapping, "transform", label),
    }
    review_required = _optional_bool(mapping, "review_required", label)
    if review_required is not None:
        payload["review_required"] = review_required
    losses = _optional_string_list(mapping, "losses", label)
    if losses is not None:
        payload["losses"] = losses
    return cast(dict[str, JsonValue], payload)


def validate_clankergraph_document(value: object) -> dict[str, JsonValue]:
    mapping = _mapping(value, "clankergraph")
    schema_version = mapping.get("schema_version")
    if schema_version != CLANKERGRAPH_SCHEMA_VERSION:
        raise ValidationFailure(
            f"clankergraph.schema_version must be {CLANKERGRAPH_SCHEMA_VERSION}."
        )
    graph_role = _enum(
        mapping.get("graph_role"), GRAPH_ROLES, "clankergraph.graph_role"
    )

    raw_nodes = mapping.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValidationFailure("clankergraph.nodes must be a list.")
    raw_node_items = cast(list[object], raw_nodes)
    nodes = [
        _node(item, f"nodes[{index}]") for index, item in enumerate(raw_node_items)
    ]

    raw_edges = mapping.get("edges")
    if not isinstance(raw_edges, list):
        raise ValidationFailure("clankergraph.edges must be a list.")
    raw_edge_items = cast(list[object], raw_edges)
    edges = [
        _edge(item, f"edges[{index}]") for index, item in enumerate(raw_edge_items)
    ]

    raw_derivations = mapping.get("derivations", [])
    if not isinstance(raw_derivations, list):
        raise ValidationFailure("clankergraph.derivations must be a list when present.")
    raw_derivation_items = cast(list[object], raw_derivations)
    derivations = [
        _derivation(item, f"derivations[{index}]")
        for index, item in enumerate(raw_derivation_items)
    ]

    node_ids: set[str] = set()
    for node in nodes:
        node_id = cast(str, node["id"])
        if node_id in node_ids:
            raise ValidationFailure(f"nodes contains duplicate id {node_id!r}.")
        node_ids.add(node_id)

    edge_ids: set[str] = set()
    for edge in edges:
        edge_id = cast(str, edge["id"])
        if edge_id in edge_ids:
            raise ValidationFailure(f"edges contains duplicate id {edge_id!r}.")
        edge_ids.add(edge_id)
        for endpoint in (cast(str, edge["from"]), cast(str, edge["to"])):
            if endpoint not in node_ids:
                raise ValidationFailure(
                    f"edge {edge_id!r} references missing node {endpoint!r}."
                )

    for derivation in derivations:
        derivation_id = cast(str, derivation["id"])
        for node_id in cast(list[str], derivation["from_node_ids"]) + cast(
            list[str], derivation["to_node_ids"]
        ):
            if node_id not in node_ids:
                raise ValidationFailure(
                    f"derivation {derivation_id!r} references missing node {node_id!r}."
                )

    payload: dict[str, object] = {
        "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
        "graph_id": _required_string(mapping, "graph_id", "clankergraph"),
        "graph_role": graph_role,
        "title": _required_string(mapping, "title", "clankergraph"),
        "nodes": nodes,
        "edges": edges,
        "derivations": derivations,
    }
    for key in ("scope", "produced_by", "source_snapshot", "metadata"):
        item = _optional_mapping(mapping, key, "clankergraph")
        if item is not None:
            payload[key] = item
    return cast(dict[str, JsonValue], payload)


def load_clankergraph_document(path: Path) -> dict[str, JsonValue]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"Failed to parse clankergraph JSON: {exc}") from exc
    return validate_clankergraph_document(payload)


def summarize_clankergraph_document(
    document: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    nodes = cast(list[Mapping[str, JsonValue]], document["nodes"])
    nodes_by_kind: dict[str, int] = {}
    for node in nodes:
        kind = cast(str, node["kind"])
        nodes_by_kind[kind] = nodes_by_kind.get(kind, 0) + 1
    return cast(
        dict[str, JsonValue],
        {
            "schema_version": document["schema_version"],
            "graph_id": document["graph_id"],
            "graph_role": document["graph_role"],
            "node_count": len(nodes),
            "edge_count": len(cast(list[JsonValue], document["edges"])),
            "derivation_count": len(cast(list[JsonValue], document["derivations"])),
            "nodes_by_kind": nodes_by_kind,
        },
    )


def belief_input_from_clankergraph(
    document: Mapping[str, JsonValue],
    *,
    era_id: str,
    session_id: str | None = None,
    author: str | None = None,
) -> dict[str, JsonValue]:
    graph_role = cast(str, document["graph_role"])
    graph_id = cast(str, document["graph_id"])
    nodes = cast(list[Mapping[str, JsonValue]], document["nodes"])
    beliefs: list[dict[str, JsonValue]] = []

    if graph_role == "context":
        context_belief = _context_patterns_belief(graph_id, nodes)
        if context_belief is not None:
            beliefs.append(context_belief)

    for index, node in enumerate(_proposal_nodes(graph_role, nodes), start=1):
        beliefs.append(_proposal_belief_from_node(graph_id, graph_role, node, index))

    if not beliefs:
        beliefs.append(
            cast(
                dict[str, JsonValue],
                {
                    "id": f"graph_{_safe_id(graph_id)}_available",
                    "kind": "proposal",
                    "confidence_level": 1,
                    "evidence_sources": ["other"],
                    "proposal_text": (
                        f"Read clankergraph {graph_id!r} before candidate work. "
                        "It contains structured context but no node that should directly steer optimization."
                    ),
                    "suggested_scope": "other",
                    "rationale": "Graph import stayed metadata-only because no safe intervention mapping was present.",
                    "context": {
                        "metadata": {
                            "clankergraph": {
                                "graph_id": graph_id,
                                "graph_role": graph_role,
                                "compiler_policy": "metadata_only_without_exact_intervention_mapping",
                            }
                        },
                        "tags": ["clankergraph", f"clankergraph:{graph_role}"],
                    },
                },
            )
        )

    session_context: dict[str, object] = {"era_id": era_id}
    if session_id is not None:
        session_context["session_id"] = session_id
    if author is not None:
        session_context["author"] = author
    return cast(
        dict[str, JsonValue],
        {
            "session_context": session_context,
            "beliefs": beliefs,
        },
    )


def _context_patterns_belief(
    graph_id: str,
    nodes: Sequence[Mapping[str, JsonValue]],
) -> dict[str, JsonValue] | None:
    preferred = [
        _node_text(node)
        for node in nodes
        if node.get("kind") in {"pattern", "design_decision"}
    ]
    discouraged = [
        _node_text(node) for node in nodes if node.get("kind") == "anti_pattern"
    ]
    preferred = [item for item in preferred if item]
    discouraged = [item for item in discouraged if item]
    if not preferred and not discouraged:
        return None
    payload: dict[str, object] = {
        "id": f"graph_{_safe_id(graph_id)}_codebase_patterns",
        "kind": "codebase_patterns",
        "confidence_level": 2,
        "evidence_sources": ["code_inspection"],
        "rationale": "Context graph imported as codebase-fit guidance, not performance proof.",
        "context": {
            "metadata": {
                "clankergraph": {
                    "graph_id": graph_id,
                    "graph_role": "context",
                    "compiler_policy": "context_patterns_only",
                    "losses": [
                        "graph topology flattened into review guidance",
                        "codebase context does not become benchmark evidence",
                    ],
                }
            },
            "tags": ["clankergraph", "clankergraph:context"],
        },
    }
    if preferred:
        payload["preferred_patterns"] = preferred
    if discouraged:
        payload["discouraged_patterns"] = discouraged
    return cast(dict[str, JsonValue], payload)


def _proposal_nodes(
    graph_role: str,
    nodes: Sequence[Mapping[str, JsonValue]],
) -> list[Mapping[str, JsonValue]]:
    actionable_by_role = {
        "evidence": {"action", "explanation", "open_question"},
        "belief": {"belief", "query", "constraint", "risk"},
        "benchmark": {"compare", "eval_run", "candidate", "promotion_gate"},
        "context": {"prior_work", "issue", "pull_request", "project"},
    }
    allowed = actionable_by_role.get(graph_role, set())
    return [node for node in nodes if cast(str, node.get("kind")) in allowed]


def _proposal_belief_from_node(
    graph_id: str,
    graph_role: str,
    node: Mapping[str, JsonValue],
    index: int,
) -> dict[str, JsonValue]:
    node_id = cast(str, node["id"])
    status = cast(str | None, node.get("status"))
    confidence = cast(Mapping[str, JsonValue] | None, node.get("confidence"))
    level = _belief_confidence(status, confidence)
    kind = cast(str, node["kind"])
    return cast(
        dict[str, JsonValue],
        {
            "id": f"graph_{_safe_id(graph_id)}_{index:03d}",
            "kind": "proposal",
            "confidence_level": level,
            "evidence_sources": [_evidence_source_for_role(graph_role)],
            "proposal_text": _proposal_text(graph_role, kind, node),
            "suggested_scope": _suggested_scope_for_node(graph_role, kind),
            "rationale": (
                "Imported from Clankergraph without registry-backed intervention mapping; "
                "safe compiler policy keeps this as a proposal instead of a direct optimizer prior."
            ),
            "context": {
                "metadata": {
                    "clankergraph": {
                        "graph_id": graph_id,
                        "graph_role": graph_role,
                        "node_id": node_id,
                        "node_kind": kind,
                        "node_status": status,
                        "compiler_policy": "metadata_only_proposal",
                        "losses": [
                            "graph claim not mapped to a concrete registry gene/state",
                            "topology and provenance remain in the source graph",
                        ],
                    }
                },
                "tags": ["clankergraph", f"clankergraph:{graph_role}", f"node:{kind}"],
            },
        },
    )


def _belief_confidence(
    status: str | None,
    confidence: Mapping[str, JsonValue] | None,
) -> int:
    if status in {"confirmed", "validated", "implemented"}:
        return 3
    if status in {"likely", "proposed"}:
        return 2
    if confidence is not None and confidence.get("level") == "high":
        return 2
    return 1


def _evidence_source_for_role(graph_role: str) -> str:
    if graph_role == "benchmark":
        return "benchmark"
    if graph_role == "context":
        return "code_inspection"
    return "other"


def _suggested_scope_for_node(graph_role: str, kind: str) -> str:
    if graph_role == "evidence" and kind == "action":
        return "patch"
    if graph_role == "belief":
        return "gene"
    if graph_role == "context":
        return "other"
    return "other"


def _proposal_text(
    graph_role: str,
    kind: str,
    node: Mapping[str, JsonValue],
) -> str:
    text = _node_text(node)
    if graph_role == "benchmark":
        return f"Account for benchmark {kind}: {text}"
    if graph_role == "belief":
        return f"Review graph belief {kind} before applying optimizer influence: {text}"
    if graph_role == "context":
        return f"Coordinate against context {kind}: {text}"
    return text


def _node_text(node: Mapping[str, JsonValue]) -> str:
    title = node.get("title")
    body = node.get("body")
    if isinstance(title, str) and isinstance(body, str):
        return f"{title}: {body}"
    if isinstance(body, str):
        return body
    if isinstance(title, str):
        return title
    return cast(str, node["id"])


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return normalized or "graph"
