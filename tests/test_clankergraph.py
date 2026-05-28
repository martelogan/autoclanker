from __future__ import annotations

import json

from pathlib import Path
from typing import cast

import pytest

from autoclanker.bayes_layer import ingest_human_beliefs
from autoclanker.bayes_layer.types import ValidationFailure
from autoclanker.clankergraph import (
    CLANKERGRAPH_SCHEMA_VERSION,
    belief_input_from_clankergraph,
    load_clankergraph_document,
    summarize_clankergraph_document,
    validate_clankergraph_document,
)
from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


@covers("M1-001", "M1-003")
def test_clankergraph_evidence_compiles_to_metadata_only_proposals() -> None:
    document = load_clankergraph_document(
        ROOT / "examples/clankergraph/evidence.clankergraph.json"
    )
    payload = belief_input_from_clankergraph(
        document,
        era_id="era_graph_v1",
        session_id="graph_session",
        author="test",
    )
    batch = ingest_human_beliefs(payload)

    assert payload["session_context"] == {
        "era_id": "era_graph_v1",
        "session_id": "graph_session",
        "author": "test",
    }
    assert len(batch.beliefs) == 2
    assert {belief.kind for belief in batch.beliefs} == {"proposal"}
    first_belief = cast(dict[str, object], cast(list[object], payload["beliefs"])[0])
    context = cast(dict[str, object], first_belief["context"])
    metadata = cast(dict[str, object], context["metadata"])
    graph_metadata = cast(dict[str, object], metadata["clankergraph"])
    assert graph_metadata["compiler_policy"] == "metadata_only_proposal"
    assert graph_metadata["graph_role"] == "evidence"


@covers("M1-001", "M1-003")
def test_clankergraph_context_compiles_to_codebase_patterns() -> None:
    document = load_clankergraph_document(
        ROOT / "examples/clankergraph/context.clankergraph.json"
    )
    payload = belief_input_from_clankergraph(document, era_id="era_graph_v1")
    batch = ingest_human_beliefs(payload)

    assert len(batch.beliefs) == 1
    belief = batch.beliefs[0]
    assert belief.kind == "codebase_patterns"
    belief_payload = cast(dict[str, object], cast(list[object], payload["beliefs"])[0])
    assert (
        "Prefer the existing matcher registry"
        in cast(list[str], belief_payload["preferred_patterns"])[0]
    )
    assert (
        "Avoid one-off global caches"
        in cast(list[str], belief_payload["discouraged_patterns"])[0]
    )


@covers("M1-001")
def test_clankergraph_validation_and_summary() -> None:
    document = validate_clankergraph_document(
        json.loads(
            (ROOT / "examples/clankergraph/evidence.clankergraph.json").read_text(
                encoding="utf-8"
            )
        )
    )
    summary = summarize_clankergraph_document(document)

    assert document["schema_version"] == CLANKERGRAPH_SCHEMA_VERSION
    assert summary["graph_id"] == "parser-evidence"
    assert summary["graph_role"] == "evidence"
    assert summary["node_count"] == 3
    assert cast(dict[str, object], summary["nodes_by_kind"])["action"] == 1


@covers("M1-001", "M1-003")
def test_external_investigation_style_evidence_graph_validates_and_compiles_safely() -> None:
    document = load_clankergraph_document(
        ROOT / "examples/clankergraph/investigation_evidence.clankergraph.json"
    )
    summary = summarize_clankergraph_document(document)
    assert summary["graph_role"] == "evidence"
    assert summary["node_count"] == 4
    assert cast(dict[str, object], summary["nodes_by_kind"]) == {
        "case": 1,
        "explanation": 1,
        "observation": 1,
        "x.investigation.work_log": 1,
    }

    payload = belief_input_from_clankergraph(document, era_id="era_investigation")
    batch = ingest_human_beliefs(payload)
    assert len(batch.beliefs) == 1
    belief_payload = cast(dict[str, object], cast(list[object], payload["beliefs"])[0])
    assert belief_payload["kind"] == "proposal"
    assert "Heavy template work" in str(belief_payload["proposal_text"])
    context = cast(dict[str, object], belief_payload["context"])
    metadata = cast(dict[str, object], context["metadata"])
    graph_metadata = cast(dict[str, object], metadata["clankergraph"])
    assert graph_metadata["compiler_policy"] == "metadata_only_proposal"


@covers("M1-001")
def test_clankergraph_preserves_rich_metadata_topology_and_derivations() -> None:
    document = validate_clankergraph_document(
        {
            "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
            "graph_id": "rich-graph",
            "graph_role": "context",
            "title": "Rich graph",
            "scope": {"repository": "example"},
            "produced_by": {"tool": "fixture"},
            "source_snapshot": {"revision": "abc123"},
            "metadata": {"purpose": "coverage"},
            "nodes": [
                {
                    "id": "node:pattern",
                    "kind": "pattern",
                    "title": "Prefer native adapters",
                    "body": "Keep integration points generic.",
                    "status": "x.reviewed",
                    "confidence": {"level": "high", "basis": "operator"},
                    "properties": {"subsystem": "bench"},
                    "provenance": {
                        "source": "manual",
                        "source_refs": ["docs/CLANKERBENCH.md"],
                    },
                },
                {
                    "id": "node:decision",
                    "kind": "design_decision",
                    "title": "Use graph handoff",
                },
            ],
            "edges": [
                {
                    "id": "edge:pattern-decision",
                    "from": "node:pattern",
                    "to": "node:decision",
                    "kind": "supports",
                    "strength": "strong",
                    "confidence": "high",
                    "properties": {"reviewed": True},
                    "provenance": {"source": "compiled"},
                }
            ],
            "derivations": [
                {
                    "id": "derivation:context-to-belief",
                    "from_graph_role": "context",
                    "from_node_ids": ["node:pattern"],
                    "to_graph_role": "belief",
                    "to_node_ids": ["node:decision"],
                    "transform": "human_reviewed_summary",
                    "losses": ["topology compressed"],
                    "review_required": True,
                }
            ],
        }
    )

    assert document["scope"] == {"repository": "example"}
    assert document["produced_by"] == {"tool": "fixture"}
    nodes = cast(list[dict[str, object]], document["nodes"])
    confidence = cast(dict[str, object], nodes[0]["confidence"])
    assert confidence["basis"] == "operator"
    edges = cast(list[dict[str, object]], document["edges"])
    assert edges[0]["strength"] == "strong"
    derivations = cast(list[dict[str, object]], document["derivations"])
    assert derivations[0]["review_required"] is True


@covers("M1-001")
def test_clankergraph_rejects_missing_references_and_bad_kinds() -> None:
    valid = {
        "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
        "graph_id": "graph",
        "graph_role": "evidence",
        "title": "Graph",
        "nodes": [{"id": "node:one", "kind": "observation"}],
        "edges": [],
    }

    with pytest.raises(ValidationFailure, match="schema_version"):
        validate_clankergraph_document({**valid, "schema_version": "wrong"})
    with pytest.raises(ValidationFailure, match="known value"):
        validate_clankergraph_document(
            {**valid, "nodes": [{"id": "node:one", "kind": "mystery"}]}
        )
    with pytest.raises(ValidationFailure, match="missing node"):
        validate_clankergraph_document(
            {
                **valid,
                "edges": [
                    {
                        "id": "edge:missing",
                        "from": "node:one",
                        "to": "node:two",
                        "kind": "supports",
                    }
                ],
            }
        )

    validate_clankergraph_document(
        {**valid, "nodes": [{"id": "node:one", "kind": "x.internal"}]}
    )


@covers("M1-001")
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (None, "JSON object"),
        ({"schema_version": CLANKERGRAPH_SCHEMA_VERSION}, "graph_role"),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": "node",
                "edges": [],
            },
            "nodes",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [],
                "edges": "edge",
            },
            "edges",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [{"id": "node:one", "kind": "observation"}],
                "edges": [],
                "derivations": {},
            },
            "derivations",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [
                    {"id": "node:one", "kind": "observation"},
                    {"id": "node:one", "kind": "observation"},
                ],
                "edges": [],
            },
            "duplicate id",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [{"id": "node:one", "kind": "observation"}],
                "edges": [
                    {
                        "id": "edge:one",
                        "from": "node:one",
                        "to": "node:one",
                        "kind": "supports",
                    },
                    {
                        "id": "edge:one",
                        "from": "node:one",
                        "to": "node:one",
                        "kind": "supports",
                    },
                ],
            },
            "duplicate id",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [{"id": "node:one", "kind": "observation"}],
                "edges": [],
                "derivations": [
                    {
                        "id": "derivation:missing",
                        "from_graph_role": "context",
                        "from_node_ids": ["node:missing"],
                        "to_graph_role": "belief",
                        "to_node_ids": ["node:one"],
                        "transform": "summary",
                    }
                ],
            },
            "missing node",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [
                    {
                        "id": "node:one",
                        "kind": "observation",
                        "confidence": {"level": "certain"},
                    }
                ],
                "edges": [],
            },
            "confidence.level",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [
                    {
                        "id": "node:one",
                        "kind": "observation",
                        "provenance": {"source": "rumor"},
                    }
                ],
                "edges": [],
            },
            "provenance.source",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [{"id": "node:one", "kind": "observation"}],
                "edges": [
                    {
                        "id": "edge:one",
                        "from": "node:one",
                        "to": "node:one",
                        "kind": "supports",
                        "strength": "loud",
                    }
                ],
            },
            "strength",
        ),
        (
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "graph",
                "graph_role": "evidence",
                "title": "Graph",
                "nodes": [{"id": "node:one", "kind": "observation"}],
                "edges": [],
                "metadata": [],
            },
            "metadata",
        ),
    ],
)
def test_clankergraph_rejects_malformed_shapes(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValidationFailure, match=message):
        validate_clankergraph_document(payload)


@covers("M1-003")
def test_clankergraph_role_specific_belief_compilation_policies() -> None:
    benchmark_payload = belief_input_from_clankergraph(
        validate_clankergraph_document(
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "benchmark-graph",
                "graph_role": "benchmark",
                "title": "Benchmark graph",
                "nodes": [
                    {
                        "id": "compare:win",
                        "kind": "compare",
                        "title": "Candidate beats baseline",
                        "status": "validated",
                    }
                ],
                "edges": [],
            }
        ),
        era_id="era_graph_v1",
    )
    benchmark_belief = cast(
        dict[str, object],
        cast(list[object], benchmark_payload["beliefs"])[0],
    )
    assert benchmark_belief["confidence_level"] == 3
    assert benchmark_belief["evidence_sources"] == ["benchmark"]
    assert str(benchmark_belief["proposal_text"]).startswith("Account for benchmark")

    belief_payload = belief_input_from_clankergraph(
        validate_clankergraph_document(
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "belief-graph",
                "graph_role": "belief",
                "title": "Belief graph",
                "nodes": [
                    {
                        "id": "query:pair",
                        "kind": "query",
                        "title": "Inspect paired move",
                        "confidence": {"level": "high"},
                    }
                ],
                "edges": [],
            }
        ),
        era_id="era_graph_v1",
    )
    belief_belief = cast(
        dict[str, object], cast(list[object], belief_payload["beliefs"])[0]
    )
    assert belief_belief["confidence_level"] == 2
    assert belief_belief["suggested_scope"] == "gene"
    assert "Review graph belief" in str(belief_belief["proposal_text"])

    fallback_payload = belief_input_from_clankergraph(
        validate_clankergraph_document(
            {
                "schema_version": CLANKERGRAPH_SCHEMA_VERSION,
                "graph_id": "context-without-actionable-nodes",
                "graph_role": "context",
                "title": "Context graph",
                "nodes": [{"id": "section:overview", "kind": "section"}],
                "edges": [],
            }
        ),
        era_id="era_graph_v1",
    )
    fallback_belief = cast(
        dict[str, object],
        cast(list[object], fallback_payload["beliefs"])[0],
    )
    fallback_context = cast(dict[str, object], fallback_belief["context"])
    fallback_metadata = cast(dict[str, object], fallback_context["metadata"])
    fallback_graph = cast(dict[str, object], fallback_metadata["clankergraph"])
    assert fallback_graph["compiler_policy"] == (
        "metadata_only_without_exact_intervention_mapping"
    )


@covers("M1-001")
def test_clankergraph_loader_reports_invalid_json(tmp_path: Path) -> None:
    graph_path = tmp_path / "bad.clankergraph.json"
    graph_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValidationFailure, match="Failed to parse"):
        load_clankergraph_document(graph_path)


@covers("M4-004")
def test_cli_graph_validate_and_beliefs_from_graph(
    capsys: pytest.CaptureFixture[str],
) -> None:
    graph_path = ROOT / "examples/clankergraph/evidence.clankergraph.json"

    assert main(["graph", "validate", "--input", str(graph_path)]) == 0
    validate_payload = json.loads(capsys.readouterr().out)
    assert validate_payload["ok"] is True
    assert validate_payload["summary"]["graph_role"] == "evidence"

    assert main(["graph", "summarize", "--input", str(graph_path)]) == 0
    summary_payload = json.loads(capsys.readouterr().out)
    assert summary_payload["graph_id"] == "parser-evidence"

    assert (
        main(
            [
                "beliefs",
                "from-graph",
                "--input",
                str(graph_path),
                "--era-id",
                "era_graph_v1",
            ]
        )
        == 0
    )
    beliefs_payload = json.loads(capsys.readouterr().out)
    assert beliefs_payload["session_context"]["era_id"] == "era_graph_v1"
    assert beliefs_payload["beliefs"][0]["kind"] == "proposal"
