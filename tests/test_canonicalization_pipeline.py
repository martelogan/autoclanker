from __future__ import annotations

import io
import json

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationRequest,
    CanonicalizationSuggestion,
    canonicalize_belief_input,
)
from autoclanker.bayes_layer.registry import build_fixture_registry
from autoclanker.bayes_layer.types import GeneStateRef, SessionContext
from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError("Expected a JSON object.")
    return cast(dict[str, object], value)


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(argv)
    if exit_code != 0:
        raise AssertionError(
            f"autoclanker {' '.join(argv)!r} failed: {stderr.getvalue().strip()}"
        )
    return _require_mapping(json.loads(stdout.getvalue()))


def _assert_png(path: Path) -> None:
    raw = path.read_bytes()
    assert raw.startswith(PNG_MAGIC)


@covers("M4-005")
def test_adapter_surface_exposes_strategy_and_risk_families() -> None:
    payload = _run_cli(["adapter", "surface"])
    surface_summary = _require_mapping(payload["surface_summary"])
    surface_kind_counts = _require_mapping(surface_summary["surface_kind_counts"])
    semantic_level_counts = _require_mapping(surface_summary["semantic_level_counts"])
    surface = _require_mapping(payload["surface"])

    assert surface_kind_counts["runtime_option"] == 5
    assert surface_kind_counts["search_angle"] == 1
    assert surface_kind_counts["risk_family"] == 1
    assert semantic_level_counts["concrete"] == 5
    assert semantic_level_counts["strategy"] == 1
    assert semantic_level_counts["risk"] == 1

    cluster_family = _require_mapping(surface["search.incident_cluster_pass"])
    assert cluster_family["surface_kind"] == "search_angle"
    assert cluster_family["semantic_level"] == "strategy"
    assert cluster_family["materializable"] is False

    memory_risk = _require_mapping(surface["risk.capture_memory_pressure"])
    assert memory_risk["surface_kind"] == "risk_family"
    assert memory_risk["semantic_level"] == "risk"
    assert memory_risk["materializable"] is False


@covers("M1-005")
def test_hybrid_canonicalization_emits_overlay_and_provenance() -> None:
    payload = _run_cli(
        [
            "beliefs",
            "canonicalize-ideas",
            "--ideas-json",
            '["A repeated-format fast path probably helps this parser."]',
            "--era-id",
            "era_demo_v1",
            "--canonicalization-model",
            "stub",
        ]
    )

    beliefs = cast(list[object], payload["beliefs"])
    first_belief = _require_mapping(beliefs[0])
    summary = _require_mapping(payload["canonicalization_summary"])
    records = cast(list[object], summary["records"])
    first_record = _require_mapping(records[0])
    surface_overlay = _require_mapping(payload["surface_overlay"])
    overlay_registry = _require_mapping(surface_overlay["registry"])

    assert summary["mode"] == "hybrid"
    assert summary["model_name"] == "stub"
    assert first_belief["kind"] == "idea"
    assert _require_mapping(first_belief["gene"]) == {
        "gene_id": "search.repeated_format_fast_path",
        "state_id": "path_compiled_context",
    }
    assert first_record["status"] == "resolved"
    assert first_record["source"] == "hybrid"
    assert overlay_registry["search.repeated_format_fast_path"]


@covers("M1-005")
def test_precanonicalized_beliefs_preserve_surface_overlay() -> None:
    canonicalized = _run_cli(
        [
            "beliefs",
            "canonicalize-ideas",
            "--ideas-json",
            '["A repeated-format fast path probably helps this parser."]',
            "--era-id",
            "era_demo_v1",
            "--canonicalization-model",
            "stub",
        ]
    )
    surface_overlay = _require_mapping(canonicalized["surface_overlay"])

    outcome = canonicalize_belief_input(
        {
            "session_context": canonicalized["session_context"],
            "beliefs": canonicalized["beliefs"],
            "surface_overlay": surface_overlay,
        },
        registry=build_fixture_registry(),
    )

    assert outcome.surface_overlay_payload is not None
    assert outcome.registry.has_ref(
        GeneStateRef(
            gene_id="search.repeated_format_fast_path",
            state_id="path_compiled_context",
        )
    )


@covers("M1-005")
def test_beginner_ideas_can_canonicalize_against_input_surface_overlay() -> None:
    outcome = canonicalize_belief_input(
        {
            "ideas": [
                {
                    "id": "domain_aot",
                    "idea": "Move money formatting to the request boundary.",
                    "confidence": 3,
                    "option": "domain.settings=aot_money_format",
                }
            ],
            "surface_overlay": {
                "registry": {
                    "domain.settings": {
                        "states": ["baseline", "aot_money_format"],
                        "default_state": "baseline",
                        "description": "Domain-local settings optimization lane.",
                        "state_descriptions": {
                            "baseline": "Keep current runtime behavior.",
                            "aot_money_format": "Precompute money format settings at the request boundary.",
                        },
                        "surface_kind": "mutation_family",
                        "semantic_level": "strategy",
                        "materializable": False,
                        "origin": "idea_file",
                    }
                }
            },
        },
        registry=build_fixture_registry(),
        fallback_session_context=SessionContext(era_id="era_domain_v1"),
    )

    assert outcome.surface_overlay_payload is not None
    assert outcome.registry.has_ref(
        GeneStateRef(
            gene_id="domain.settings",
            state_id="aot_money_format",
        )
    )
    belief = outcome.beliefs.beliefs[0]
    assert belief.kind == "idea"
    assert _require_mapping(outcome.beliefs.canonical_payload)["beliefs"]


@covers("M1-005")
def test_hybrid_canonicalization_keeps_unknown_ideas_as_proposals() -> None:
    payload = _run_cli(
        [
            "beliefs",
            "canonicalize-ideas",
            "--ideas-json",
            '["Try a moonbeam dragon refactor with kaleidoscope anchors."]',
            "--era-id",
            "era_demo_v1",
            "--canonicalization-model",
            "stub",
        ]
    )

    beliefs = cast(list[object], payload["beliefs"])
    first_belief = _require_mapping(beliefs[0])
    summary = _require_mapping(payload["canonicalization_summary"])
    records = cast(list[object], summary["records"])
    first_record = _require_mapping(records[0])

    assert first_belief["kind"] == "proposal"
    assert first_record["status"] == "needs_review"
    assert "surface_overlay" not in payload


@covers("M1-005")
def test_llm_mode_is_model_first_with_explicit_deterministic_fallback() -> None:
    class _FallbackModel:
        name = "fallback-model"

        def __init__(self) -> None:
            self.seen_ids: tuple[str, ...] = ()

        def canonicalize(
            self, request: CanonicalizationRequest
        ) -> tuple[CanonicalizationSuggestion, ...]:
            self.seen_ids = tuple(item.belief_id for item in request.ideas)
            return tuple(
                CanonicalizationSuggestion(
                    belief_id=item.belief_id,
                    belief=None,
                    source="llm",
                    summary="Model declined to emit a typed replacement.",
                    needs_review=not item.rationale.startswith("Compiled regex"),
                )
                for item in request.ideas
            )

    model = _FallbackModel()
    outcome = canonicalize_belief_input(
        {
            "ideas": [
                {
                    "idea": "Compiled regex matching probably helps this parser on repeated log formats.",
                    "confidence": 2,
                }
            ]
        },
        fallback_session_context=SessionContext(era_id="era_demo_v1"),
        registry=build_fixture_registry(),
        mode="llm",
        model=model,
    )

    assert model.seen_ids == ("idea_001",)
    assert outcome.summary is not None
    assert outcome.summary.mode == "llm"
    assert outcome.summary.records[0].source == "deterministic"
    assert (
        outcome.summary.records[0].summary
        == "Model returned no typed replacement; kept deterministic resolution."
    )
    assert outcome.beliefs.beliefs[0].kind == "idea"


@covers("M3-005", "M3-014")
def test_hybrid_session_persists_surface_and_influence_artifacts(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )

    init_output = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "hybrid_parser",
            "--era-id",
            "era_log_parser_v1",
            "--session-root",
            str(session_root),
            "--ideas-json",
            '["A repeated-format fast path probably helps this parser."]',
            "--canonicalization-model",
            "stub",
        ]
    )
    preview_digest = str(init_output["preview_digest"])
    session_path = Path(str(init_output["session_path"]))

    assert init_output["canonicalization_mode"] == "hybrid"
    assert init_output["surface_overlay_active"] is True
    assert _require_mapping(init_output["canonicalization_summary"])["mode"] == "hybrid"
    assert (session_path / "surface_snapshot.json").exists()
    assert (session_path / "surface_overlay.json").exists()
    assert (session_path / "canonicalization_summary.json").exists()

    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            "hybrid_parser",
            "--preview-digest",
            preview_digest,
            "--session-root",
            str(session_root),
        ]
    )

    suggest_output = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "hybrid_parser",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )
    ranked_candidates = [
        _require_mapping(item)
        for item in cast(list[object], suggest_output["ranked_candidates"])
    ]
    report_artifacts = _require_mapping(suggest_output["report_artifacts"])
    influence_summary = [
        _require_mapping(item)
        for item in cast(list[object], suggest_output["influence_summary"])
    ]
    top_influence = next(
        item
        for item in influence_summary
        if item["candidate_id"] == "cand_c_compiled_context_pair"
    )

    assert ranked_candidates[0]["candidate_id"] == "cand_c_compiled_context_pair"
    assert Path(str(report_artifacts["results_markdown"])).exists()
    assert Path(str(report_artifacts["candidate_rankings_plot"])).exists()
    assert any(
        "main:search.repeated_format_fast_path=path_compiled_context" in str(entry)
        for entry in cast(list[object], top_influence["influence_summary"])
    )

    commit_output = _run_cli(
        [
            "session",
            "recommend-commit",
            "--session-id",
            "hybrid_parser",
            "--session-root",
            str(session_root),
        ]
    )
    commit_report_artifacts = _require_mapping(commit_output["report_artifacts"])
    results_path = Path(str(commit_report_artifacts["results_markdown"]))
    results_text = results_path.read_text(encoding="utf-8")
    assert "Session Results" in results_text
    assert "Follow-up queries" in results_text
    assert "Belief changes" in results_text
    assert "Proposal summary" in results_text
    assert "Commit recommendation" in results_text
    _assert_png(Path(str(commit_report_artifacts["convergence_plot"])))
    _assert_png(Path(str(commit_report_artifacts["candidate_rankings_plot"])))
    _assert_png(Path(str(commit_report_artifacts["prior_graph_plot"])))
    _assert_png(Path(str(commit_report_artifacts["posterior_graph_plot"])))
    delta_artifact = _require_mapping(
        json.loads(
            (session_path / "belief_delta_summary.json").read_text(encoding="utf-8")
        )
    )
    assert delta_artifact["era_id"] == "era_log_parser_v1"
    proposal_artifact = _require_mapping(
        json.loads((session_path / "proposal_ledger.json").read_text(encoding="utf-8"))
    )
    assert proposal_artifact["current_proposal_id"] is not None
    influence_artifact = _require_mapping(
        json.loads(
            (session_path / "influence_summary.json").read_text(encoding="utf-8")
        )
    )
    persisted_influences = cast(list[object], influence_artifact["influence_summary"])
    assert any(
        _require_mapping(item)["target_ref"]
        == "main:search.repeated_format_fast_path=path_compiled_context"
        for item in persisted_influences
    )


@covers("M3-005")
def test_session_render_report_refreshes_visual_artifacts(tmp_path: Path) -> None:
    session_root = tmp_path / "sessions"
    session_id = "render_report_demo"
    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            session_id,
            "--era-id",
            "era_render_v1",
            "--session-root",
            str(session_root),
        ]
    )

    render_output = _run_cli(
        [
            "session",
            "render-report",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
        ]
    )
    report_artifacts = _require_mapping(render_output["report_artifacts"])
    results_text = Path(str(report_artifacts["results_markdown"])).read_text(
        encoding="utf-8"
    )

    assert "Session Results" in results_text
    assert "Observation count: `0`" in results_text
    _assert_png(Path(str(report_artifacts["convergence_plot"])))
    _assert_png(Path(str(report_artifacts["candidate_rankings_plot"])))
    _assert_png(Path(str(report_artifacts["prior_graph_plot"])))
    _assert_png(Path(str(report_artifacts["posterior_graph_plot"])))


@covers("M3-015")
def test_hybrid_session_review_bundle_derives_four_briefs_without_extra_artifacts(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    session_id = "review_bundle_demo"
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )

    init_output = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            session_id,
            "--era-id",
            "era_review_bundle_v1",
            "--session-root",
            str(session_root),
            "--ideas-json",
            '["A repeated-format fast path probably helps this parser."]',
            "--canonicalization-model",
            "stub",
        ]
    )
    preview_digest = str(init_output["preview_digest"])
    session_path = Path(str(init_output["session_path"]))

    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            session_id,
            "--preview-digest",
            preview_digest,
            "--session-root",
            str(session_root),
        ]
    )
    _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )

    review_bundle = _run_cli(
        [
            "session",
            "review-bundle",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
            "--format",
            "json",
        ]
    )
    assert sorted(
        {
            "session",
            "prior_brief",
            "run_brief",
            "posterior_brief",
            "proposal_brief",
            "lanes",
            "proposals",
            "lineage",
            "trust",
            "evidence",
            "next_action",
        }
    ) == sorted(review_bundle.keys())
    assert _require_mapping(review_bundle["session"])["session_id"] == session_id
    assert _require_mapping(review_bundle["run_brief"])["summary"]
    assert cast(list[object], review_bundle["lanes"])
    assert cast(list[object], review_bundle["proposals"])

    results_text = (session_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Prior Brief" in results_text
    assert "## Run Brief" in results_text
    assert "## Posterior Brief" in results_text
    assert "## Proposal Brief" in results_text
    assert "## Lane Decisions" in results_text
    assert "## Proposal Recommendations" in results_text
    assert "## How to read the evidence views" in results_text
    assert not (session_path / "review_bundle.json").exists()


@covers("M3-015")
def test_review_bundle_infers_lane_and_proposal_lineage_from_candidate_genotype(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    session_id = "review_lineage_demo"
    frontier_path = tmp_path / "frontier.json"
    frontier_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "candidate_id": "cand_lineage",
                        "family_id": "family_lineage",
                        "origin_kind": "seed",
                        "genotype": [
                            {
                                "gene_id": "search.repeated_format_fast_path",
                                "state_id": "path_compiled_context",
                            }
                        ],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    init_output = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            session_id,
            "--era-id",
            "era_review_lineage_v1",
            "--session-root",
            str(session_root),
            "--ideas-json",
            '["A repeated-format fast path probably helps this parser."]',
            "--canonicalization-model",
            "stub",
        ]
    )
    preview_digest = str(init_output["preview_digest"])

    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            session_id,
            "--preview-digest",
            preview_digest,
            "--session-root",
            str(session_root),
        ]
    )
    _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(frontier_path),
        ]
    )

    review_bundle = _run_cli(
        [
            "session",
            "review-bundle",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
        ]
    )
    lane = _require_mapping(cast(list[object], review_bundle["lanes"])[0])
    proposal = _require_mapping(cast(list[object], review_bundle["proposals"])[0])
    lineage = _require_mapping(review_bundle["lineage"])
    recommended = _require_mapping(lineage["recommended_proposal"])

    assert lane["source_idea_ids"] == ["idea_001"]
    assert "idea_001" in cast(list[object], lane["source_belief_ids"])
    assert proposal["source_idea_ids"] == ["idea_001"]
    assert recommended["source_idea_ids"] == ["idea_001"]
    assert recommended["candidate_id"] == "cand_lineage"
