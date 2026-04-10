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
from autoclanker.bayes_layer.types import SessionContext
from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


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


@covers("M3-005")
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
    assert any(
        "main:search.repeated_format_fast_path=path_compiled_context" in str(entry)
        for entry in cast(list[object], top_influence["influence_summary"])
    )

    _run_cli(
        [
            "session",
            "recommend-commit",
            "--session-id",
            "hybrid_parser",
            "--session-root",
            str(session_root),
        ]
    )
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
