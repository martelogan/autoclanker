from __future__ import annotations

import json

from pathlib import Path

import pytest

from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.types import ValidAdapterConfig, to_json_value
from autoclanker.cli import EXIT_SESSION_ERROR, EXIT_VALIDATION_ERROR, main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _fixture_adapter() -> FixtureAdapter:
    return FixtureAdapter(
        ValidAdapterConfig(
            kind="fixture",
            mode="fixture",
            session_root=".autoclanker",
        )
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(to_json_value(payload), indent=2), encoding="utf-8")


@covers("M3-002", "M3-014", "M6-001", "M6-003")
def test_session_manifest_and_status_record_preview_gate_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    beliefs_path = ROOT / "examples/human_beliefs/basic_session.yaml"

    assert (
        main(
            [
                "session",
                "init",
                "--beliefs-input",
                str(beliefs_path),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    init_output = json.loads(capsys.readouterr().out)
    session_id = str(init_output["session_id"])

    assert init_output["beliefs_status"] == "preview_pending"
    assert init_output["compiled_priors_active"] is False
    assert init_output["preview_digest"]
    assert init_output["adapter_execution_mode"] == "fixture"

    store = FilesystemSessionStore(root=session_root)
    manifest = store.load_manifest(session_id)
    assert manifest.beliefs_status == "preview_pending"
    assert manifest.compiled_priors_active is False
    assert manifest.preview_digest is not None

    assert (
        main(
            [
                "session",
                "apply-beliefs",
                "--session-id",
                session_id,
                "--preview-digest",
                str(init_output["preview_digest"]),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    applied_output = json.loads(capsys.readouterr().out)

    assert applied_output["beliefs_status"] == "applied"
    assert applied_output["compiled_priors_active"] is True

    assert (
        main(
            [
                "session",
                "status",
                "--session-id",
                session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    status_output = json.loads(capsys.readouterr().out)
    artifact_paths = status_output["artifact_paths"]
    assert artifact_paths["results_markdown"].endswith("RESULTS.md")
    assert artifact_paths["convergence_plot"].endswith("convergence.png")
    assert artifact_paths["candidate_rankings_plot"].endswith("candidate_rankings.png")
    assert artifact_paths["prior_graph_plot"].endswith("belief_graph_prior.png")
    assert artifact_paths["posterior_graph_plot"].endswith("belief_graph_posterior.png")
    assert artifact_paths["belief_delta_summary"].endswith("belief_delta_summary.json")
    assert artifact_paths["proposal_ledger"].endswith("proposal_ledger.json")


@covers("M3-004", "M6-002")
def test_session_ingest_eval_rejects_wrong_era(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    assert (
        main(
            [
                "session",
                "init",
                "--session-id",
                "session_wrong_era",
                "--era-id",
                "era_expected",
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    eval_result = _fixture_adapter().evaluate_candidate(
        era_id="era_other",
        candidate_id="cand_wrong_era",
        genotype=_fixture_adapter().build_registry().default_genotype(),
        seed=7,
    )
    eval_path = tmp_path / "wrong-era.json"
    _write_json(eval_path, eval_result)

    exit_code = main(
        [
            "session",
            "ingest-eval",
            "--session-id",
            "session_wrong_era",
            "--input",
            str(eval_path),
            "--session-root",
            str(session_root),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_SESSION_ERROR
    assert '"ok": false' in captured.err
    assert "era_expected" in captured.err


@covers("M1-002", "M6-002")
def test_candidate_input_validation_returns_stable_validation_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    assert (
        main(
            [
                "session",
                "init",
                "--session-id",
                "session_bad_candidates",
                "--era-id",
                "era_003",
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    candidates_path = tmp_path / "bad-candidates.json"
    candidates_path.write_text(
        json.dumps({"candidates": [{"candidate_id": "cand_bad", "genotype": [{}]}]}),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "session",
            "suggest",
            "--session-id",
            "session_bad_candidates",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_VALIDATION_ERROR
    assert '"ok": false' in captured.err
    assert "gene_id" in captured.err
