from __future__ import annotations

import json

from pathlib import Path

import pytest

from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.types import ValidAdapterConfig, to_json_value
from autoclanker.cli import EXIT_SESSION_ERROR, main
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


@covers("M3-001", "M3-003", "M6-001", "M6-003")
def test_cli_session_flow_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    beliefs_path = ROOT / "examples/human_beliefs/basic_session.yaml"

    exit_code = main(
        [
            "session",
            "init",
            "--beliefs-input",
            str(beliefs_path),
            "--session-root",
            str(session_root),
        ]
    )
    init_output = json.loads(capsys.readouterr().out)
    session_id = str(init_output["session_id"])

    assert exit_code == 0
    assert session_id == "demo_basic"
    assert init_output["beliefs_status"] == "preview_pending"
    assert init_output["compiled_priors_active"] is False

    assert (
        main(
            [
                "session",
                "fit",
                "--session-id",
                session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == EXIT_SESSION_ERROR
    )
    fit_error = capsys.readouterr()
    assert '"ok": false' in fit_error.err
    assert "apply-beliefs" in fit_error.err

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
    apply_output = json.loads(capsys.readouterr().out)
    assert apply_output["beliefs_status"] == "applied"
    assert apply_output["compiled_priors_active"] is True

    adapter = _fixture_adapter()
    eval_result = adapter.evaluate_candidate(
        era_id="era_003",
        candidate_id="cand_cli",
        genotype=adapter.build_registry().default_genotype(),
        seed=11,
    )
    eval_path = tmp_path / "eval.json"
    _write_json(eval_path, eval_result)

    assert (
        main(
            [
                "session",
                "ingest-eval",
                "--session-id",
                session_id,
                "--input",
                str(eval_path),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "session",
                "fit",
                "--session-id",
                session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    fit_output = json.loads(capsys.readouterr().out)

    assert fit_output["observation_count"] >= 1

    assert (
        main(
            [
                "session",
                "suggest",
                "--session-id",
                session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    suggest_output = json.loads(capsys.readouterr().out)

    assert suggest_output["ranked_candidates"]
    assert (
        main(
            [
                "session",
                "recommend-commit",
                "--session-id",
                session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    decision_output = json.loads(capsys.readouterr().out)

    assert decision_output["session_id"] == session_id
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

    assert status_output["ready_for_fit"] is True
    assert status_output["compiled_priors_active"] is True
