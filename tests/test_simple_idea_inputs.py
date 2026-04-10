from __future__ import annotations

import json

from pathlib import Path
from typing import cast

import pytest

from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _read_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return cast(dict[str, object], json.loads(capsys.readouterr().out))


@covers("M1-001", "M1-003", "M6-001", "M6-003", "M7-005")
def test_simple_idea_file_can_drive_session_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    ideas_path = ROOT / "examples" / "idea_inputs" / "bayes_quickstart.json"
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )

    assert (
        main(
            [
                "session",
                "init",
                "--beliefs-input",
                str(ideas_path),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    init_payload = _read_stdout(capsys)
    session_id = str(init_payload["session_id"])
    assert session_id == "quickstart_log_parser"
    assert init_payload["beliefs_status"] == "preview_pending"

    assert (
        main(
            [
                "session",
                "apply-beliefs",
                "--session-id",
                session_id,
                "--preview-digest",
                str(init_payload["preview_digest"]),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    apply_payload = _read_stdout(capsys)
    assert apply_payload["beliefs_status"] == "applied"

    assert (
        main(
            [
                "session",
                "suggest",
                "--session-id",
                session_id,
                "--candidates-input",
                str(candidates_path),
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    suggest_payload = _read_stdout(capsys)
    ranked = cast(list[object], suggest_payload["ranked_candidates"])
    top_candidate = cast(dict[str, object], ranked[0])
    assert top_candidate["candidate_id"] == "cand_c_compiled_context_pair"


@covers("M1-001", "M1-003", "M6-001", "M7-005")
def test_minimal_idea_file_without_session_context_can_init_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    ideas_path = ROOT / "examples" / "idea_inputs" / "minimal.yaml"

    assert (
        main(
            [
                "session",
                "init",
                "--beliefs-input",
                str(ideas_path),
                "--session-id",
                "minimal_demo_session",
                "--era-id",
                "era_demo_v1",
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    init_payload = _read_stdout(capsys)
    assert init_payload["session_id"] == "minimal_demo_session"
    assert init_payload["beliefs_status"] == "preview_pending"
    assert init_payload["preview_digest"]


@covers("M1-001", "M1-003", "M6-001", "M7-005")
def test_inline_beginner_ideas_json_can_init_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"

    assert (
        main(
            [
                "session",
                "init",
                "--ideas-json",
                '[{"idea":"Compiled regex matching probably helps this parser on repeated log formats.","confidence":2}]',
                "--session-id",
                "inline_demo_session",
                "--era-id",
                "era_demo_v1",
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    init_payload = _read_stdout(capsys)
    assert init_payload["session_id"] == "inline_demo_session"
    assert init_payload["beliefs_status"] == "preview_pending"
    assert init_payload["preview_digest"]


@covers("M1-001", "M1-003", "M6-001", "M7-005")
def test_string_only_inline_ideas_json_can_init_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"

    assert (
        main(
            [
                "session",
                "init",
                "--ideas-json",
                '["Compiled regex matching probably helps this parser on repeated log formats."]',
                "--session-id",
                "inline_string_demo_session",
                "--era-id",
                "era_demo_v1",
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    init_payload = _read_stdout(capsys)
    assert init_payload["session_id"] == "inline_string_demo_session"
    assert init_payload["beliefs_status"] == "preview_pending"
    assert init_payload["preview_digest"]
