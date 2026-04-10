from __future__ import annotations

import io
import json
import subprocess
import sys

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast

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


def _candidate_by_id(
    ranked_candidates: list[dict[str, object]],
    candidate_id: str,
) -> dict[str, object]:
    for candidate in ranked_candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise AssertionError(f"Missing candidate {candidate_id!r}.")


def _require_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"Expected numeric field {key!r}.")
    return float(value)


def _require_int(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise AssertionError(f"Expected integer field {key!r}.")
    return value


@covers("M1-003", "M6-001", "M7-003")
def test_bayes_quickstart_basic_beliefs_shift_cold_start_ranking(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    beliefs_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "beliefs.yaml"
    )
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )
    expected = _require_mapping(
        json.loads(
            (
                ROOT
                / "examples"
                / "live_exercises"
                / "bayes_quickstart"
                / "expected_outcome.json"
            ).read_text(encoding="utf-8")
        )
    )
    preview_expected = _require_mapping(expected["preview"])
    cold_expected = _require_mapping(expected["cold_start"])

    init_output = _run_cli(
        [
            "session",
            "init",
            "--beliefs-input",
            str(beliefs_path),
            "--session-root",
            str(session_root),
        ]
    )
    preview_digest = str(init_output["preview_digest"])
    session_id = str(init_output["session_id"])

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "quickstart_control",
            "--era-id",
            "era_log_parser_v1",
            "--session-root",
            str(session_root),
        ]
    )
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

    belief_payload = _run_cli(
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
    control_payload = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "quickstart_control",
            "--candidates-input",
            str(candidates_path),
            "--session-root",
            str(session_root),
        ]
    )

    ranked_with_beliefs = [
        _require_mapping(item)
        for item in cast(list[object], belief_payload["ranked_candidates"])
    ]
    ranked_control = [
        _require_mapping(item)
        for item in cast(list[object], control_payload["ranked_candidates"])
    ]

    good_pair = _candidate_by_id(ranked_with_beliefs, "cand_c_compiled_context_pair")
    lr_only = _candidate_by_id(ranked_with_beliefs, "cand_b_compiled_matcher")
    risky = _candidate_by_id(ranked_with_beliefs, "cand_d_wide_capture_window")

    assert (
        ranked_with_beliefs[0]["candidate_id"] == cold_expected["beliefs_top_candidate"]
    )
    assert ranked_control[0]["candidate_id"] == cold_expected["control_top_candidate"]
    assert _require_float(good_pair, "predicted_utility") - _require_float(
        lr_only, "predicted_utility"
    ) >= _require_float(cold_expected, "min_good_pair_margin_over_lr_only")
    assert _require_float(risky, "valid_probability") <= _require_float(
        cold_expected, "max_risky_valid_probability"
    )
    assert _require_float(risky, "predicted_utility") <= _require_float(
        cold_expected, "max_risky_predicted_utility"
    )

    preview_payload = _require_mapping(
        json.loads(
            (session_root / session_id / "compiled_preview.json").read_text(
                encoding="utf-8"
            )
        )
    )
    preview_items = [
        _require_mapping(item)
        for item in cast(list[object], preview_payload["belief_previews"])
    ]
    compiled_count = sum(
        1 for item in preview_items if item["compile_status"] == "compiled"
    )
    proposal_preview = next(
        item for item in preview_items if item["belief_id"] == "qs4"
    )
    risk_preview = next(item for item in preview_items if item["belief_id"] == "qs3")
    risk_compiled_items = [
        _require_mapping(item)
        for item in cast(list[object], risk_preview["compiled_items"])
    ]

    assert compiled_count >= _require_int(
        preview_expected, "compiled_belief_count_at_least"
    )
    assert (
        proposal_preview["compile_status"]
        == preview_expected["proposal_compile_status"]
    )
    assert any(
        item["target_ref"] == preview_expected["risk_target_ref"]
        for item in risk_compiled_items
    )


@covers("M7-002")
def test_bayes_quickstart_replay_script_runs() -> None:
    process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "live" / "replay_bayes_quickstart.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _require_mapping(json.loads(process.stdout))
    observed = _require_mapping(payload["observed"])
    preview_summary = _require_mapping(payload["preview_summary"])

    assert (
        payload["backing_live_exercise"] == "examples/live_exercises/bayes_quickstart"
    )
    assert observed["beliefs_top_candidate"] == "cand_c_compiled_context_pair"
    assert observed["control_top_candidate"] == "cand_a_default"
    assert observed["proposal_compile_status"] == "metadata_only"
    assert _require_int(preview_summary, "compiled_belief_count") >= 3


@covers("M7-005")
def test_bayes_quickstart_ideas_demo_script_runs() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "live" / "replay_ideas_demo.py"),
            "--exercise",
            "bayes_quickstart",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _require_mapping(json.loads(process.stdout))
    observed = _require_mapping(payload["observed"])
    preview_summary = _require_mapping(payload["preview_summary"])
    top_genotype = _require_mapping(observed["top_genotype"])

    assert payload["exercise"] == "bayes_quickstart"
    assert observed["top_candidate"] == "cand_c_compiled_context_pair"
    assert observed["control_top_candidate"] == "cand_a_default"
    assert top_genotype["parser.matcher"] == "matcher_compiled"
    assert top_genotype["parser.plan"] == "plan_context_pair"
    assert preview_summary["proposal_compile_status"] == "metadata_only"
    assert _require_int(preview_summary, "compiled_belief_count") >= 3
