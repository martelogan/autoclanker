from __future__ import annotations

import json
import subprocess
import sys

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError("Expected a JSON object.")
    return cast(dict[str, object], value)


def _require_candidate_list(payload: Mapping[str, object]) -> list[dict[str, object]]:
    ranked_candidates = payload.get("ranked_candidates")
    if not isinstance(ranked_candidates, list):
        raise AssertionError("Expected a ranked_candidates list.")
    candidates: list[dict[str, object]] = []
    for item in cast(list[object], ranked_candidates):
        if not isinstance(item, dict):
            raise AssertionError("Each ranked candidate must be an object.")
        candidates.append(cast(dict[str, object], item))
    return candidates


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"Expected numeric field {key!r}.")
    return float(value)


def _require_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise AssertionError(f"Expected boolean field {key!r}.")
    return value


def _candidate_by_id(
    ranked_candidates: list[dict[str, object]],
    candidate_id: str,
) -> dict[str, object]:
    for candidate in ranked_candidates:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise AssertionError(f"Missing candidate {candidate_id!r}.")


def _ingest_eval_directory(
    *,
    session_id: str,
    eval_dir: Path,
    session_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for eval_path in sorted(eval_dir.glob("*.json")):
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


@covers("M2-002", "M3-003", "M6-001", "M7-003")
def test_bayes_complex_showcase_outperforms_control_via_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    eval_dir = tmp_path / "evals"
    beliefs_path = (
        ROOT / "examples" / "live_exercises" / "bayes_complex" / "beliefs.yaml"
    )
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_complex" / "candidates.json"
    )
    expected = json.loads(
        (
            ROOT
            / "examples"
            / "live_exercises"
            / "bayes_complex"
            / "expected_outcome.json"
        ).read_text(encoding="utf-8")
    )
    expected_mapping = _require_mapping(expected)

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
    init_output = _require_mapping(json.loads(capsys.readouterr().out))
    belief_session_id = str(init_output["session_id"])
    preview_digest = str(init_output["preview_digest"])

    control_session_id = "exercise_bayes_complex_control"
    assert (
        main(
            [
                "session",
                "init",
                "--session-id",
                control_session_id,
                "--era-id",
                "era_parser_advanced",
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
                "apply-beliefs",
                "--session-id",
                belief_session_id,
                "--preview-digest",
                preview_digest,
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
                "suggest",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        == 0
    )
    cold_beliefs = _require_mapping(json.loads(capsys.readouterr().out))

    assert (
        main(
            [
                "session",
                "suggest",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        == 0
    )
    cold_control = _require_mapping(json.loads(capsys.readouterr().out))

    cold_expected = _require_mapping(expected_mapping["cold_start"])
    cold_belief_candidates = _require_candidate_list(cold_beliefs)
    cold_control_candidates = _require_candidate_list(cold_control)
    cold_good_pair = _candidate_by_id(
        cold_belief_candidates, "cand_c_compiled_context_pair"
    )
    cold_lr_only = _candidate_by_id(cold_belief_candidates, "cand_b_compiled_matcher")
    cold_default = _candidate_by_id(cold_belief_candidates, "cand_a_default")
    cold_bad_oom = _candidate_by_id(
        cold_belief_candidates, "cand_d_wide_window_large_chunk"
    )

    assert (
        cold_belief_candidates[0]["candidate_id"]
        == cold_expected["beliefs_top_candidate"]
    )
    assert (
        cold_control_candidates[0]["candidate_id"]
        == cold_expected["control_top_candidate"]
    )
    assert _require_float(cold_good_pair, "predicted_utility") >= _require_float(
        cold_expected, "min_good_pair_predicted_utility"
    )
    assert _require_float(cold_good_pair, "predicted_utility") - _require_float(
        cold_lr_only, "predicted_utility"
    ) >= _require_float(cold_expected, "min_good_pair_margin_over_lr_only")
    assert _require_float(cold_bad_oom, "valid_probability") <= _require_float(
        cold_expected, "max_bad_oom_valid_probability"
    )
    assert _require_float(cold_bad_oom, "valid_probability") < _require_float(
        cold_default, "valid_probability"
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "live" / "generate_bayes_complex_evals.py"),
            "--output-dir",
            str(eval_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    assert sorted(path.name for path in eval_dir.glob("*.json")) == [
        "cand_a_default.json",
        "cand_b_compiled_matcher.json",
        "cand_d_wide_window_large_chunk.json",
    ]

    _ingest_eval_directory(
        session_id=belief_session_id,
        eval_dir=eval_dir,
        session_root=session_root,
        capsys=capsys,
    )
    _ingest_eval_directory(
        session_id=control_session_id,
        eval_dir=eval_dir,
        session_root=session_root,
        capsys=capsys,
    )

    assert (
        main(
            [
                "session",
                "fit",
                "--session-id",
                belief_session_id,
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
                control_session_id,
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
                "suggest",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        == 0
    )
    fitted_beliefs = _require_mapping(json.loads(capsys.readouterr().out))
    assert (
        main(
            [
                "session",
                "suggest",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        == 0
    )
    fitted_control = _require_mapping(json.loads(capsys.readouterr().out))

    belief_good_pair = _candidate_by_id(
        _require_candidate_list(fitted_beliefs), "cand_c_compiled_context_pair"
    )
    control_good_pair = _candidate_by_id(
        _require_candidate_list(fitted_control), "cand_c_compiled_context_pair"
    )
    post_expected = _require_mapping(expected_mapping["post_observations"])
    assert _require_float(belief_good_pair, "predicted_utility") - _require_float(
        control_good_pair, "predicted_utility"
    ) >= _require_float(post_expected, "min_good_pair_uplift_over_control")

    assert (
        main(
            [
                "session",
                "recommend-commit",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    belief_decision = _require_mapping(json.loads(capsys.readouterr().out))
    assert (
        main(
            [
                "session",
                "recommend-commit",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root),
            ]
        )
        == 0
    )
    control_decision = _require_mapping(json.loads(capsys.readouterr().out))

    assert _require_bool(belief_decision, "recommended") == _require_bool(
        post_expected, "beliefs_recommended"
    )
    assert _require_bool(control_decision, "recommended") == _require_bool(
        post_expected, "control_recommended"
    )
    assert _require_float(belief_decision, "gain_probability") >= _require_float(
        post_expected, "min_beliefs_gain_probability"
    )
    assert _require_float(control_decision, "gain_probability") <= _require_float(
        post_expected, "max_control_gain_probability"
    )
