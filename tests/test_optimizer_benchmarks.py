from __future__ import annotations

import io
import json

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


def _top_candidate_id(payload: dict[str, object]) -> str:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    return str(_require_mapping(ranked_candidates[0])["candidate_id"])


def _candidate_by_id(
    payload: dict[str, object],
    candidate_id: str,
) -> dict[str, object]:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    for item in ranked_candidates:
        mapping = _require_mapping(item)
        if mapping["candidate_id"] == candidate_id:
            return mapping
    raise AssertionError(f"Missing candidate {candidate_id!r}.")


def _predictive_margin(payload: dict[str, object], candidate_id: str) -> float:
    candidate = _candidate_by_id(payload, candidate_id)
    value = candidate.get("predicted_utility")
    if not isinstance(value, (int, float)):
        raise AssertionError("predicted_utility must be numeric.")
    return float(value)


def _require_sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError("Expected a JSON list.")
    return cast(list[object], value)


def _require_float(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise AssertionError("Expected a numeric value.")
    return float(value)


def _init_and_apply(
    *,
    session_root: Path,
    session_id: str,
    ideas_json: str | None = None,
    canonicalization_model: str | None = None,
) -> None:
    argv = [
        "session",
        "init",
        "--session-id",
        session_id,
        "--era-id",
        "era_log_parser_v1",
        "--session-root",
        str(session_root),
    ]
    if ideas_json is not None:
        argv.extend(["--ideas-json", ideas_json])
    if canonicalization_model is not None:
        argv.extend(["--canonicalization-model", canonicalization_model])
    init_output = _run_cli(argv)
    preview_digest = init_output.get("preview_digest")
    if preview_digest is not None:
        _run_cli(
            [
                "session",
                "apply-beliefs",
                "--session-id",
                session_id,
                "--preview-digest",
                str(preview_digest),
                "--session-root",
                str(session_root),
            ]
        )


@covers("M7-006")
def test_zero_eval_cold_start_parser_lanes_show_value_from_typed_canonical_bayes(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )

    _init_and_apply(session_root=session_root, session_id="outer_control")
    _init_and_apply(
        session_root=session_root,
        session_id="proposal_only",
        ideas_json='["Try a moonbeam dragon refactor with kaleidoscope anchors."]',
    )
    _init_and_apply(
        session_root=session_root,
        session_id="deterministic_bayes",
        ideas_json='["Compiled regex matching probably helps this parser on repeated log formats.","Compiled matching works best together with the context pair plan."]',
    )
    _init_and_apply(
        session_root=session_root,
        session_id="hybrid_bayes",
        ideas_json='["A repeated-format fast path probably helps this parser."]',
        canonicalization_model="stub",
    )

    control = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "outer_control",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )
    proposal_only = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "proposal_only",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )
    deterministic = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "deterministic_bayes",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )
    hybrid = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "hybrid_bayes",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(candidates_path),
        ]
    )

    assert _top_candidate_id(control) == "cand_a_default"
    assert _top_candidate_id(proposal_only) == "cand_a_default"
    assert _top_candidate_id(deterministic) == "cand_c_compiled_context_pair"
    assert _top_candidate_id(hybrid) == "cand_c_compiled_context_pair"

    control_good_pair = _predictive_margin(control, "cand_c_compiled_context_pair")
    proposal_good_pair = _predictive_margin(
        proposal_only, "cand_c_compiled_context_pair"
    )
    deterministic_good_pair = _predictive_margin(
        deterministic, "cand_c_compiled_context_pair"
    )
    hybrid_good_pair = _predictive_margin(hybrid, "cand_c_compiled_context_pair")

    assert proposal_good_pair == control_good_pair
    assert deterministic_good_pair > control_good_pair
    assert hybrid_good_pair > proposal_good_pair


@covers("M7-011")
def test_frontier_heavy_parser_lanes_preserve_family_budgeting_and_merge_guidance(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    frontier_path = tmp_path / "frontier.json"
    frontier_path.write_text(
        json.dumps(
            {
                "frontier_id": "parser_frontier_demo",
                "default_family_id": "family_default",
                "candidates": [
                    {
                        "candidate_id": "cand_a_default",
                        "family_id": "baseline",
                        "origin_kind": "seed",
                        "genotype": [
                            {"gene_id": "parser.matcher", "state_id": "matcher_basic"},
                            {"gene_id": "parser.plan", "state_id": "plan_default"},
                            {"gene_id": "capture.window", "state_id": "window_default"},
                        ],
                    },
                    {
                        "candidate_id": "cand_b_compiled_matcher",
                        "family_id": "matcher_family",
                        "origin_kind": "belief",
                        "parent_belief_ids": ["belief_compiled"],
                        "budget_weight": 2.0,
                        "genotype": [
                            {
                                "gene_id": "parser.matcher",
                                "state_id": "matcher_compiled",
                            },
                            {"gene_id": "parser.plan", "state_id": "plan_default"},
                            {"gene_id": "capture.window", "state_id": "window_default"},
                        ],
                    },
                    {
                        "candidate_id": "cand_c_compiled_context_pair",
                        "family_id": "plan_family",
                        "origin_kind": "merge",
                        "parent_candidate_ids": [
                            "cand_a_default",
                            "cand_b_compiled_matcher",
                        ],
                        "origin_query_ids": ["query_pair_001"],
                        "notes": "merge compiled matcher and context pairing",
                        "genotype": [
                            {
                                "gene_id": "parser.matcher",
                                "state_id": "matcher_compiled",
                            },
                            {
                                "gene_id": "parser.plan",
                                "state_id": "plan_context_pair",
                            },
                            {"gene_id": "capture.window", "state_id": "window_default"},
                        ],
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _init_and_apply(session_root=session_root, session_id="frontier_control")
    _init_and_apply(
        session_root=session_root,
        session_id="frontier_deterministic",
        ideas_json='["Compiled regex matching probably helps this parser on repeated log formats.","Compiled matching works best together with the context pair plan."]',
    )
    _init_and_apply(
        session_root=session_root,
        session_id="frontier_hybrid",
        ideas_json='["A repeated-format fast path probably helps this parser."]',
        canonicalization_model="stub",
    )

    control = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "frontier_control",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(frontier_path),
        ]
    )
    deterministic = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "frontier_deterministic",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(frontier_path),
        ]
    )
    hybrid = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "frontier_hybrid",
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(frontier_path),
        ]
    )

    assert _top_candidate_id(control) == "cand_a_default"
    assert _top_candidate_id(deterministic) == "cand_c_compiled_context_pair"
    assert _top_candidate_id(hybrid) == "cand_c_compiled_context_pair"

    for payload in (deterministic, hybrid):
        frontier_summary = _require_mapping(payload["frontier_summary"])
        assert frontier_summary["family_count"] == 3
        family_representatives = _require_sequence(
            frontier_summary["family_representatives"]
        )
        assert len(family_representatives) == 3
        pending_merges = _require_sequence(
            frontier_summary["pending_merge_suggestions"]
        )
        assert pending_merges
        merge = _require_mapping(pending_merges[0])
        assert set(_require_sequence(merge["family_ids"])).issubset(
            {"baseline", "matcher_family", "plan_family"}
        )
        assert len(_require_sequence(merge["family_ids"])) == 2
        assert set(_require_sequence(merge["candidate_ids"])).issubset(
            {
                "cand_a_default",
                "cand_b_compiled_matcher",
                "cand_c_compiled_context_pair",
            }
        )
        assert len(_require_sequence(merge["candidate_ids"])) == 2
        budget_allocations = _require_mapping(frontier_summary["budget_allocations"])
        assert budget_allocations == {
            "baseline": 0.25,
            "matcher_family": 0.5,
            "plan_family": 0.25,
        }
        assert (
            round(
                sum(_require_float(value) for value in budget_allocations.values()), 3
            )
            == 1.0
        )
