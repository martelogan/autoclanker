from __future__ import annotations

import io
import json
import subprocess

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast

from autoclanker.cli import EXIT_SESSION_ERROR, main
from tests.compliance import covers


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


def _run_cli_expect_failure(argv: list[str]) -> tuple[int, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(argv)
    return exit_code, stderr.getvalue()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _init_git_workspace(root: Path) -> None:
    (root / "benchmarks").mkdir(parents=True, exist_ok=True)
    (root / "benchmarks" / "fixture.txt").write_text("benchmark:v1\n", encoding="utf-8")
    (root / "autoclanker.eval.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (root / "requirements.lock").write_text("fixture==1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "autoclanker-tests@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "autoclanker tests"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _write_fixture_adapter_config(
    path: Path,
    *,
    workspace_root: Path,
    eval_policy: dict[str, object] | None = None,
) -> None:
    metadata: dict[str, object] = {
        "benchmark_root": str(workspace_root / "benchmarks"),
        "eval_harness_path": str(workspace_root / "autoclanker.eval.sh"),
        "environment_paths": [str(workspace_root / "requirements.lock")],
        "workspace_root": str(workspace_root),
        "workspace_snapshot_mode": "git_worktree",
    }
    if eval_policy is not None:
        metadata["eval_policy"] = eval_policy
    _write_json(
        path,
        {
            "adapter": {
                "kind": "fixture",
                "mode": "fixture",
                "session_root": ".autoclanker",
                "metadata": metadata,
            }
        },
    )


def _candidate_payload(
    candidate_id: str, genotype: list[dict[str, str]]
) -> dict[str, object]:
    return {"candidate_id": candidate_id, "genotype": genotype}


@covers("M3-007", "M3-008")
def test_session_init_persists_eval_contract_and_reports_drift(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    _write_fixture_adapter_config(adapter_config, workspace_root=workspace_root)

    init_payload = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "contract_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert init_payload["eval_contract_matches_current"] is True
    assert init_payload["eval_contract_drift_status"] == "locked"
    contract = _require_mapping(init_payload["eval_contract"])
    assert contract["measurement_mode"] == "parallel_ok"
    assert contract["stabilization_mode"] == "soft"
    session_path = session_root / "contract_demo"
    assert (session_path / "eval_contract.json").exists()

    (workspace_root / "benchmarks" / "fixture.txt").write_text(
        "benchmark:v2\n",
        encoding="utf-8",
    )
    drifted = _run_cli(
        [
            "session",
            "status",
            "--session-id",
            "contract_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert drifted["eval_contract_drift_status"] == "drifted"
    assert drifted["eval_contract_matches_current"] is False


@covers("M3-015")
def test_review_bundle_uses_same_eval_contract_drift_truth_as_status(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    _write_fixture_adapter_config(adapter_config, workspace_root=workspace_root)

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "review_contract_demo",
            "--era-id",
            "era_review_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    review_bundle = _run_cli(
        [
            "session",
            "review-bundle",
            "--session-id",
            "review_contract_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    status = _run_cli(
        [
            "session",
            "status",
            "--session-id",
            "review_contract_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert _require_mapping(review_bundle["trust"])["status"] == "locked"
    assert _require_mapping(review_bundle["trust"])[
        "locked_eval_contract_digest"
    ] == status["eval_contract_digest"]
    assert _require_mapping(review_bundle["trust"])[
        "current_eval_contract_digest"
    ] == status["current_eval_contract_digest"]
    assert _require_mapping(review_bundle["trust"])[
        "status"
    ] == status["eval_contract_drift_status"]

    (workspace_root / "benchmarks" / "fixture.txt").write_text(
        "benchmark:v2\n",
        encoding="utf-8",
    )
    drifted_review_bundle = _run_cli(
        [
            "session",
            "review-bundle",
            "--session-id",
            "review_contract_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    drifted_status = _run_cli(
        [
            "session",
            "status",
            "--session-id",
            "review_contract_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert _require_mapping(drifted_review_bundle["trust"])["status"] == "drifted"
    assert _require_mapping(drifted_review_bundle["trust"])[
        "status"
    ] == drifted_status["eval_contract_drift_status"]
    assert _require_mapping(drifted_review_bundle["trust"])[
        "eval_contract_matches_current"
    ] is False


@covers("M3-009", "M3-011", "M3-013")
def test_session_run_eval_uses_isolated_worktree_and_records_eval_run_artifact(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    candidate_path = tmp_path / "candidate.json"
    _write_fixture_adapter_config(
        adapter_config,
        workspace_root=workspace_root,
        eval_policy={"mode": "exclusive", "stabilization": "soft"},
    )
    _write_json(
        candidate_path,
        _candidate_payload(
            "cand_compiled",
            [{"gene_id": "parser.matcher", "state_id": "matcher_compiled"}],
        ),
    )

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "run_eval_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    result = _run_cli(
        [
            "session",
            "run-eval",
            "--session-id",
            "run_eval_demo",
            "--candidate-input",
            str(candidate_path),
            "--seed",
            "7",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    eval_result = _require_mapping(result["result"])
    execution = _require_mapping(eval_result["execution_metadata"])
    assert execution["isolation_mode"] == "git_worktree"
    assert isinstance(execution["workspace_root"], str)
    assert not str(execution["workspace_root"]).startswith(str(workspace_root))
    assert execution["measurement_mode"] == "exclusive"
    assert execution["stabilization_mode"] == "soft"
    assert execution["lease_acquired"] is True
    assert isinstance(execution["lease_scope"], str)
    assert isinstance(execution["lease_wait_sec"], int | float)
    assert isinstance(execution["stabilization_delay_sec"], int | float)
    assert execution["lease_wait_sec"] >= 0
    assert execution["stabilization_delay_sec"] >= 0
    assert (
        session_root / "run_eval_demo" / "eval_runs" / "cand_compiled.json"
    ).exists()
    status = _run_cli(
        [
            "session",
            "status",
            "--session-id",
            "run_eval_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert status["last_eval_measurement_mode"] == "exclusive"
    assert status["last_eval_stabilization_mode"] == "soft"
    assert status["last_eval_used_lease"] is True
    assert isinstance(status["last_eval_noisy_system"], bool)


@covers("M3-010")
def test_session_ingest_eval_rejects_contract_drift(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    eval_result_path = tmp_path / "eval.json"
    candidate_path = tmp_path / "temp-candidate.json"
    _write_fixture_adapter_config(adapter_config, workspace_root=workspace_root)
    _write_json(
        candidate_path,
        _candidate_payload(
            "cand_temp",
            [{"gene_id": "parser.matcher", "state_id": "matcher_compiled"}],
        ),
    )

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "ingest_contract_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    valid_result = _require_mapping(
        _run_cli(
            [
                "session",
                "run-eval",
                "--session-id",
                "ingest_contract_demo",
                "--candidate-input",
                str(candidate_path),
                "--session-root",
                str(session_root),
                "--adapter-config",
                str(adapter_config),
            ]
        )["result"]
    )
    # Remove the auto-appended observation and reuse the result as an external ingest payload.
    observations_path = session_root / "ingest_contract_demo" / "observations.jsonl"
    observations_path.write_text("", encoding="utf-8")
    contract = _require_mapping(valid_result["eval_contract"])
    contract["benchmark_tree_digest"] = "sha256:mismatched"
    valid_result["eval_contract"] = contract
    _write_json(eval_result_path, valid_result)

    exit_code, stderr = _run_cli_expect_failure(
        [
            "session",
            "ingest-eval",
            "--session-id",
            "ingest_contract_demo",
            "--input",
            str(eval_result_path),
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert exit_code == EXIT_SESSION_ERROR
    assert "locked session contract" in stderr


@covers("M4-006")
def test_session_suggest_and_frontier_status_preserve_frontier_metadata(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    frontier_path = tmp_path / "frontier.json"
    _write_fixture_adapter_config(adapter_config, workspace_root=workspace_root)
    _write_json(
        frontier_path,
        {
            "frontier_id": "frontier_parser",
            "default_family_id": "family_default",
            "candidates": [
                {
                    "candidate_id": "cand_default",
                    "family_id": "baseline",
                    "origin_kind": "seed",
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_basic"}
                    ],
                },
                {
                    "candidate_id": "cand_compiled",
                    "family_id": "matcher_family",
                    "origin_kind": "belief",
                    "parent_belief_ids": ["belief_compiled"],
                    "budget_weight": 2.0,
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"}
                    ],
                },
                {
                    "candidate_id": "cand_context_pair",
                    "family_id": "plan_family",
                    "origin_kind": "merge",
                    "parent_candidate_ids": ["cand_compiled", "cand_default"],
                    "origin_query_ids": ["query_compare_001"],
                    "notes": "combine matcher and plan",
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"},
                        {"gene_id": "parser.plan", "state_id": "plan_context_pair"},
                    ],
                },
            ],
        },
    )

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "frontier_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    suggest = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "frontier_demo",
            "--candidates-input",
            str(frontier_path),
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    ranked_candidates = cast(list[object], suggest["ranked_candidates"])
    top_candidate = _require_mapping(ranked_candidates[0])
    assert top_candidate["family_id"] in {"matcher_family", "plan_family", "baseline"}
    assert "frontier_summary" in suggest
    compiled_candidate = next(
        _require_mapping(item)
        for item in ranked_candidates
        if _require_mapping(item)["candidate_id"] == "cand_compiled"
    )
    assert compiled_candidate["origin_kind"] == "belief"
    assert compiled_candidate["parent_belief_ids"] == ["belief_compiled"]
    assert compiled_candidate["budget_weight"] == 2.0
    merged_candidate = next(
        _require_mapping(item)
        for item in ranked_candidates
        if _require_mapping(item)["candidate_id"] == "cand_context_pair"
    )
    assert merged_candidate["origin_kind"] == "merge"
    assert merged_candidate["parent_candidate_ids"] == [
        "cand_compiled",
        "cand_default",
    ]
    assert merged_candidate["origin_query_ids"] == ["query_compare_001"]
    assert merged_candidate["notes"] == "combine matcher and plan"

    frontier_status = _run_cli(
        [
            "session",
            "frontier-status",
            "--session-id",
            "frontier_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    summary = _require_mapping(frontier_status["frontier_summary"])
    assert summary["family_count"] == 3
    assert _require_mapping(summary["budget_allocations"]) == {
        "baseline": 0.25,
        "matcher_family": 0.5,
        "plan_family": 0.25,
    }
    pending_merges = cast(list[object], summary["pending_merge_suggestions"])
    assert pending_merges
    merge = _require_mapping(pending_merges[0])
    assert set(cast(list[object], merge["candidate_ids"])) <= {
        "cand_default",
        "cand_compiled",
        "cand_context_pair",
    }
    assert set(cast(list[object], merge["family_ids"])) <= {
        "baseline",
        "matcher_family",
        "plan_family",
    }
    assert (session_root / "frontier_demo" / "frontier_status.json").exists()
    proposal_ledger = _require_mapping(
        json.loads(
            (session_root / "frontier_demo" / "proposal_ledger.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert (
        proposal_ledger["current_proposal_id"]
        == f"proposal_{top_candidate['candidate_id']}"
    )
    proposal_entries = cast(list[object], proposal_ledger["entries"])
    assert any(
        _require_mapping(item)["readiness_state"] in {"candidate", "deferred"}
        for item in proposal_entries
    )
    delta_summary = _require_mapping(
        json.loads(
            (session_root / "frontier_demo" / "belief_delta_summary.json").read_text(
                encoding="utf-8"
            )
        )
    )
    assert "notes" in delta_summary
    query_artifact = _require_mapping(
        json.loads((session_root / "frontier_demo" / "query.json").read_text("utf-8"))
    )
    assert query_artifact["objective_backend"] == "heuristic_independent_normal"
    assert query_artifact["acquisition_backend"] == "optimistic_upper_confidence"
    assert query_artifact["follow_up_query_type"] == "pairwise_preference"
    assert isinstance(query_artifact["follow_up_comparison"], str)
    persisted_ranked = cast(list[object], query_artifact["ranked_candidates"])
    persisted_compiled = next(
        _require_mapping(item)
        for item in persisted_ranked
        if _require_mapping(item)["candidate_id"] == "cand_compiled"
    )
    assert persisted_compiled["parent_belief_ids"] == ["belief_compiled"]
    status = _run_cli(
        [
            "session",
            "status",
            "--session-id",
            "frontier_demo",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    assert status["last_objective_backend"] == "heuristic_independent_normal"
    assert status["last_acquisition_backend"] == "optimistic_upper_confidence"
    assert status["last_follow_up_query_type"] == "pairwise_preference"
    assert isinstance(status["last_follow_up_comparison"], str)
    assert cast(int, status["proposal_entry_count"]) >= 1
    assert status["latest_proposal_id"] == f"proposal_{top_candidate['candidate_id']}"
    assert status["latest_proposal_state"] in {"candidate", "deferred"}


@covers("M3-012", "M4-006")
def test_session_run_frontier_executes_batch_and_persists_frontier_summary(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    _init_git_workspace(workspace_root)
    adapter_config = tmp_path / "adapter.json"
    session_root = tmp_path / "sessions"
    frontier_path = tmp_path / "frontier.json"
    _write_fixture_adapter_config(adapter_config, workspace_root=workspace_root)
    _write_json(
        frontier_path,
        {
            "frontier_id": "frontier_run_batch",
            "default_family_id": "family_default",
            "candidates": [
                {
                    "candidate_id": "cand_default",
                    "family_id": "baseline",
                    "origin_kind": "seed",
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_basic"}
                    ],
                },
                {
                    "candidate_id": "cand_compiled",
                    "family_id": "matcher_family",
                    "origin_kind": "belief",
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"}
                    ],
                },
            ],
        },
    )

    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "frontier_batch_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    result = _run_cli(
        [
            "session",
            "run-frontier",
            "--session-id",
            "frontier_batch_demo",
            "--frontier-input",
            str(frontier_path),
            "--session-root",
            str(session_root),
            "--adapter-config",
            str(adapter_config),
        ]
    )
    results = cast(list[object], result["results"])
    assert len(results) == 2
    ranked_candidates = cast(list[object], result["ranked_candidates"])
    assert len(ranked_candidates) == 2
    summary = _require_mapping(result["frontier_summary"])
    assert summary["candidate_count"] == 2
    assert summary["family_count"] == 2
    assert (
        session_root / "frontier_batch_demo" / "eval_runs" / "cand_default.json"
    ).exists()
    assert (
        session_root / "frontier_batch_demo" / "eval_runs" / "cand_compiled.json"
    ).exists()
    assert (session_root / "frontier_batch_demo" / "frontier_status.json").exists()


@covers("M4-004")
def test_legacy_candidate_pool_still_normalizes_into_default_frontier_family(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "sessions"
    candidates_path = tmp_path / "legacy-candidates.json"
    _write_json(
        candidates_path,
        {
            "candidates": [
                {
                    "candidate_id": "cand_legacy",
                    "genotype": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"}
                    ],
                }
            ]
        },
    )
    _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "legacy_demo",
            "--era-id",
            "era_contract_v1",
            "--session-root",
            str(session_root),
        ]
    )
    suggest = _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            "legacy_demo",
            "--candidates-input",
            str(candidates_path),
            "--session-root",
            str(session_root),
        ]
    )
    ranked_candidates = cast(list[object], suggest["ranked_candidates"])
    first = _require_mapping(ranked_candidates[0])
    assert first["family_id"] == "family_default"
