from __future__ import annotations

import io
import json
import os

from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import cast

import pytest

from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]
_PROVIDER = "anthropic"


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


def _suggest_payload(session_root: Path, session_id: str) -> dict[str, object]:
    return _run_cli(
        [
            "session",
            "suggest",
            "--session-id",
            session_id,
            "--session-root",
            str(session_root),
            "--candidates-input",
            str(
                ROOT
                / "examples"
                / "live_exercises"
                / "bayes_quickstart"
                / "candidates.json"
            ),
        ]
    )


def _candidate_utility(payload: dict[str, object], candidate_id: str) -> float:
    ranked = cast(list[object], payload["ranked_candidates"])
    for item in ranked:
        mapping = _require_mapping(item)
        if mapping["candidate_id"] == candidate_id:
            utility = mapping["predicted_utility"]
            if not isinstance(utility, int | float):
                raise AssertionError("predicted_utility must be numeric.")
            return float(utility)
    raise AssertionError(f"Missing candidate {candidate_id!r}.")


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("AUTOCLANKER_ENABLE_LLM_LIVE") != "1"
    or not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AUTOCLANKER_ANTHROPIC_API_KEY")
    ),
    reason="Requires opt-in billed LLM live execution and an Anthropic API key.",
)
@covers("M1-LIVE-001")
def test_live_anthropic_canonicalize_ideas_cli_emits_typed_belief_payload() -> None:
    ideas_json = json.dumps(
        [
            "Prefer the lane that front-loads the grammar work and keeps nearby breadcrumbs attached to each alarm.",
            "Treat the ballooning retention posture as dangerous on long traces.",
        ]
    )

    payload = _run_cli(
        [
            "beliefs",
            "canonicalize-ideas",
            "--era-id",
            "era_log_parser_v1",
            "--canonicalization-model",
            _PROVIDER,
            "--ideas-json",
            ideas_json,
        ]
    )

    summary = _require_mapping(payload["canonicalization_summary"])
    records = [
        _require_mapping(item) for item in cast(list[object], summary["records"])
    ]
    beliefs = [
        _require_mapping(item) for item in cast(list[object], payload["beliefs"])
    ]

    assert summary["mode"] == "hybrid"
    assert str(summary["model_name"]).startswith("anthropic:")
    assert any(record["status"] == "resolved" for record in records)
    assert any(belief["kind"] != "proposal" for belief in beliefs)
    assert all(
        belief["kind"]
        in {
            "proposal",
            "idea",
            "relation",
            "expert_prior",
            "graph_directive",
        }
        for belief in beliefs
    )
    if "surface_overlay" in payload:
        surface_overlay = _require_mapping(payload["surface_overlay"])
        assert isinstance(surface_overlay["registry"], dict)
        assert isinstance(surface_overlay["surface_summary"], dict)


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("AUTOCLANKER_ENABLE_LLM_LIVE") != "1"
    or not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AUTOCLANKER_ANTHROPIC_API_KEY")
    ),
    reason="Requires opt-in billed LLM live execution and an Anthropic API key.",
)
@covers("M6-LIVE-001")
def test_live_anthropic_canonicalization_changes_parser_ranking(tmp_path: Path) -> None:
    ideas_json = json.dumps(
        [
            "Prefer the lane that front-loads the grammar work and keeps nearby breadcrumbs attached to each alarm.",
            "Treat the ballooning retention posture as dangerous on long traces.",
        ]
    )
    session_root = tmp_path / "sessions"

    proposal_only_init = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "proposal_only",
            "--era-id",
            "era_log_parser_v1",
            "--session-root",
            str(session_root),
            "--ideas-json",
            ideas_json,
        ]
    )
    assert proposal_only_init["beliefs_status"] == "preview_pending"
    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            "proposal_only",
            "--preview-digest",
            str(proposal_only_init["preview_digest"]),
            "--session-root",
            str(session_root),
        ]
    )
    proposal_suggest = _suggest_payload(session_root, "proposal_only")
    assert (
        _require_mapping(cast(list[object], proposal_suggest["ranked_candidates"])[0])[
            "candidate_id"
        ]
        == "cand_a_default"
    )

    llm_init = _run_cli(
        [
            "session",
            "init",
            "--session-id",
            "llm_guided",
            "--era-id",
            "era_log_parser_v1",
            "--session-root",
            str(session_root),
            "--ideas-json",
            ideas_json,
            "--canonicalization-model",
            _PROVIDER,
        ]
    )
    summary = _require_mapping(llm_init["canonicalization_summary"])
    assert summary["mode"] == "hybrid"
    assert str(summary["model_name"]).startswith("anthropic:")
    records = cast(list[object], summary["records"])
    assert any(_require_mapping(item)["status"] == "resolved" for item in records)
    session_path = Path(str(llm_init["session_path"]))
    assert (session_path / "canonicalization_summary.json").exists()
    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            "llm_guided",
            "--preview-digest",
            str(llm_init["preview_digest"]),
            "--session-root",
            str(session_root),
        ]
    )
    llm_suggest = _suggest_payload(session_root, "llm_guided")

    assert (
        _require_mapping(cast(list[object], llm_suggest["ranked_candidates"])[0])[
            "candidate_id"
        ]
        != "cand_a_default"
    )
    assert _candidate_utility(
        llm_suggest, "cand_c_compiled_context_pair"
    ) > _candidate_utility(proposal_suggest, "cand_c_compiled_context_pair")
