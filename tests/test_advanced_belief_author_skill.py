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


@covers("M7-007")
def test_advanced_belief_author_skill_mentions_provider_backed_authoring() -> None:
    skill_path = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "advanced-belief-author"
        / "SKILL.md"
    )
    rendered = skill_path.read_text(encoding="utf-8")

    assert "AUTOCLANKER_CANONICALIZATION_MODEL=anthropic" in rendered
    assert "autoclanker adapter surface" in rendered
    assert "expert_prior" in rendered
    assert "graph_directive" in rendered


@covers("M7-007")
def test_advanced_belief_author_skill_recommended_beginner_command_is_runnable() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(
            [
                "beliefs",
                "canonicalize-ideas",
                "--input",
                str(root / "examples" / "idea_inputs" / "minimal.json"),
                "--era-id",
                "era_my_app_v1",
            ]
        )
    assert exit_code == 0, stderr.getvalue()
    payload = cast(dict[str, object], json.loads(stdout.getvalue()))
    assert payload["belief_count"] == 2
    first_belief = cast(list[object], payload["beliefs"])[0]
    assert cast(dict[str, object], first_belief)["kind"] == "idea"


@covers("M7-007")
def test_advanced_belief_author_skill_provider_guided_workflow_emits_valid_advanced_batch() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    stdout = io.StringIO()
    stderr = io.StringIO()
    ideas_json = json.dumps(
        [
            "Give compiled matching an explicit stronger prior on this parser.",
            "Keep compiled matching and the context-pair plan together in the interaction screen.",
        ]
    )

    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(
            [
                "beliefs",
                "canonicalize-ideas",
                "--ideas-json",
                ideas_json,
                "--era-id",
                "era_log_parser_v1",
                "--canonicalization-mode",
                "llm",
                "--canonicalization-model",
                "tests.fixtures.advanced_skill_canonicalizer",
            ]
        )
    assert exit_code == 0, stderr.getvalue()
    payload = cast(dict[str, object], json.loads(stdout.getvalue()))
    beliefs = cast(list[object], payload["beliefs"])
    kinds = {str(cast(dict[str, object], belief)["kind"]) for belief in beliefs}
    assert {"expert_prior", "graph_directive"} <= kinds

    summary = cast(dict[str, object], payload["canonicalization_summary"])
    records = cast(list[object], summary["records"])
    assert summary["mode"] == "llm"
    assert summary["model_name"] == "advanced-skill-stub"
    assert all(
        cast(dict[str, object], record)["status"] == "resolved" for record in records
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        preview_exit_code = main(
            [
                "beliefs",
                "preview",
                "--ideas-json",
                ideas_json,
                "--era-id",
                "era_log_parser_v1",
                "--canonicalization-mode",
                "llm",
                "--canonicalization-model",
                "tests.fixtures.advanced_skill_canonicalizer",
            ]
        )
    assert preview_exit_code == 0, stderr.getvalue()
    preview_payload = cast(dict[str, object], json.loads(stdout.getvalue()))
    assert cast(list[object], preview_payload["belief_previews"])

    skill_rendered = (
        root / "skills" / "advanced-belief-author" / "SKILL.md"
    ).read_text(encoding="utf-8")
    assert "autoclanker adapter surface" in skill_rendered
    assert "AUTOCLANKER_CANONICALIZATION_MODEL=anthropic" in skill_rendered
    assert "expert_prior" in skill_rendered
    assert "graph_directive" in skill_rendered


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("AUTOCLANKER_ENABLE_LLM_LIVE") != "1"
    or not (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AUTOCLANKER_ANTHROPIC_API_KEY")
    ),
    reason="Requires opt-in billed LLM live execution and an Anthropic API key.",
)
@covers("M7-LIVE-001")
def test_advanced_belief_author_skill_live_provider_path_emits_advanced_beliefs() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    stdout = io.StringIO()
    stderr = io.StringIO()

    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(
            [
                "beliefs",
                "canonicalize-ideas",
                "--input",
                str(root / "tests" / "fixtures" / "live_advanced_skill_ideas.json"),
                "--era-id",
                "era_log_parser_v1",
                "--canonicalization-mode",
                "llm",
                "--canonicalization-model",
                "anthropic",
            ]
        )
    assert exit_code == 0, stderr.getvalue()
    payload = cast(dict[str, object], json.loads(stdout.getvalue()))
    beliefs = cast(list[object], payload["beliefs"])
    kinds = {str(cast(dict[str, object], belief)["kind"]) for belief in beliefs}
    assert {"expert_prior", "graph_directive"} <= kinds

    summary = cast(dict[str, object], payload["canonicalization_summary"])
    records = cast(list[object], summary["records"])
    assert summary["mode"] == "llm"
    assert str(summary["model_name"]).startswith("anthropic:")
    assert all(
        cast(dict[str, object], record)["status"] == "resolved" for record in records
    )


@covers("M7-010")
def test_advanced_belief_author_skill_prefers_json_for_machine_authored_outputs() -> (
    None
):
    root = Path(__file__).resolve().parents[1]
    skill_path = root / "skills" / "advanced-belief-author" / "SKILL.md"
    belief_reference_path = root / "docs" / "BELIEF_INPUT_REFERENCE.md"
    readme_path = root / "README.md"

    skill_rendered = skill_path.read_text(encoding="utf-8")
    belief_reference_rendered = belief_reference_path.read_text(encoding="utf-8")
    readme_rendered = readme_path.read_text(encoding="utf-8")

    assert (
        "prefer a full JSON belief batch for machine-authored outputs" in skill_rendered
    )
    assert "rough ideas" in belief_reference_rendered
    assert "machine-authored JSON belief batch by default" in belief_reference_rendered
    assert "Advanced Beliefs" in readme_rendered
    assert "advanced JSON belief batch" in readme_rendered
