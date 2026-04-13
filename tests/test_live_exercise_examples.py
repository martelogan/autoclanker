from __future__ import annotations

import json

from pathlib import Path
from typing import cast

import pytest

from autoclanker.bayes_layer import ingest_human_beliefs, load_serialized_payload
from autoclanker.cli import main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


@covers("M6-002", "M7-002", "M7-005")
def test_live_exercise_example_payloads_and_docs_are_present() -> None:
    exercise_root = ROOT / "examples" / "live_exercises"
    idea_root = ROOT / "examples" / "idea_inputs"
    examples_root_index = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    docs_rendered = (ROOT / "docs" / "LIVE_EXERCISES.md").read_text(encoding="utf-8")
    example_index = (exercise_root / "README.md").read_text(encoding="utf-8")
    idea_index = (idea_root / "README.md").read_text(encoding="utf-8")
    skill_rendered = (
        ROOT / "skills" / "advanced-belief-author" / "SKILL.md"
    ).read_text(encoding="utf-8")

    for name in (
        "autoresearch_simple",
        "cevolve_synergy",
        "bayes_quickstart",
        "bayes_complex",
    ):
        readme = exercise_root / name / "README.md"
        assert readme.exists()
        assert name in docs_rendered
        assert readme.read_text(encoding="utf-8").strip()
        rendered = readme.read_text(encoding="utf-8")
        assert "Minimum required files" in rendered
        assert "Allowed idea inputs" in rendered

    for path in (
        exercise_root / "autoresearch_simple" / "adapter.local.yaml",
        exercise_root / "cevolve_synergy" / "adapter.local.yaml",
    ):
        assert path.exists()
        rendered = path.read_text(encoding="utf-8")
        assert "adapter_module" in rendered

    for path in (
        exercise_root / "autoresearch_simple" / "beliefs.yaml",
        exercise_root / "cevolve_synergy" / "beliefs.yaml",
        exercise_root / "bayes_quickstart" / "beliefs.yaml",
        exercise_root / "bayes_complex" / "beliefs.yaml",
    ):
        beliefs = ingest_human_beliefs(load_serialized_payload(path))
        assert beliefs.beliefs

    for path in (
        exercise_root / "autoresearch_simple" / "expected_outcome.json",
        exercise_root / "cevolve_synergy" / "expected_outcome.json",
        exercise_root / "bayes_quickstart" / "expected_outcome.json",
        exercise_root / "bayes_complex" / "expected_outcome.json",
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path.parent.name == "bayes_complex":
            assert "cold_start" in payload
            assert "post_observations" in payload
        elif path.parent.name == "bayes_quickstart":
            assert "preview" in payload
            assert "cold_start" in payload
        else:
            assert "baseline" in payload
            assert "expectation" in payload

    quickstart_app = exercise_root / "bayes_quickstart" / "app.py"
    assert quickstart_app.exists()
    for path in (
        idea_root / "minimal.json",
        idea_root / "bayes_quickstart.json",
        idea_root / "autoresearch_simple.json",
        idea_root / "cevolve_synergy.json",
        idea_root / "minimal.yaml",
        idea_root / "bayes_quickstart.yaml",
        idea_root / "autoresearch_simple.yaml",
        idea_root / "cevolve_synergy.yaml",
    ):
        assert path.exists()
        rendered = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            json.loads(rendered)
            if path.name == "minimal.json":
                assert '"confidence"' in rendered
            else:
                assert '"session_context"' in rendered
                assert '"ideas"' in rendered
        elif path.name == "minimal.yaml":
            assert "canonicalize-ideas" in rendered
            assert "--era-id" in rendered
        else:
            assert "beliefs expand-ideas" in rendered
    bayes_candidates = json.loads(
        (exercise_root / "bayes_complex" / "candidates.json").read_text(
            encoding="utf-8"
        )
    )
    assert "candidates" in bayes_candidates
    assert len(bayes_candidates["candidates"]) >= 5
    assert "generate_bayes_complex_evals.py" in docs_rendered
    assert "replay_bayes_quickstart.py" in docs_rendered
    assert "replay_ideas_demo.py" in docs_rendered
    assert "adapter.local.yaml" in docs_rendered
    assert "replay_backing_exercise.py" in docs_rendered
    assert "BELIEF_INPUT_REFERENCE.md" in docs_rendered
    assert "replay_ideas_demo.py" in example_index
    assert "bayes_quickstart" in examples_root_index
    assert "docs/toy_examples/" in examples_root_index
    assert "examples/toy_examples" not in examples_root_index
    assert "bayes_quickstart.yaml" in idea_index
    assert "bayes_quickstart.json" in idea_index
    assert "autoresearch_simple.json" in idea_index
    assert "cevolve_synergy.json" in idea_index
    assert "minimal.json" in idea_index
    assert "beliefs expand-ideas" in skill_rendered
    assert "adapter registry" in skill_rendered


@covers("M7-005")
def test_belief_input_reference_documents_minimum_files_and_bounds() -> None:
    rendered = (ROOT / "docs" / "BELIEF_INPUT_REFERENCE.md").read_text(encoding="utf-8")

    for token in (
        "Smallest runnable files",
        "session_context.era_id",
        "confidence_level",
        "minimal.json",
        "--ideas-json",
        "option",
        "canonicalize-ideas",
        "minimal.yaml",
        "--input -",
        "effect_strength",
        "suggested_scope",
        "constraint_type",
        "graph_directive",
        "matcher_compiled",
        "threshold_32",
        "partition_hoare",
        "replay_ideas_demo.py --exercise bayes_quickstart",
    ):
        assert token in rendered


@covers("M7-005")
def test_bayes_quickstart_live_exercise_files_drive_real_cli_flow(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session_root = tmp_path / "sessions"
    beliefs_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "beliefs.yaml"
    )
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )

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
    init_payload = cast(dict[str, object], json.loads(capsys.readouterr().out))
    session_id = str(init_payload["session_id"])
    preview_digest = str(init_payload["preview_digest"])

    assert (
        main(
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
        == 0
    )
    apply_payload = cast(dict[str, object], json.loads(capsys.readouterr().out))
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
    suggest_payload = cast(dict[str, object], json.loads(capsys.readouterr().out))
    ranked = cast(list[object], suggest_payload["ranked_candidates"])
    top_candidate = cast(dict[str, object], ranked[0])
    assert top_candidate["candidate_id"] == "cand_c_compiled_context_pair"
