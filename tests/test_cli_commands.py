from __future__ import annotations

import io
import json
import sys

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest

from autoclanker.cli import EXIT_VALIDATION_ERROR, main
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _read_stdout(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


@contextmanager
def _patched_stdin(text: str) -> Generator[None]:
    original = sys.stdin
    sys.stdin = io.StringIO(text)
    try:
        yield
    finally:
        sys.stdin = original


@covers("M4-004")
def test_cli_beliefs_eval_and_adapter_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "beliefs",
                "validate",
                "--input",
                str(ROOT / "examples/human_beliefs/basic_session.json"),
            ]
        )
        == 0
    )
    validate_payload = _read_stdout(capsys)
    assert validate_payload["belief_count"] == 4

    assert (
        main(
            [
                "beliefs",
                "preview",
                "--input",
                str(ROOT / "examples/human_beliefs/expert_session.yaml"),
            ]
        )
        == 0
    )
    preview_payload = _read_stdout(capsys)
    assert preview_payload["belief_previews"]

    assert (
        main(
            [
                "beliefs",
                "compile",
                "--input",
                str(ROOT / "examples/human_beliefs/expert_session.yaml"),
            ]
        )
        == 0
    )
    compile_payload = _read_stdout(capsys)
    assert compile_payload["pair_priors"]

    assert (
        main(
            [
                "beliefs",
                "validate",
                "--input",
                str(ROOT / "examples" / "idea_inputs" / "bayes_quickstart.json"),
            ]
        )
        == 0
    )
    idea_validate_payload = _read_stdout(capsys)
    assert idea_validate_payload["belief_count"] == 4

    assert (
        main(
            [
                "beliefs",
                "expand-ideas",
                "--input",
                str(ROOT / "examples" / "idea_inputs" / "bayes_quickstart.json"),
            ]
        )
        == 0
    )
    expand_payload = _read_stdout(capsys)
    assert [
        str(cast(dict[str, object], item)["kind"])
        for item in cast(list[object], expand_payload["beliefs"])
    ] == ["idea", "relation", "idea", "proposal"]

    assert (
        main(
            [
                "eval",
                "validate",
                "--input",
                str(ROOT / "examples/eval_results/valid_eval_result.json"),
            ]
        )
        == 0
    )
    eval_payload = _read_stdout(capsys)
    assert eval_payload["status"] == "valid"

    assert main(["adapter", "list"]) == 0
    list_payload = _read_stdout(capsys)
    assert "fixture" in cast(list[object], list_payload["known_kinds"])

    assert (
        main(
            [
                "adapter",
                "validate-config",
                "--input",
                str(ROOT / "examples/adapters/fixture.yaml"),
            ]
        )
        == 0
    )
    validate_config_payload = _read_stdout(capsys)
    assert validate_config_payload["kind"] == "fixture"

    assert (
        main(
            [
                "adapter",
                "probe",
                "--input",
                str(ROOT / "examples/adapters/fixture.yaml"),
            ]
        )
        == 0
    )
    probe_payload = _read_stdout(capsys)
    assert probe_payload["available"] is True

    assert main(["adapter", "registry"]) == 0
    registry_payload = _read_stdout(capsys)
    assert registry_payload["kind"] == "fixture"
    fixture_registry = cast(dict[str, object], registry_payload["registry"])
    assert "parser.matcher" in fixture_registry
    matcher_definition = cast(dict[str, object], fixture_registry["parser.matcher"])
    assert matcher_definition["description"]


@covers("M1-001", "M1-003", "M4-004", "M7-005")
def test_cli_can_expand_beginner_ideas_from_stdin_with_cli_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    stdin_payload = """ideas:
  - idea: Compiled regex matching probably helps this parser on repeated log formats.
    confidence: 2
"""

    with _patched_stdin(stdin_payload):
        assert (
            main(
                [
                    "beliefs",
                    "expand-ideas",
                    "--input",
                    "-",
                    "--era-id",
                    "era_demo_v1",
                    "--session-id",
                    "demo_session",
                ]
            )
            == 0
        )
    expand_payload = _read_stdout(capsys)
    session_context = cast(dict[str, object], expand_payload["session_context"])
    assert session_context["era_id"] == "era_demo_v1"
    assert session_context["session_id"] == "demo_session"
    beliefs = cast(list[object], expand_payload["beliefs"])
    first_belief = cast(dict[str, object], beliefs[0])
    assert first_belief["kind"] == "idea"
    assert first_belief["id"] == "idea_001"
    gene = cast(dict[str, object], first_belief["gene"])
    assert gene["gene_id"] == "parser.matcher"
    assert gene["state_id"] == "matcher_compiled"


@covers("M1-001", "M1-003", "M4-004", "M7-005")
def test_cli_can_expand_beginner_ideas_from_inline_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "beliefs",
                "expand-ideas",
                "--ideas-json",
                '{"idea":"Compiled regex matching probably helps this parser on repeated log formats.","confidence":2}',
                "--era-id",
                "era_demo_v1",
            ]
        )
        == 0
    )
    expand_payload = _read_stdout(capsys)
    beliefs = cast(list[object], expand_payload["beliefs"])
    first_belief = cast(dict[str, object], beliefs[0])
    assert first_belief["kind"] == "idea"
    assert first_belief["id"] == "idea_001"
    gene = cast(dict[str, object], first_belief["gene"])
    assert gene["gene_id"] == "parser.matcher"
    assert gene["state_id"] == "matcher_compiled"


@covers("M1-001", "M1-003", "M4-004", "M7-005")
def test_cli_can_expand_string_only_beginner_ideas_from_inline_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "beliefs",
                "expand-ideas",
                "--ideas-json",
                '["Compiled regex matching probably helps this parser on repeated log formats."]',
                "--era-id",
                "era_demo_v1",
            ]
        )
        == 0
    )
    expand_payload = _read_stdout(capsys)
    beliefs = cast(list[object], expand_payload["beliefs"])
    first_belief = cast(dict[str, object], beliefs[0])
    assert first_belief["kind"] == "idea"
    assert first_belief["confidence_level"] == 2
    gene = cast(dict[str, object], first_belief["gene"])
    assert gene["gene_id"] == "parser.matcher"
    assert gene["state_id"] == "matcher_compiled"


@covers("M1-001", "M1-003", "M4-004", "M7-005")
def test_cli_can_canonicalize_beginner_ideas_with_alias_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "beliefs",
                "canonicalize-ideas",
                "--ideas-json",
                '{"idea":"Streaming summary output sounds useful here.","confidence":1}',
                "--era-id",
                "era_demo_v1",
            ]
        )
        == 0
    )
    payload = _read_stdout(capsys)
    beliefs = cast(list[object], payload["beliefs"])
    first_belief = cast(dict[str, object], beliefs[0])
    assert first_belief["kind"] == "idea"
    gene = cast(dict[str, object], first_belief["gene"])
    assert gene["gene_id"] == "emit.summary"
    assert gene["state_id"] == "summary_streaming"


@covers("M7-007")
def test_canonicalize_ideas_and_expand_ideas_are_equivalent_for_beginner_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "--ideas-json",
        '{"idea":"Compiled regex matching probably helps this parser on repeated log formats.","confidence":2}',
        "--era-id",
        "era_demo_v1",
    ]

    assert main(["beliefs", "canonicalize-ideas", *argv]) == 0
    canonicalized = _read_stdout(capsys)
    assert main(["beliefs", "expand-ideas", *argv]) == 0
    expanded = _read_stdout(capsys)

    assert canonicalized == expanded


@covers("M1-001", "M1-003", "M4-004", "M7-005")
def test_cli_preserves_unresolved_high_level_ideas_as_proposals_with_suggestions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "beliefs",
                "expand-ideas",
                "--ideas-json",
                '{"idea":"Try a moonbeam dragon refactor with kaleidoscope anchors.","confidence":2}',
                "--era-id",
                "era_demo_v1",
            ]
        )
        == 0
    )
    payload = _read_stdout(capsys)
    beliefs = cast(list[object], payload["beliefs"])
    first_belief = cast(dict[str, object], beliefs[0])
    assert first_belief["kind"] == "proposal"
    context = cast(dict[str, object], first_belief["context"])
    metadata = cast(dict[str, object], context["metadata"])
    assert metadata["canonicalization_mode"] == "needs_review"
    suggested_options = metadata.get("suggested_options")
    assert suggested_options is None or isinstance(suggested_options, list)


@covers("M1-002", "M6-002")
def test_cli_emits_validation_error_for_invalid_belief_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_path = tmp_path / "invalid_beliefs.json"
    invalid_path.write_text('{"beliefs": []}', encoding="utf-8")

    exit_code = main(
        [
            "beliefs",
            "validate",
            "--input",
            str(invalid_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_VALIDATION_ERROR
    assert '"ok": false' in captured.err


@covers("M6-004")
def test_cli_supports_output_after_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "validate.json"
    root_output_path = tmp_path / "validate-root.json"

    exit_code = main(
        [
            "beliefs",
            "validate",
            "--input",
            str(ROOT / "examples/human_beliefs/basic_session.json"),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    stdout_payload = _read_stdout(capsys)
    file_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert file_payload == stdout_payload
    assert stdout_payload["belief_count"] == 4

    exit_code = main(
        [
            "--output",
            str(root_output_path),
            "beliefs",
            "validate",
            "--input",
            str(ROOT / "examples/human_beliefs/basic_session.json"),
        ]
    )

    assert exit_code == 0
    root_stdout_payload = _read_stdout(capsys)
    root_file_payload = json.loads(root_output_path.read_text(encoding="utf-8"))
    assert root_file_payload == root_stdout_payload
    assert root_stdout_payload["belief_count"] == 4


@covers("M8-002")
def test_cli_generates_portable_issue_seed_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "issue-seed"
    seed_path = ROOT / "examples" / "issue_seeder" / "pipeline_optimization.seed.json"

    assert (
        main(
            [
                "issue-seed",
                "generate",
                "--input",
                str(seed_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    payload = _read_stdout(capsys)
    artifacts = cast(dict[str, object], payload["artifacts"])

    for name in (
        "issue_body.md",
        "autoclanker.ideas.json",
        "artifact-manifest.json",
        "run-contract.json",
        "lane-ledger.md",
        "pi.prompt.txt",
        "headless-command.sh",
        "host-adapter-contract.md",
    ):
        assert name in artifacts
        assert (output_dir / name).exists()

    issue_body = (output_dir / "issue_body.md").read_text(encoding="utf-8")
    ideas = json.loads((output_dir / "autoclanker.ideas.json").read_text())
    run_contract = json.loads((output_dir / "run-contract.json").read_text())
    manifest = json.loads((output_dir / "artifact-manifest.json").read_text())
    pi_prompt = (output_dir / "pi.prompt.txt").read_text(encoding="utf-8")
    headless = (output_dir / "headless-command.sh").read_text(encoding="utf-8")
    host_contract = (output_dir / "host-adapter-contract.md").read_text(
        encoding="utf-8"
    )

    assert "## Start Here" in issue_body
    assert "bigbets:idea-family" in issue_body
    assert "evidence artifacts as search inputs" in issue_body
    assert ideas["session_context"]["era_id"] == "era_pipeline_optimization_v1"
    assert len(cast(list[object], ideas["ideas"])) == 3
    assert run_contract["schema_version"] == "autoclanker.run-contract.v1"
    assert "fixed_eval_surface_preserved" in run_contract["acceptance_gates"]
    assert run_contract["canonicalization_mode"] == "deterministic"
    assert "constraints" in run_contract
    assert "evidence_intake_checklist" in run_contract
    assert "required_session_outputs" in run_contract
    assert manifest["schema_version"] == "autoclanker.issue-seed.v1"
    assert manifest["big_bet"] == "minimum_cost_pipeline"
    assert "headless-command.sh" in manifest["expected_workspace_files"]
    assert "/autoclanker run" in pi_prompt
    assert "handoffPrompt" in pi_prompt
    assert "autoclanker session init" in headless
    assert "pi-autoclanker command run" in headless
    assert "--canonicalization-mode deterministic" in headless
    assert "--canonicalization-mode hybrid" not in headless
    assert "--adapter-config" not in headless
    assert (output_dir / "headless-command.sh").stat().st_mode & 0o111
    assert "provider keys" in host_contract.lower()

    assert (
        main(
            [
                "beliefs",
                "validate",
                "--input",
                str(output_dir / "autoclanker.ideas.json"),
            ]
        )
        == 0
    )
    validate_payload = _read_stdout(capsys)
    assert isinstance(validate_payload["belief_count"], int)
    assert validate_payload["belief_count"] >= 1


@covers("M8-002")
def test_issue_seed_can_opt_into_adapter_and_model_commands(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed_path = tmp_path / "adapter.seed.json"
    output_dir = tmp_path / "adapter-seed"
    seed_path.write_text(
        json.dumps(
            {
                "title": "Adapter seed",
                "goal": "Exercise optional hosted adapter seams.",
                "target_repo": "owner/repo",
                "ideas": ["Use host canonicalization for unresolved lanes."],
                "adapter_config_path": "adapter.local.yaml",
                "canonicalization_mode": "hybrid",
                "canonicalization_model": "stub",
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "issue-seed",
                "generate",
                "--input",
                str(seed_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    _read_stdout(capsys)
    run_contract = json.loads((output_dir / "run-contract.json").read_text())
    headless = (output_dir / "headless-command.sh").read_text(encoding="utf-8")

    assert run_contract["canonicalization_mode"] == "hybrid"
    assert run_contract["canonicalization_model"] == "stub"
    assert "--canonicalization-mode hybrid" in headless
    assert "--canonicalization-model stub" in headless
    assert "  --adapter-config adapter.local.yaml > autoclanker.init.json" in headless
    assert "#   --adapter-config adapter.local.yaml" in headless


@covers("M8-002")
def test_cli_generates_generic_request_rendering_issue_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "request-rendering-seed"
    seed_path = ROOT / "examples" / "issue_seeder" / "request_rendering.seed.json"

    assert (
        main(
            [
                "issue-seed",
                "generate",
                "--input",
                str(seed_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    _read_stdout(capsys)
    issue_body = (output_dir / "issue_body.md").read_text(encoding="utf-8")
    run_contract = json.loads((output_dir / "run-contract.json").read_text())

    assert "request rendering" in issue_body.lower()
    assert "component" in issue_body
    assert "response output and correctness" in issue_body
    forbidden = {
        "".join(
            chr(code) for code in (83, 116, 111, 114, 101, 102, 114, 111, 110, 116)
        ),
        "".join(chr(code) for code in (76, 105, 113, 117, 105, 100)),
    }
    assert not any(term in issue_body for term in forbidden)
    assert run_contract["canonicalization_mode"] == "deterministic"

    assert (
        main(
            [
                "beliefs",
                "validate",
                "--input",
                str(output_dir / "autoclanker.ideas.json"),
            ]
        )
        == 0
    )
    validate_payload = _read_stdout(capsys)
    assert isinstance(validate_payload["belief_count"], int)
    assert validate_payload["belief_count"] >= 1


@covers("M8-002")
def test_issue_seed_rejects_unsafe_or_ambiguous_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_seed = tmp_path / "invalid.seed.json"
    invalid_seed.write_text(
        json.dumps(
            {
                "title": "Invalid seed",
                "goal": "Show validation.",
                "target_repo": "owner/repo",
                "run_intensity": "forever",
                "artifacts": [""],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "issue-seed",
                "generate",
                "--input",
                str(invalid_seed),
            ]
        )
        == EXIT_VALIDATION_ERROR
    )
    err_payload = json.loads(capsys.readouterr().err)
    assert "run_intensity" in err_payload["error"]
