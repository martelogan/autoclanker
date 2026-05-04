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
