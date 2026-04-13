from __future__ import annotations

import importlib
import inspect
import os
import pkgutil

from collections import defaultdict
from pathlib import Path
from typing import cast

import pytest

import tests

from autoclanker.bayes_layer import load_serialized_payload, validate_adapter_config
from autoclanker.bayes_layer.adapters.external import resolved_repo_path
from tests.compliance import covers, load_requirement_matrix

_DOC_SYNC_ONLY_TESTS = {
    "tests.test_advanced_belief_author_skill::test_advanced_belief_author_skill_mentions_provider_backed_authoring",
    "tests.test_compliance_matrix::test_human_readable_compliance_mirror_stays_in_sync",
    "tests.test_compliance_matrix::test_primary_docs_focus_on_current_library_surface",
    "tests.test_live_exercise_examples::test_belief_input_reference_documents_minimum_files_and_bounds",
}
_REQUIRES_BEHAVIORAL_EVIDENCE = {
    "M0-002",
    "M1-005",
    "M1-LIVE-001",
    "M4-004",
    "M5-LIVE-001",
    "M5-LIVE-002",
    "M6-LIVE-001",
    "M7-LIVE-001",
    "M7-005",
    "M7-006",
    "M7-007",
}


def _mark_names(func: object) -> set[str]:
    marks_raw = getattr(func, "pytestmark", ())
    marks: list[object]
    if isinstance(marks_raw, tuple | list):
        sequence = cast(tuple[object, ...] | list[object], marks_raw)
        marks = [mark for mark in sequence]
    else:
        marks = [cast(object, marks_raw)]
    names: set[str] = set()
    for mark in marks:
        name = getattr(mark, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


@covers("M7-001")
def test_compliance_matrix_is_fully_covered() -> None:
    matrix = load_requirement_matrix()
    coverage: dict[str, set[str]] = defaultdict(set)
    required_lane_coverage: dict[str, set[str]] = defaultdict(set)
    live_lane_coverage: dict[str, set[str]] = defaultdict(set)
    invalid_required_tests: list[str] = []

    for module_info in pkgutil.iter_modules(tests.__path__, prefix="tests."):
        if not module_info.name.startswith("tests.test_"):
            continue
        module = importlib.import_module(module_info.name)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            requirement_ids = getattr(func, "__autoclanker_requirement_ids__", ())
            if not requirement_ids:
                continue
            mark_names = _mark_names(func)
            node_id = f"{module_info.name}::{name}"
            for requirement_id in requirement_ids:
                coverage[requirement_id].add(node_id)
                if "live" in mark_names or "upstream_live" in mark_names:
                    live_lane_coverage[requirement_id].add(node_id)
                elif (
                    "skip" not in mark_names
                    and "skipif" not in mark_names
                    and "xfail" not in mark_names
                ):
                    required_lane_coverage[requirement_id].add(node_id)
                elif requirement_id.startswith("M") and not requirement_id.startswith(
                    "M5-LIVE"
                ):
                    invalid_required_tests.append(node_id)

    assert not invalid_required_tests
    assert all(entry.status != "todo" for entry in matrix)

    missing = [
        entry.requirement_id
        for entry in matrix
        if not coverage.get(entry.requirement_id)
    ]
    assert not missing

    missing_required_lane = [
        entry.requirement_id
        for entry in matrix
        if entry.gate == "required"
        and not required_lane_coverage.get(entry.requirement_id)
    ]
    assert not missing_required_lane

    missing_live_lane = [
        entry.requirement_id
        for entry in matrix
        if entry.gate == "live" and not live_lane_coverage.get(entry.requirement_id)
    ]
    assert not missing_live_lane

    missing_behavioral = [
        entry.requirement_id
        for entry in matrix
        if entry.requirement_id in _REQUIRES_BEHAVIORAL_EVIDENCE
        and not any(
            node_id not in _DOC_SYNC_ONLY_TESTS
            for node_id in coverage.get(entry.requirement_id, set())
        )
    ]
    assert not missing_behavioral


@covers("M7-001")
def test_human_readable_compliance_mirror_stays_in_sync() -> None:
    matrix = load_requirement_matrix()
    mirror_path = Path(__file__).resolve().parents[1] / "docs" / "COMPLIANCE_MATRIX.md"
    rendered = mirror_path.read_text(encoding="utf-8")
    for entry in matrix:
        assert entry.requirement_id in rendered
        assert entry.gate in rendered
        assert entry.description in rendered


@covers("M7-002")
def test_example_adapter_paths_match_runtime_resolution_rules() -> None:
    root = Path(__file__).resolve().parents[1]
    examples = {
        "autoresearch.local.yaml": root / ".local" / "real-upstreams" / "autoresearch",
        "cevolve.local.yaml": root / ".local" / "real-upstreams" / "cevolve",
    }
    docs_rendered = (root / "docs" / "INTEGRATIONS.md").read_text(encoding="utf-8")
    for filename, expected_repo_path in examples.items():
        config_path = root / "examples" / "adapters" / filename
        config = validate_adapter_config(
            load_serialized_payload(config_path),
            base_dir=config_path.parent,
        )
        assert resolved_repo_path(config) == expected_repo_path
        assert (
            f"repo_path: ../../.local/real-upstreams/{expected_repo_path.name}"
            in docs_rendered
        )
    assert "mode: auto" in docs_rendered
    assert "python_module" in docs_rendered
    assert "command" in docs_rendered


@covers("M7-008")
def test_docs_distinguish_upstream_live_from_billed_model_live() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    reference = (root / "docs" / "BELIEF_INPUT_REFERENCE.md").read_text(
        encoding="utf-8"
    )
    exercises = (root / "docs" / "LIVE_EXERCISES.md").read_text(encoding="utf-8")

    assert "./bin/dev test-upstream-live" in readme
    assert "AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live" in readme
    assert "non-billed" in readme
    assert "billed real-model-provider lane" in readme
    assert "AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live" in reference
    assert "real-upstream contract smoke test" in exercises


@covers("M7-009")
def test_primary_docs_focus_on_current_library_surface() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    contributing = (root / ".github" / "CONTRIBUTING.md").read_text(encoding="utf-8")
    style = (root / "docs" / "STYLE.md").read_text(encoding="utf-8")

    assert "BAYES_HANDOFF.md" not in readme
    assert "CODEX_TASK.md" not in readme
    assert "PLAN.md" not in readme
    assert "PI_EXTENSION_FUTURE.md" not in readme
    assert "codex-autonomous" not in readme
    assert "starter-python-project" not in readme
    assert ".github/CONTRIBUTING.md" in readme
    assert "docs/STYLE.md" in readme
    assert "Development setup" in contributing
    assert "Validation" in contributing
    assert "Style Guide" in style
    assert "pyright" in style


@pytest.mark.upstream_live
@covers("M5-LIVE-001", "M5-LIVE-002")
def test_upstream_live_requirements_run_inside_the_actual_upstream_live_lane() -> None:
    assert os.environ.get("AUTOCLANKER_TEST_LANE") == "upstream_live"
    assert (
        os.environ.get("AUTOCLANKER_TEST_LANE_KIND") == "real_upstream_contract_smoke"
    )


@pytest.mark.live
@covers("M1-LIVE-001", "M6-LIVE-001")
def test_billed_live_requirements_run_inside_the_actual_billed_live_lane() -> None:
    assert os.environ.get("AUTOCLANKER_TEST_LANE") == "live"
    assert os.environ.get("AUTOCLANKER_TEST_LANE_KIND") == "billed_model_provider"
    assert os.environ.get("AUTOCLANKER_TEST_LANE_PROVIDER") == "anthropic"
    assert os.environ.get("AUTOCLANKER_ENABLE_LLM_LIVE") == "1"
    assert bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AUTOCLANKER_ANTHROPIC_API_KEY")
    )
