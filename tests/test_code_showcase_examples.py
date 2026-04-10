from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path
from typing import cast

from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{path} must contain a JSON object.")
    return cast(dict[str, object], payload)


def _numeric(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"Expected numeric field {key!r}.")
    return float(value)


def _string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise AssertionError(f"Expected string field {key!r}.")
    return value


@covers("M7-004")
def test_code_showcase_examples_are_documented_and_runnable() -> None:
    showcase_root = ROOT / "docs" / "toy_examples"
    docs_rendered = (ROOT / "docs" / "TOY_EXAMPLES.md").read_text(encoding="utf-8")
    live_docs = (ROOT / "docs" / "LIVE_EXERCISES.md").read_text(encoding="utf-8")
    summary_process = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "showcase" / "run_toy_examples.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(summary_process.stdout)
    if not isinstance(summary, dict):
        raise AssertionError("Toy-mirror summary must be a JSON object.")

    for name in (
        "autoresearch_command_autocomplete",
        "cevolve_sort_partition",
        "bayes_pair_feature_trainer",
    ):
        readme = showcase_root / name / "README.md"
        expected = showcase_root / name / "expected_outcome.json"
        app_path = showcase_root / name / "app.py"
        benchmark_path = showcase_root / name / "benchmark.py"
        variants_dir = showcase_root / name / "variants"
        assert readme.exists()
        assert expected.exists()
        assert app_path.exists()
        assert benchmark_path.exists()
        assert variants_dir.exists()
        assert name in docs_rendered
        assert name in live_docs
        assert readme.read_text(encoding="utf-8").strip()
        assert _read_json(expected)
        assert "replay_backing_exercise.py" in readme.read_text(encoding="utf-8")

    assert "replay_backing_exercise.py" in docs_rendered

    autoresearch = cast(dict[str, object], summary["autoresearch_command_autocomplete"])
    cevolve = cast(dict[str, object], summary["cevolve_sort_partition"])
    bayes = cast(dict[str, object], summary["bayes_pair_feature_trainer"])

    autoresearch_about = cast(dict[str, object], autoresearch["about"])
    cevolve_about = cast(dict[str, object], cevolve["about"])
    bayes_about = cast(dict[str, object], bayes["about"])
    assert "backing_replay_command" in autoresearch_about
    assert "backing_replay_command" in cevolve_about
    assert "backing_replay_command" in bayes_about
    assert "primary_files" in autoresearch_about
    assert "primary_files" in cevolve_about
    assert "primary_files" in bayes_about
    assert "manual_walkthrough" in autoresearch_about
    assert "manual_walkthrough" in cevolve_about
    assert "manual_walkthrough" in bayes_about

    autoresearch_baseline = cast(dict[str, object], autoresearch["baseline"])
    autoresearch_optimized = cast(dict[str, object], autoresearch["optimized"])
    autoresearch_failure = cast(dict[str, object], autoresearch["failure_variant"])
    autoresearch_comparison = cast(dict[str, object], autoresearch["comparison"])
    autoresearch_expected = _read_json(
        showcase_root / "autoresearch_command_autocomplete" / "expected_outcome.json"
    )
    autoresearch_expectation = cast(
        dict[str, object], autoresearch_expected["expectation"]
    )
    assert _numeric(autoresearch_baseline, "val_bpb") > _numeric(
        autoresearch_optimized, "val_bpb"
    )
    assert _numeric(autoresearch_comparison, "val_bpb_improvement") >= _numeric(
        autoresearch_expectation, "val_bpb_improvement_at_least"
    )
    assert _string(autoresearch_failure, "status") == _string(
        autoresearch_expectation, "failure_status"
    )
    assert _string(autoresearch_baseline, "demo_role") == "readable_snapshot_only"
    assert (
        _string(autoresearch_baseline, "demo_layout")
        == "app.py + benchmark.py + variants/"
    )
    assert "backing_replay_command" in autoresearch_baseline
    assert "what_is_being_optimized" in autoresearch_baseline
    assert "manual_app_command" in autoresearch_baseline
    assert "autoclanker_relationship" in autoresearch_baseline
    assert "llm_note" in autoresearch_baseline

    cevolve_baseline = cast(dict[str, object], cevolve["baseline"])
    cevolve_optimized = cast(dict[str, object], cevolve["optimized"])
    cevolve_single_threshold = cast(dict[str, object], cevolve["single_threshold"])
    cevolve_single_partition = cast(dict[str, object], cevolve["single_partition"])
    cevolve_comparison = cast(dict[str, object], cevolve["comparison"])
    cevolve_expected = _read_json(
        showcase_root / "cevolve_sort_partition" / "expected_outcome.json"
    )
    cevolve_expectation = cast(dict[str, object], cevolve_expected["expectation"])
    assert _numeric(cevolve_baseline, "time_ms") > _numeric(
        cevolve_optimized, "time_ms"
    )
    assert _numeric(cevolve_single_threshold, "time_ms") > _numeric(
        cevolve_optimized, "time_ms"
    )
    assert _numeric(cevolve_single_partition, "time_ms") > _numeric(
        cevolve_optimized, "time_ms"
    )
    assert _numeric(cevolve_comparison, "time_ms_improvement") >= _numeric(
        cevolve_expectation, "time_ms_improvement_at_least"
    )
    assert _numeric(cevolve_comparison, "synergy_margin_vs_best_single") >= _numeric(
        cevolve_expectation, "synergy_margin_at_least"
    )
    assert (
        _string(cevolve_baseline, "demo_layout") == "app.py + benchmark.py + variants/"
    )
    assert cevolve_baseline["variant"] == "baseline"
    assert "backing_replay_command" in cevolve_baseline
    assert "what_is_being_optimized" in cevolve_baseline
    assert "manual_app_command" in cevolve_baseline
    assert "autoclanker_relationship" in cevolve_baseline
    assert "llm_note" in cevolve_baseline

    bayes_baseline = cast(dict[str, object], bayes["baseline"])
    bayes_local_best = cast(dict[str, object], bayes["local_observed_best"])
    bayes_belief_guided = cast(dict[str, object], bayes["belief_guided"])
    bayes_risky = cast(dict[str, object], bayes["risky_oom"])
    bayes_comparison = cast(dict[str, object], bayes["comparison"])
    bayes_expected = _read_json(
        showcase_root / "bayes_pair_feature_trainer" / "expected_outcome.json"
    )
    bayes_expectation = cast(dict[str, object], bayes_expected["expectation"])
    assert _numeric(bayes_local_best, "utility") > _numeric(bayes_baseline, "utility")
    assert _numeric(bayes_belief_guided, "utility") > _numeric(
        bayes_local_best, "utility"
    )
    assert _numeric(
        bayes_comparison, "belief_guided_utility_gain_over_local_best"
    ) >= _numeric(bayes_expectation, "belief_guided_utility_gain_over_local_best")
    assert _string(bayes_risky, "status") == _string(bayes_expectation, "risky_status")
    assert _string(bayes_baseline, "demo_layout") == "app.py + benchmark.py + variants/"
    assert bayes_baseline["variant"] == "baseline"
    assert "backing_replay_command" in bayes_baseline
    assert "what_is_being_optimized" in bayes_baseline
    assert "manual_app_command" in bayes_baseline
    assert "autoclanker_relationship" in bayes_baseline
    assert "llm_note" in bayes_baseline


@covers("M7-004")
def test_code_showcase_variants_are_self_describing_when_run_directly() -> None:
    for relative_path in (
        ROOT
        / "docs"
        / "toy_examples"
        / "autoresearch_command_autocomplete"
        / "variants"
        / "baseline.py",
        ROOT
        / "docs"
        / "toy_examples"
        / "cevolve_sort_partition"
        / "variants"
        / "baseline.py",
        ROOT
        / "docs"
        / "toy_examples"
        / "bayes_pair_feature_trainer"
        / "variants"
        / "baseline.py",
    ):
        process = subprocess.run(
            [sys.executable, str(relative_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(process.stdout)
        if not isinstance(payload, dict):
            raise AssertionError("Variant payload must be a JSON object.")
        assert payload["demo_layout"] == "app.py + benchmark.py + variants/"
        assert payload["backing_replay_command"]
        assert payload["what_changed"]


@covers("M7-004")
def test_code_showcase_apps_are_self_describing_when_run_directly() -> None:
    for relative_path in (
        ROOT / "docs" / "toy_examples" / "autoresearch_command_autocomplete" / "app.py",
        ROOT / "docs" / "toy_examples" / "cevolve_sort_partition" / "app.py",
        ROOT / "docs" / "toy_examples" / "bayes_pair_feature_trainer" / "app.py",
    ):
        process = subprocess.run(
            [sys.executable, str(relative_path), "--variant", "baseline"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(process.stdout)
        if not isinstance(payload, dict):
            raise AssertionError("App payload must be a JSON object.")
        assert payload["what_the_app_does"]
        assert payload["optimization_surface"]
        assert payload["benchmark_preview"]
        assert payload["next_step_benchmark_command"]
        assert payload["next_step_replay_command"]


@covers("M7-004")
def test_bayes_code_showcase_replay_script_runs() -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "showcase" / "replay_backing_exercise.py"),
            "--showcase",
            "bayes_pair_feature_trainer",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict):
        raise AssertionError("Replay payload must be a JSON object.")
    observed = cast(dict[str, object], payload["observed"])
    assert payload["showcase"] == "bayes_pair_feature_trainer"
    assert payload["backing_demo_kind"] == "autoclanker_cli_session"
    assert "manual_replay_commands" in payload
    assert observed["cold_beliefs_top_candidate"] == "cand_c_compiled_context_pair"
    assert observed["cold_control_top_candidate"] == "cand_a_default"
    assert observed["belief_commit_recommended"] is True
    assert observed["control_commit_recommended"] is False
