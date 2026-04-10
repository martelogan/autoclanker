from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from autoclanker.cli import main as autoclanker_main

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True, slots=True)
class _ExerciseSpec:
    exercise: str
    goal: str
    minimum_required_files: tuple[str, ...]
    optional_files: tuple[str, ...]
    input_path: Path
    adapter_config_path: Path | None = None
    candidates_path: Path | None = None


_EXERCISES: dict[str, _ExerciseSpec] = {
    "bayes_quickstart": _ExerciseSpec(
        exercise="bayes_quickstart",
        goal="Use rough human optimization ideas to move the cold-start ranking.",
        minimum_required_files=(
            "examples/idea_inputs/bayes_quickstart.yaml",
            "examples/live_exercises/bayes_quickstart/candidates.json",
        ),
        optional_files=(
            "examples/live_exercises/bayes_quickstart/app.py",
            "examples/live_exercises/bayes_quickstart/expected_outcome.json",
        ),
        input_path=ROOT / "examples" / "idea_inputs" / "bayes_quickstart.yaml",
        candidates_path=ROOT
        / "examples"
        / "live_exercises"
        / "bayes_quickstart"
        / "candidates.json",
    ),
    "autoresearch_simple": _ExerciseSpec(
        exercise="autoresearch_simple",
        goal="Use live upstream-backed ideas to rank better autoresearch train.py settings.",
        minimum_required_files=(
            "examples/idea_inputs/autoresearch_simple.yaml",
            "examples/live_exercises/autoresearch_simple/adapter.local.yaml",
        ),
        optional_files=(
            "examples/live_exercises/autoresearch_simple/expected_outcome.json",
            "docs/toy_examples/autoresearch_command_autocomplete/",
        ),
        input_path=ROOT / "examples" / "idea_inputs" / "autoresearch_simple.yaml",
        adapter_config_path=ROOT
        / "examples"
        / "live_exercises"
        / "autoresearch_simple"
        / "adapter.local.yaml",
    ),
    "cevolve_synergy": _ExerciseSpec(
        exercise="cevolve_synergy",
        goal="Use live upstream-backed ideas to surface the synergistic cEvolve combination.",
        minimum_required_files=(
            "examples/idea_inputs/cevolve_synergy.yaml",
            "examples/live_exercises/cevolve_synergy/adapter.local.yaml",
        ),
        optional_files=(
            "examples/live_exercises/cevolve_synergy/train.py",
            "examples/live_exercises/cevolve_synergy/expected_outcome.json",
        ),
        input_path=ROOT / "examples" / "idea_inputs" / "cevolve_synergy.yaml",
        adapter_config_path=ROOT
        / "examples"
        / "live_exercises"
        / "cevolve_synergy"
        / "adapter.local.yaml",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the lowest-cruft autoclanker session/beliefs flow for one of the "
            "starter live exercises."
        )
    )
    parser.add_argument(
        "--exercise",
        required=True,
        choices=tuple(sorted(_EXERCISES)),
        help="Which exercise to replay.",
    )
    parser.add_argument(
        "--session-root",
        help="Optional session root. Defaults to a temporary directory.",
    )
    return parser.parse_args()


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("Expected a JSON object.")
    return cast(dict[str, object], value)


def _require_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeError("Expected a JSON list.")
    return cast(list[object], value)


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = autoclanker_main(argv)
    if exit_code != 0:
        raise RuntimeError(
            f"autoclanker {' '.join(argv)!r} failed with exit code {exit_code}: "
            f"{stderr.getvalue().strip()}"
        )
    payload = json.loads(stdout.getvalue())
    if not isinstance(payload, dict):
        raise RuntimeError("Expected a JSON object from autoclanker CLI.")
    return cast(dict[str, object], payload)


def _top_candidate(suggest_payload: dict[str, object]) -> dict[str, object]:
    ranked = [
        _require_mapping(item)
        for item in _require_list(suggest_payload["ranked_candidates"])
    ]
    if not ranked:
        raise RuntimeError("Expected at least one ranked candidate.")
    return ranked[0]


def _genotype_mapping(candidate_payload: dict[str, object]) -> dict[str, str]:
    genotype_items = _require_list(candidate_payload["genotype"])
    genotype: dict[str, str] = {}
    for item in genotype_items:
        mapping = _require_mapping(item)
        genotype[str(mapping["gene_id"])] = str(mapping["state_id"])
    return genotype


def _preview_summary(preview_payload: dict[str, object]) -> dict[str, object]:
    preview_items = [
        _require_mapping(item)
        for item in _require_list(preview_payload["belief_previews"])
    ]
    compiled_count = 0
    compiled_targets: list[str] = []
    proposal_compile_status: str | None = None
    for item in preview_items:
        if item.get("compile_status") == "compiled":
            compiled_count += 1
        for compiled_item in _require_list(item.get("compiled_items", [])):
            compiled_targets.append(str(_require_mapping(compiled_item)["target_ref"]))
        if (
            proposal_compile_status is None
            and item.get("compile_status") == "metadata_only"
        ):
            proposal_compile_status = str(item["compile_status"])
    return {
        "compiled_belief_count": compiled_count,
        "compiled_targets": sorted(compiled_targets),
        "proposal_compile_status": proposal_compile_status,
    }


def _manual_commands(spec: _ExerciseSpec) -> list[str]:
    commands: list[str] = []
    if spec.adapter_config_path is not None:
        commands.append("./bin/dev test-upstream-live")
        commands.append(
            "./bin/dev exec -- autoclanker adapter registry --input "
            f"{spec.adapter_config_path.relative_to(ROOT)}"
        )
    else:
        commands.append("./bin/dev exec -- autoclanker adapter registry")
    commands.append(
        "./bin/dev exec -- autoclanker beliefs expand-ideas --input "
        f"{spec.input_path.relative_to(ROOT)}"
    )
    commands.append(
        "./bin/dev exec -- python scripts/live/replay_ideas_demo.py "
        f"--exercise {spec.exercise}"
    )
    return commands


def _run_one(spec: _ExerciseSpec, session_root: Path) -> dict[str, object]:
    preview_argv = ["beliefs", "preview", "--input", str(spec.input_path)]
    expand_payload = _run_cli(
        ["beliefs", "expand-ideas", "--input", str(spec.input_path)]
    )
    if spec.adapter_config_path is not None:
        preview_argv.extend(["--adapter-config", str(spec.adapter_config_path)])
        probe = _run_cli(["adapter", "probe", "--input", str(spec.adapter_config_path)])
        registry_payload = _run_cli(
            ["adapter", "registry", "--input", str(spec.adapter_config_path)]
        )
    else:
        probe = None
        registry_payload = _run_cli(["adapter", "registry"])
    preview_payload = _run_cli(preview_argv)

    init_argv = [
        "session",
        "init",
        "--beliefs-input",
        str(spec.input_path),
        "--session-root",
        str(session_root),
    ]
    if spec.adapter_config_path is not None:
        init_argv.extend(["--adapter-config", str(spec.adapter_config_path)])
    init_payload = _run_cli(init_argv)
    session_id = str(init_payload["session_id"])

    apply_argv = [
        "session",
        "apply-beliefs",
        "--session-id",
        session_id,
        "--preview-digest",
        str(init_payload["preview_digest"]),
        "--session-root",
        str(session_root),
    ]
    if spec.adapter_config_path is not None:
        apply_argv.extend(["--adapter-config", str(spec.adapter_config_path)])
    _run_cli(apply_argv)

    suggest_argv = [
        "session",
        "suggest",
        "--session-id",
        session_id,
        "--session-root",
        str(session_root),
    ]
    if spec.adapter_config_path is not None:
        suggest_argv.extend(["--adapter-config", str(spec.adapter_config_path)])
    if spec.candidates_path is not None:
        suggest_argv.extend(["--candidates-input", str(spec.candidates_path)])
    suggest_payload = _run_cli(suggest_argv)
    top_candidate = _top_candidate(suggest_payload)
    observed: dict[str, object] = {
        "top_candidate": str(top_candidate["candidate_id"]),
        "top_genotype": _genotype_mapping(top_candidate),
        "top_predicted_utility": float(
            cast(int | float, top_candidate["predicted_utility"])
        ),
        "top_valid_probability": float(
            cast(int | float, top_candidate["valid_probability"])
        ),
        "query_prompt": str(
            _require_mapping(_require_list(suggest_payload["queries"])[0])["prompt"]
        )
        if _require_list(suggest_payload["queries"])
        else None,
    }

    if spec.exercise == "bayes_quickstart":
        control_session_id = "quickstart_control_replay"
        _run_cli(
            [
                "session",
                "init",
                "--session-id",
                control_session_id,
                "--era-id",
                str(init_payload["era_id"]),
                "--session-root",
                str(session_root),
            ]
        )
        control_suggest = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root),
                "--candidates-input",
                str(spec.candidates_path),
            ]
        )
        observed["control_top_candidate"] = str(
            _top_candidate(control_suggest)["candidate_id"]
        )

    payload: dict[str, object] = {
        "exercise": spec.exercise,
        "goal": spec.goal,
        "minimum_required_files": list(spec.minimum_required_files),
        "optional_files": list(spec.optional_files),
        "manual_replay_commands": _manual_commands(spec),
        "registry": registry_payload,
        "expanded_belief_kinds": [
            str(_require_mapping(item)["kind"])
            for item in _require_list(expand_payload["beliefs"])
        ],
        "preview_summary": _preview_summary(preview_payload),
        "observed": observed,
    }
    if probe is not None:
        payload["probe"] = probe
    return payload


def main() -> int:
    args = _parse_args()
    spec = _EXERCISES[args.exercise]
    if args.session_root:
        session_root = Path(args.session_root).resolve()
        session_root.mkdir(parents=True, exist_ok=True)
        payload = _run_one(spec, session_root)
    else:
        with tempfile.TemporaryDirectory(
            prefix=f"autoclanker-{spec.exercise}-"
        ) as temp_dir:
            payload = _run_one(spec, Path(temp_dir))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
