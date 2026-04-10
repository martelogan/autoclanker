from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile

from collections.abc import Callable
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    ValidAdapterConfig,
    ValidEvalResult,
)
from autoclanker.cli import main as autoclanker_main

ROOT = Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the actual autoclanker-backed demo behind one of the "
            "human-readable toy mirrors."
        )
    )
    parser.add_argument(
        "--showcase",
        required=True,
        choices=(
            "autoresearch_command_autocomplete",
            "cevolve_sort_partition",
            "bayes_pair_feature_trainer",
        ),
        help="Which toy mirror to replay.",
    )
    parser.add_argument(
        "--session-root",
        help=(
            "Optional session root for the Bayes replay. Defaults to a temporary "
            "directory."
        ),
    )
    return parser.parse_args()


def _expected_payload(exercise: str) -> dict[str, object]:
    path = ROOT / "examples" / "live_exercises" / exercise / "expected_outcome.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} must contain a JSON object.")
    return cast(dict[str, object], payload)


def _resolve_repo_path(kind: str) -> Path:
    env_name = (
        "AUTOCLANKER_LIVE_AUTORESEARCH_PATH"
        if kind == "autoresearch"
        else "AUTOCLANKER_LIVE_CEVOLVE_PATH"
    )
    candidate_paths = [
        os.environ.get(env_name),
        str(ROOT / ".local" / "real-upstreams-pristine" / kind),
        str(ROOT / ".local" / "real-upstreams" / kind),
        str(ROOT / "references" / kind),
    ]
    for raw_path in candidate_paths:
        if raw_path is None or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser().resolve()
        if path.exists():
            return path
    raise RuntimeError(
        f"Could not resolve a repo path for {kind}. Run ./bin/dev test-upstream-live once, "
        f"or set {env_name}."
    )


def _metric_float(result: ValidEvalResult, key: str) -> float:
    return float(cast(float | int | str, result.raw_metrics[key]))


def _genotype_from_mapping(
    adapter: AutoresearchAdapter | CevolveAdapter,
    mapping: dict[str, object],
) -> tuple[GeneStateRef, ...]:
    registry = adapter.build_registry()
    genotype = {ref.gene_id: ref for ref in registry.default_genotype()}
    for gene_id, state_id in mapping.items():
        genotype[str(gene_id)] = GeneStateRef(
            gene_id=str(gene_id),
            state_id=str(state_id),
        )
    return tuple(genotype.values())


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


def _quiet_call(
    func: Callable[..., object],
    /,
    *args: object,
    **kwargs: object,
) -> object:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return func(*args, **kwargs)


def _autoresearch_manual_replay_commands() -> list[str]:
    return [
        "./bin/dev test-upstream-live",
        (
            "./bin/dev exec -- autoclanker adapter probe --input "
            "examples/live_exercises/autoresearch_simple/adapter.local.yaml"
        ),
        (
            "./bin/dev exec -- python "
            "scripts/showcase/replay_backing_exercise.py --showcase "
            "autoresearch_command_autocomplete"
        ),
    ]


def _cevolve_manual_replay_commands() -> list[str]:
    return [
        "./bin/dev test-upstream-live",
        (
            "./bin/dev exec -- autoclanker adapter probe --input "
            "examples/live_exercises/cevolve_synergy/adapter.local.yaml"
        ),
        (
            "./bin/dev exec -- python "
            "scripts/showcase/replay_backing_exercise.py --showcase "
            "cevolve_sort_partition"
        ),
    ]


def _bayes_manual_replay_commands() -> list[str]:
    return [
        (
            "./bin/dev exec -- autoclanker session init "
            "--session-id bayes-showcase "
            "--beliefs-input examples/live_exercises/bayes_complex/beliefs.yaml "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- autoclanker session apply-beliefs "
            "--session-id bayes-showcase "
            "--preview-digest <preview_digest_from_init> "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- autoclanker session suggest "
            "--session-id bayes-showcase "
            "--candidates-input examples/live_exercises/bayes_complex/candidates.json "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- python "
            "scripts/live/generate_bayes_complex_evals.py "
            "--output-dir .autoclanker-exercises/evals"
        ),
        (
            "./bin/dev exec -- autoclanker session ingest-eval "
            "--session-id bayes-showcase "
            "--input .autoclanker-exercises/evals/eval_001.json "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- autoclanker session fit "
            "--session-id bayes-showcase "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- autoclanker session recommend-commit "
            "--session-id bayes-showcase "
            "--session-root .autoclanker-exercises"
        ),
        (
            "./bin/dev exec -- python "
            "scripts/showcase/replay_backing_exercise.py --showcase "
            "bayes_pair_feature_trainer"
        ),
    ]


def _replay_autoresearch() -> dict[str, object]:
    repo_path = _resolve_repo_path("autoresearch")
    expected = _expected_payload("autoresearch_simple")
    adapter = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path=str(repo_path),
            allow_missing=False,
            metadata={
                "adapter_module": os.environ.get(
                    "AUTOCLANKER_LIVE_AUTORESEARCH_ADAPTER_MODULE",
                    "autoclanker.bayes_layer.live_upstreams",
                )
            },
        )
    )
    baseline = cast(
        ValidEvalResult,
        _quiet_call(
            adapter.evaluate_candidate,
            era_id="era_live_showcase",
            candidate_id="autoresearch_baseline",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], expected["baseline"]),
            ),
            seed=7,
        ),
    )
    improved = cast(
        ValidEvalResult,
        _quiet_call(
            adapter.evaluate_candidate,
            era_id="era_live_showcase",
            candidate_id="autoresearch_improved",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], expected["improved"]),
            ),
            seed=11,
        ),
    )
    failure = cast(
        ValidEvalResult,
        _quiet_call(
            adapter.evaluate_candidate,
            era_id="era_live_showcase",
            candidate_id="autoresearch_failure",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], expected["failure_candidate"]),
            ),
            seed=13,
        ),
    )
    return {
        "showcase": "autoresearch_command_autocomplete",
        "goal": (
            "Lower validation bits-per-byte on the upstream-anchored autoresearch demo."
        ),
        "toy_code_role": (
            "Readable code snapshot only; it mirrors the optimization shape, "
            "not the literal upstream file or knob names."
        ),
        "mirror_scope": (
            "The toy autocomplete app is a simplified mirror of the same "
            "mostly-additive optimization landscape."
        ),
        "metric_name": "val_bpb",
        "optimize_direction": "minimize",
        "optimized_knobs": [
            "DEPTH",
            "WINDOW_PATTERN",
            "TOTAL_BATCH_SIZE",
            "MATRIX_LR",
            "WARMUP_RATIO",
        ],
        "backing_demo_kind": "live_adapter",
        "backing_live_exercise": "examples/live_exercises/autoresearch_simple",
        "toy_code_dir": "docs/toy_examples/autoresearch_command_autocomplete",
        "manual_replay_commands": _autoresearch_manual_replay_commands(),
        "observed": {
            "baseline_val_bpb": _metric_float(baseline, "val_bpb"),
            "improved_val_bpb": _metric_float(improved, "val_bpb"),
            "failure_status": failure.status,
        },
    }


def _replay_cevolve() -> dict[str, object]:
    repo_path = _resolve_repo_path("cevolve")
    expected = _expected_payload("cevolve_synergy")
    adapter = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path=str(repo_path),
            allow_missing=False,
            metadata={
                "adapter_module": os.environ.get(
                    "AUTOCLANKER_LIVE_CEVOLVE_ADAPTER_MODULE",
                    "autoclanker.bayes_layer.live_upstreams",
                )
            },
        )
    )
    baseline = cast(
        ValidEvalResult,
        _quiet_call(
            adapter.evaluate_candidate,
            era_id="era_live_showcase",
            candidate_id="cevolve_baseline",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], expected["baseline"]),
            ),
            seed=5,
        ),
    )
    improved = cast(
        ValidEvalResult,
        _quiet_call(
            adapter.evaluate_candidate,
            era_id="era_live_showcase",
            candidate_id="cevolve_improved",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], expected["improved"]),
            ),
            seed=13,
        ),
    )
    single_changes = cast(list[object], expected["single_changes"])
    single_results = [
        cast(
            ValidEvalResult,
            _quiet_call(
                adapter.evaluate_candidate,
                era_id="era_live_showcase",
                candidate_id=f"cevolve_single_{index}",
                genotype=_genotype_from_mapping(
                    adapter,
                    cast(dict[str, object], item),
                ),
                seed=21 + index,
            ),
        )
        for index, item in enumerate(single_changes, start=1)
    ]
    return {
        "showcase": "cevolve_sort_partition",
        "goal": "Reduce sorting time on the live cEvolve-backed demo.",
        "toy_code_role": (
            "Readable code snapshot only; it mirrors the optimization shape, "
            "not the literal live exercise file layout."
        ),
        "mirror_scope": (
            "The toy integer sorter is a simplified mirror of the same "
            "interaction-heavy optimization landscape."
        ),
        "metric_name": "time_ms",
        "optimize_direction": "minimize",
        "optimized_knobs": [
            "INSERTION_THRESHOLD",
            "PARTITION_SCHEME",
            "PIVOT_STRATEGY",
            "USE_ITERATIVE",
        ],
        "backing_demo_kind": "live_adapter",
        "backing_live_exercise": "examples/live_exercises/cevolve_synergy",
        "toy_code_dir": "docs/toy_examples/cevolve_sort_partition",
        "manual_replay_commands": _cevolve_manual_replay_commands(),
        "observed": {
            "baseline_time_ms": _metric_float(baseline, "time_ms"),
            "improved_time_ms": _metric_float(improved, "time_ms"),
            "best_single_time_ms": min(
                _metric_float(result, "time_ms") for result in single_results
            ),
        },
    }


def _replay_bayes(session_root: Path | None) -> dict[str, object]:
    if session_root is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="autoclanker-bayes-showcase-")
        session_root_path = Path(temp_dir.name)
    else:
        temp_dir = None
        session_root_path = session_root
        session_root_path.mkdir(parents=True, exist_ok=True)

    try:
        beliefs_path = (
            ROOT / "examples" / "live_exercises" / "bayes_complex" / "beliefs.yaml"
        )
        candidates_path = (
            ROOT / "examples" / "live_exercises" / "bayes_complex" / "candidates.json"
        )
        eval_dir = session_root_path / "evals"

        init_output = _run_cli(
            [
                "session",
                "init",
                "--beliefs-input",
                str(beliefs_path),
                "--session-root",
                str(session_root_path),
            ]
        )
        belief_session_id = str(init_output["session_id"])
        preview_digest = str(init_output["preview_digest"])

        control_session_id = "showcase_bayes_control"
        _run_cli(
            [
                "session",
                "init",
                "--session-id",
                control_session_id,
                "--era-id",
                "era_parser_advanced",
                "--session-root",
                str(session_root_path),
            ]
        )
        _run_cli(
            [
                "session",
                "apply-beliefs",
                "--session-id",
                belief_session_id,
                "--preview-digest",
                preview_digest,
                "--session-root",
                str(session_root_path),
            ]
        )

        cold_beliefs = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root_path),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        cold_control = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root_path),
                "--candidates-input",
                str(candidates_path),
            ]
        )

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "live" / "generate_bayes_complex_evals.py"),
                "--output-dir",
                str(eval_dir),
            ],
            cwd=ROOT,
            check=True,
        )

        for eval_path in sorted(eval_dir.glob("*.json")):
            for session_id in (belief_session_id, control_session_id):
                _run_cli(
                    [
                        "session",
                        "ingest-eval",
                        "--session-id",
                        session_id,
                        "--input",
                        str(eval_path),
                        "--session-root",
                        str(session_root_path),
                    ]
                )

        for session_id in (belief_session_id, control_session_id):
            _run_cli(
                [
                    "session",
                    "fit",
                    "--session-id",
                    session_id,
                    "--session-root",
                    str(session_root_path),
                ]
            )

        fitted_beliefs = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root_path),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        fitted_control = _run_cli(
            [
                "session",
                "suggest",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root_path),
                "--candidates-input",
                str(candidates_path),
            ]
        )
        belief_decision = _run_cli(
            [
                "session",
                "recommend-commit",
                "--session-id",
                belief_session_id,
                "--session-root",
                str(session_root_path),
            ]
        )
        control_decision = _run_cli(
            [
                "session",
                "recommend-commit",
                "--session-id",
                control_session_id,
                "--session-root",
                str(session_root_path),
            ]
        )

        cold_belief_top = cast(
            list[dict[str, object]], cold_beliefs["ranked_candidates"]
        )[0]
        cold_control_top = cast(
            list[dict[str, object]], cold_control["ranked_candidates"]
        )[0]
        belief_good_pair = next(
            item
            for item in cast(
                list[dict[str, object]], fitted_beliefs["ranked_candidates"]
            )
            if item["candidate_id"] == "cand_c_compiled_context_pair"
        )
        control_good_pair = next(
            item
            for item in cast(
                list[dict[str, object]], fitted_control["ranked_candidates"]
            )
            if item["candidate_id"] == "cand_c_compiled_context_pair"
        )
        return {
            "showcase": "bayes_pair_feature_trainer",
            "goal": (
                "Maximize utility in the real autoclanker session flow by using "
                "beliefs to promote an unseen good pair."
            ),
            "toy_code_role": (
                "Readable code snapshot only; it mirrors the optimization shape, "
                "while the real demo uses autoclanker session commands."
            ),
            "mirror_scope": (
                "The toy trainer uses simpler feature names than the live "
                "candidate payloads, but it mirrors the same prior-guided "
                "better-unseen-pair story."
            ),
            "metric_name": "utility",
            "optimize_direction": "maximize",
            "optimized_knobs": [
                "OPTIM_LR",
                "MODEL_DEPTH",
                "MODEL_WIDTH",
                "BATCH_SIZE",
            ],
            "backing_demo_kind": "autoclanker_cli_session",
            "backing_live_exercise": "examples/live_exercises/bayes_complex",
            "toy_code_dir": "docs/toy_examples/bayes_pair_feature_trainer",
            "manual_replay_commands": _bayes_manual_replay_commands(),
            "observed": {
                "cold_beliefs_top_candidate": str(cold_belief_top["candidate_id"]),
                "cold_control_top_candidate": str(cold_control_top["candidate_id"]),
                "belief_guided_utility": float(
                    cast(float | int, belief_good_pair["predicted_utility"])
                ),
                "control_utility": float(
                    cast(float | int, control_good_pair["predicted_utility"])
                ),
                "belief_commit_recommended": bool(belief_decision["recommended"]),
                "control_commit_recommended": bool(control_decision["recommended"]),
            },
            "session_root": str(session_root_path),
        }
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> int:
    args = _parse_args()
    if args.showcase == "autoresearch_command_autocomplete":
        payload = _replay_autoresearch()
    elif args.showcase == "cevolve_sort_partition":
        payload = _replay_cevolve()
    else:
        payload = _replay_bayes(Path(args.session_root) if args.session_root else None)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
