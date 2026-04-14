from __future__ import annotations

import argparse
import io
import json
import tempfile
import time

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer import (
    EraState,
    acquisition as acquisition_module,
    compile_beliefs,
    ingest_human_beliefs,
    load_bayes_layer_config,
    load_serialized_payload,
)
from autoclanker.bayes_layer.acquisition import rank_candidates
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.surrogate_feasibility import fit_feasibility_surrogate
from autoclanker.bayes_layer.surrogate_objective import (
    _heuristic_fit_objective_surrogate,
    fit_objective_surrogate,
)
from autoclanker.bayes_layer.types import GeneStateRef, ValidAdapterConfig
from autoclanker.cli import main

ROOT = Path(__file__).resolve().parents[2]


def _run_cli(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        exit_code = main(argv)
    if exit_code != 0:
        raise RuntimeError(
            f"autoclanker {' '.join(argv)!r} failed: {stderr.getvalue().strip()}"
        )
    return _require_mapping(json.loads(stdout.getvalue()))


def _top_candidate_id(payload: dict[str, object]) -> str:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    return str(_require_mapping(ranked_candidates[0])["candidate_id"])


def _candidate_utility(payload: dict[str, object], candidate_id: str) -> float:
    ranked_candidates = cast(list[object], payload["ranked_candidates"])
    for item in ranked_candidates:
        mapping = _require_mapping(item)
        if mapping["candidate_id"] != candidate_id:
            continue
        value = mapping.get("predicted_utility")
        if not isinstance(value, int | float):
            raise ValueError("predicted_utility must be numeric.")
        return float(value)
    raise ValueError(f"Missing candidate {candidate_id!r}.")


def _require_mapping(value: object, *, label: str = "payload") -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _require_sequence(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON list.")
    return cast(list[object], value)


def _fixture_adapter() -> FixtureAdapter:
    return FixtureAdapter(
        ValidAdapterConfig(
            kind="fixture",
            mode="fixture",
            session_root=".autoclanker",
        )
    )


def _candidate_with_overrides(
    adapter: FixtureAdapter,
    *,
    matcher: str | None = None,
    plan: str | None = None,
    window: str | None = None,
    chunk: str | None = None,
) -> tuple[GeneStateRef, ...]:
    registry = adapter.build_registry()
    genotype = {ref.gene_id: ref for ref in registry.default_genotype()}
    if matcher is not None:
        genotype["parser.matcher"] = GeneStateRef(
            gene_id="parser.matcher",
            state_id=matcher,
        )
    if plan is not None:
        genotype["parser.plan"] = GeneStateRef(
            gene_id="parser.plan",
            state_id=plan,
        )
    if window is not None:
        genotype["capture.window"] = GeneStateRef(
            gene_id="capture.window",
            state_id=window,
        )
    if chunk is not None:
        genotype["io.chunk"] = GeneStateRef(
            gene_id="io.chunk",
            state_id=chunk,
        )
    return tuple(genotype.values())


def _backend_comparison_report() -> dict[str, object]:
    adapter = _fixture_adapter()
    registry = adapter.build_registry()
    beliefs = ingest_human_beliefs(
        load_serialized_payload(
            ROOT / "examples" / "human_beliefs" / "expert_session.json"
        )
    )
    compiled = compile_beliefs(
        beliefs,
        registry,
        EraState(era_id=beliefs.session_context.era_id),
    )
    observations = (
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="baseline",
            genotype=registry.default_genotype(),
            seed=1,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="compiled_only",
            genotype=_candidate_with_overrides(
                adapter,
                matcher="matcher_compiled",
            ),
            seed=2,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="compiled_pair_a",
            genotype=_candidate_with_overrides(
                adapter,
                matcher="matcher_compiled",
                plan="plan_context_pair",
            ),
            seed=3,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="compiled_pair_b",
            genotype=_candidate_with_overrides(
                adapter,
                matcher="matcher_compiled",
                plan="plan_context_pair",
            ),
            seed=4,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="wide_window",
            genotype=_candidate_with_overrides(
                adapter,
                window="window_wide",
                chunk="chunk_large",
            ),
            seed=5,
        ),
    )
    candidate_pool = (
        ("cand_a_default", registry.default_genotype()),
        (
            "cand_b_compiled_matcher",
            _candidate_with_overrides(adapter, matcher="matcher_compiled"),
        ),
        (
            "cand_c_compiled_context_pair",
            _candidate_with_overrides(
                adapter,
                matcher="matcher_compiled",
                plan="plan_context_pair",
            ),
        ),
        (
            "cand_d_wide_window_large_chunk",
            _candidate_with_overrides(
                adapter,
                window="window_wide",
                chunk="chunk_large",
            ),
        ),
    )
    config = load_bayes_layer_config()
    optimistic_config = replace(
        config,
        acquisition=replace(config.acquisition, kind="optimistic_upper_confidence"),
    )
    era_state = EraState(
        era_id=beliefs.session_context.era_id,
        observation_count=len(observations),
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
        config=config,
    )

    def lane_payload(objective_mode: str) -> dict[str, object]:
        fit_started = time.perf_counter()
        if objective_mode == "heuristic":
            objective = _heuristic_fit_objective_surrogate(
                observations,
                registry=registry,
                compiled_priors=compiled,
                era_state=era_state,
                config=config,
            )
            ranking_config = optimistic_config
        elif objective_mode == "exact_optimistic":
            objective = fit_objective_surrogate(
                observations,
                registry=registry,
                compiled_priors=compiled,
                era_state=era_state,
                config=config,
            )
            ranking_config = optimistic_config
        else:
            objective = fit_objective_surrogate(
                observations,
                registry=registry,
                compiled_priors=compiled,
                era_state=era_state,
                config=config,
            )
            ranking_config = config
        fit_runtime_ms = (time.perf_counter() - fit_started) * 1000.0
        rank_started = time.perf_counter()
        original_sample_factor = acquisition_module._sample_factor
        if objective_mode == "exact_forced_sampling_fallback":
            acquisition_module._sample_factor = lambda _covariance: None
        try:
            ranked = rank_candidates(
                candidate_pool,
                registry=registry,
                objective_posterior=objective,
                feasibility_posterior=feasibility,
                compiled_priors=compiled,
                config=ranking_config,
            )
        finally:
            acquisition_module._sample_factor = original_sample_factor
        ranking_runtime_ms = (time.perf_counter() - rank_started) * 1000.0
        return {
            "objective_backend": objective.backend,
            "acquisition_backend": ranked[0].acquisition_backend,
            "acquisition_fallback_reason": ranked[0].acquisition_fallback_reason,
            "top_candidate": ranked[0].candidate_id,
            "top_candidate_predicted_utility": ranked[0].predicted_utility,
            "top_candidate_acquisition_score": ranked[0].acquisition_score,
            "fit_runtime_ms": round(fit_runtime_ms, 4),
            "ranking_runtime_ms": round(ranking_runtime_ms, 4),
            "objective_sampleable": objective.sampleable,
            "objective_condition_number": objective.condition_number,
            "objective_fallback_reason": objective.fallback_reason,
        }

    heuristic_lane = lane_payload("heuristic")
    exact_optimistic_lane = lane_payload("exact_optimistic")
    exact_thompson_lane = lane_payload("exact_thompson")
    exact_forced_fallback_lane = lane_payload("exact_forced_sampling_fallback")
    return {
        "comparison_type": "deterministic_backend_comparison",
        "observation_count": len(observations),
        "lanes": {
            "heuristic_objective_optimistic": heuristic_lane,
            "exact_objective_optimistic": exact_optimistic_lane,
            "exact_objective_thompson": exact_thompson_lane,
            "exact_objective_sampling_fallback": exact_forced_fallback_lane,
        },
        "conclusion": {
            "exact_backend_active": exact_optimistic_lane["objective_backend"]
            == "exact_joint_linear",
            "thompson_backend_active": exact_thompson_lane["acquisition_backend"]
            == "constrained_thompson_sampling",
            "heuristic_backend_active": heuristic_lane["objective_backend"]
            == "heuristic_independent_normal",
            "fallback_free_exact_lane": exact_thompson_lane["objective_fallback_reason"]
            is None,
            "sampled_fallback_visible": exact_forced_fallback_lane[
                "acquisition_fallback_reason"
            ]
            is not None,
        },
    }


def _init_and_apply(
    *,
    session_root: Path,
    session_id: str,
    ideas_json: str | None = None,
    canonicalization_model: str | None = None,
) -> None:
    argv = [
        "session",
        "init",
        "--session-id",
        session_id,
        "--era-id",
        "era_log_parser_v1",
        "--session-root",
        str(session_root),
    ]
    if ideas_json is not None:
        argv.extend(["--ideas-json", ideas_json])
    if canonicalization_model is not None:
        argv.extend(["--canonicalization-model", canonicalization_model])
    init_output = _run_cli(argv)
    preview_digest = init_output.get("preview_digest")
    if preview_digest is None:
        return
    _run_cli(
        [
            "session",
            "apply-beliefs",
            "--session-id",
            session_id,
            "--preview-digest",
            str(preview_digest),
            "--session-root",
            str(session_root),
        ]
    )


def build_report() -> dict[str, object]:
    candidates_path = (
        ROOT / "examples" / "live_exercises" / "bayes_quickstart" / "candidates.json"
    )
    with tempfile.TemporaryDirectory(prefix="autoclanker-benchmark-") as tempdir:
        session_root = Path(tempdir) / "sessions"
        frontier_path = Path(tempdir) / "frontier.json"
        frontier_path.write_text(
            json.dumps(
                {
                    "frontier_id": "parser_frontier_demo",
                    "default_family_id": "family_default",
                    "candidates": [
                        {
                            "candidate_id": "cand_a_default",
                            "family_id": "baseline",
                            "origin_kind": "seed",
                            "genotype": [
                                {
                                    "gene_id": "parser.matcher",
                                    "state_id": "matcher_basic",
                                },
                                {"gene_id": "parser.plan", "state_id": "plan_default"},
                                {
                                    "gene_id": "capture.window",
                                    "state_id": "window_default",
                                },
                            ],
                        },
                        {
                            "candidate_id": "cand_b_compiled_matcher",
                            "family_id": "matcher_family",
                            "origin_kind": "belief",
                            "parent_belief_ids": ["belief_compiled"],
                            "budget_weight": 2.0,
                            "genotype": [
                                {
                                    "gene_id": "parser.matcher",
                                    "state_id": "matcher_compiled",
                                },
                                {"gene_id": "parser.plan", "state_id": "plan_default"},
                                {
                                    "gene_id": "capture.window",
                                    "state_id": "window_default",
                                },
                            ],
                        },
                        {
                            "candidate_id": "cand_c_compiled_context_pair",
                            "family_id": "plan_family",
                            "origin_kind": "merge",
                            "parent_candidate_ids": [
                                "cand_a_default",
                                "cand_b_compiled_matcher",
                            ],
                            "origin_query_ids": ["query_pair_001"],
                            "notes": "merge compiled matcher and context pairing",
                            "genotype": [
                                {
                                    "gene_id": "parser.matcher",
                                    "state_id": "matcher_compiled",
                                },
                                {
                                    "gene_id": "parser.plan",
                                    "state_id": "plan_context_pair",
                                },
                                {
                                    "gene_id": "capture.window",
                                    "state_id": "window_default",
                                },
                            ],
                        },
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _init_and_apply(session_root=session_root, session_id="outer_control")
        _init_and_apply(
            session_root=session_root,
            session_id="proposal_only",
            ideas_json='["Try a moonbeam dragon refactor with kaleidoscope anchors."]',
        )
        _init_and_apply(
            session_root=session_root,
            session_id="deterministic_bayes",
            ideas_json='["Compiled regex matching probably helps this parser on repeated log formats.","Compiled matching works best together with the context pair plan."]',
        )
        _init_and_apply(
            session_root=session_root,
            session_id="hybrid_bayes",
            ideas_json='["A repeated-format fast path probably helps this parser."]',
            canonicalization_model="stub",
        )

        lanes = {}
        for session_id in (
            "outer_control",
            "proposal_only",
            "deterministic_bayes",
            "hybrid_bayes",
        ):
            payload = _run_cli(
                [
                    "session",
                    "suggest",
                    "--session-id",
                    session_id,
                    "--session-root",
                    str(session_root),
                    "--candidates-input",
                    str(candidates_path),
                ]
            )
            lanes[session_id] = {
                "top_candidate": _top_candidate_id(payload),
                "good_pair_predicted_utility": _candidate_utility(
                    payload, "cand_c_compiled_context_pair"
                ),
            }

        frontier_lanes = {}
        for session_id in (
            "outer_control",
            "deterministic_bayes",
            "hybrid_bayes",
        ):
            payload = _run_cli(
                [
                    "session",
                    "suggest",
                    "--session-id",
                    session_id,
                    "--session-root",
                    str(session_root),
                    "--candidates-input",
                    str(frontier_path),
                ]
            )
            frontier_summary = _require_mapping(
                payload["frontier_summary"], label="frontier_summary"
            )
            pending_queries = _require_sequence(
                frontier_summary["pending_queries"],
                label="pending_queries",
            )
            pending_merges = _require_sequence(
                frontier_summary["pending_merge_suggestions"],
                label="pending_merge_suggestions",
            )
            family_representatives = _require_sequence(
                frontier_summary["family_representatives"],
                label="family_representatives",
            )
            budget_allocations = _require_mapping(
                frontier_summary["budget_allocations"],
                label="budget_allocations",
            )
            frontier_lanes[session_id] = {
                "top_candidate": _top_candidate_id(payload),
                "family_count": frontier_summary["family_count"],
                "family_representative_count": len(family_representatives),
                "pending_query_count": len(pending_queries),
                "pending_merge_suggestion_count": len(pending_merges),
                "budget_allocations": budget_allocations,
            }

    deterministic_gain = cast(
        float, lanes["deterministic_bayes"]["good_pair_predicted_utility"]
    ) - cast(float, lanes["outer_control"]["good_pair_predicted_utility"])
    hybrid_gain = cast(
        float, lanes["hybrid_bayes"]["good_pair_predicted_utility"]
    ) - cast(float, lanes["proposal_only"]["good_pair_predicted_utility"])
    return {
        "targets": {
            "bayes_quickstart_parser": {
                "comparison_type": "zero_eval_cold_start",
                "cold_start_evals": 0,
                "lanes": lanes,
                "conclusion": {
                    "control_vs_proposal_only_same": (
                        lanes["outer_control"]["top_candidate"]
                        == lanes["proposal_only"]["top_candidate"]
                    ),
                    "deterministic_bayes_improves_good_pair": deterministic_gain,
                    "hybrid_bayes_improves_good_pair": hybrid_gain,
                    "bayes_top_candidate": lanes["hybrid_bayes"]["top_candidate"],
                },
            },
            "parser_frontier_family_budgeting": {
                "comparison_type": "deterministic_frontier",
                "cold_start_evals": 0,
                "lanes": frontier_lanes,
                "conclusion": {
                    "control_top_candidate": frontier_lanes["outer_control"][
                        "top_candidate"
                    ],
                    "deterministic_top_candidate": frontier_lanes[
                        "deterministic_bayes"
                    ]["top_candidate"],
                    "hybrid_top_candidate": frontier_lanes["hybrid_bayes"][
                        "top_candidate"
                    ],
                    "deterministic_family_count": frontier_lanes["deterministic_bayes"][
                        "family_count"
                    ],
                    "hybrid_family_count": frontier_lanes["hybrid_bayes"][
                        "family_count"
                    ],
                    "deterministic_pending_merge_suggestions": frontier_lanes[
                        "deterministic_bayes"
                    ]["pending_merge_suggestion_count"],
                    "hybrid_pending_merge_suggestions": frontier_lanes["hybrid_bayes"][
                        "pending_merge_suggestion_count"
                    ],
                    "deterministic_budget_sum": round(
                        sum(
                            float(value)
                            for value in cast(
                                dict[str, object],
                                frontier_lanes["deterministic_bayes"][
                                    "budget_allocations"
                                ],
                            ).values()
                        ),
                        6,
                    ),
                    "hybrid_budget_sum": round(
                        sum(
                            float(value)
                            for value in cast(
                                dict[str, object],
                                frontier_lanes["hybrid_bayes"]["budget_allocations"],
                            ).values()
                        ),
                        6,
                    ),
                },
            },
            "observed_backend_comparison": _backend_comparison_report(),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare control, proposal-only, deterministic Bayes, and hybrid Bayes lanes on the parser quickstart target."
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON report instead of stdout.",
    )
    return parser.parse_args()


def main_cli() -> int:
    args = _parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
