from __future__ import annotations

from pathlib import Path

from autoclanker.bayes_layer import (
    EraState,
    compile_beliefs,
    ingest_human_beliefs,
    load_serialized_payload,
)
from autoclanker.bayes_layer.acquisition import generate_candidate_pool, rank_candidates
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.commit_policy import recommend_commit
from autoclanker.bayes_layer.feature_encoder import aggregate_eval_results
from autoclanker.bayes_layer.posterior_graph import build_posterior_graph
from autoclanker.bayes_layer.query_policy import suggest_queries
from autoclanker.bayes_layer.surrogate_feasibility import fit_feasibility_surrogate
from autoclanker.bayes_layer.surrogate_objective import fit_objective_surrogate
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    PosteriorFeature,
    ValidAdapterConfig,
)
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _make_fixture_adapter() -> FixtureAdapter:
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
    optim_lr: str | None = None,
    model_depth: str | None = None,
    model_width: str | None = None,
    batch_size: str | None = None,
) -> tuple[GeneStateRef, ...]:
    registry = adapter.build_registry()
    genotype = {ref.gene_id: ref for ref in registry.default_genotype()}
    if optim_lr is not None:
        genotype["parser.matcher"] = GeneStateRef(
            gene_id="parser.matcher", state_id=optim_lr
        )
    if model_depth is not None:
        genotype["parser.plan"] = GeneStateRef(
            gene_id="parser.plan",
            state_id=model_depth,
        )
    if model_width is not None:
        genotype["capture.window"] = GeneStateRef(
            gene_id="capture.window",
            state_id=model_width,
        )
    if batch_size is not None:
        genotype["io.chunk"] = GeneStateRef(
            gene_id="io.chunk",
            state_id=batch_size,
        )
    return tuple(genotype.values())


@covers("M2-001", "M4-001")
def test_modeling_stack_scores_candidates_and_recommends_commit() -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()
    beliefs = ingest_human_beliefs(
        load_serialized_payload(ROOT / "examples/human_beliefs/expert_session.json")
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
            candidate_id="lr_depth_a",
            genotype=_candidate_with_overrides(
                adapter,
                optim_lr="matcher_compiled",
                model_depth="plan_context_pair",
            ),
            seed=2,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="lr_depth_b",
            genotype=_candidate_with_overrides(
                adapter,
                optim_lr="matcher_compiled",
                model_depth="plan_context_pair",
            ),
            seed=3,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="wide_batch",
            genotype=_candidate_with_overrides(
                adapter,
                model_width="window_wide",
                batch_size="chunk_large",
            ),
            seed=4,
        ),
    )

    aggregates = aggregate_eval_results(observations, registry=registry)
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(
            era_id=beliefs.session_context.era_id,
            observation_count=len(observations),
        ),
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(
            era_id=beliefs.session_context.era_id,
            observation_count=len(observations),
        ),
    )
    ranked = rank_candidates(
        generate_candidate_pool(registry, compiled_priors=compiled),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
    )
    queries = suggest_queries(
        objective,
        beliefs=beliefs,
        ranked_candidates=ranked,
    )
    decision = recommend_commit(
        session_id="session_demo",
        era_id="era_003",
        ranked_candidates=ranked,
        observations=observations,
        registry=registry,
    )
    graph = build_posterior_graph(objective, compiled)

    assert any(aggregate.count > 1 for aggregate in aggregates)
    assert ranked
    assert ranked[0].candidate_id.startswith("cand_auto_")
    assert queries
    assert decision.session_id == "session_demo"
    assert graph.edges


def _expert_prior_batch(
    *,
    mean: float,
    scale: float,
    observation_weight: float,
    failure_mean: float,
    per_eval_multiplier: float,
    cross_era_transfer: float,
) -> dict[str, object]:
    return {
        "session_context": {
            "era_id": "era_003",
            "session_id": "session_decay",
            "user_profile": "expert",
        },
        "beliefs": [
            {
                "id": "prior_objective",
                "kind": "expert_prior",
                "confidence_level": 4,
                "target": {
                    "target_kind": "main_effect",
                    "gene": {
                        "gene_id": "parser.matcher",
                        "state_id": "matcher_compiled",
                    },
                },
                "prior_family": "normal",
                "mean": mean,
                "scale": scale,
                "observation_weight": observation_weight,
                "decay_override": {
                    "per_eval_multiplier": per_eval_multiplier,
                    "cross_era_transfer": cross_era_transfer,
                },
            },
            {
                "id": "prior_feasibility",
                "kind": "expert_prior",
                "confidence_level": 4,
                "target": {
                    "target_kind": "feasibility_logit",
                    "gene": {"gene_id": "capture.window", "state_id": "window_wide"},
                    "failure_mode": "oom",
                },
                "prior_family": "logit_normal",
                "mean": failure_mean,
                "scale": scale,
                "observation_weight": observation_weight,
                "decay_override": {
                    "per_eval_multiplier": per_eval_multiplier,
                    "cross_era_transfer": cross_era_transfer,
                },
            },
        ],
    }


def _feature_mean(features: tuple[PosteriorFeature, ...], feature_name: str) -> float:
    for feature in features:
        if feature.feature_name == feature_name:
            return float(feature.posterior_mean)
    raise AssertionError(f"Missing feature {feature_name!r}")


@covers("M2-002")
def test_runtime_prior_decay_and_observation_weight_shape_posteriors() -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()
    observations = (
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="lr",
            genotype=_candidate_with_overrides(adapter, optim_lr="matcher_compiled"),
            seed=1,
        ),
        adapter.evaluate_candidate(
            era_id="era_003",
            candidate_id="width",
            genotype=_candidate_with_overrides(adapter, model_width="window_wide"),
            seed=2,
        ),
    )

    weighted_batch = ingest_human_beliefs(
        _expert_prior_batch(
            mean=2.4,
            scale=0.45,
            observation_weight=4.0,
            failure_mean=2.0,
            per_eval_multiplier=0.5,
            cross_era_transfer=0.25,
        )
    )
    light_batch = ingest_human_beliefs(
        _expert_prior_batch(
            mean=2.4,
            scale=0.45,
            observation_weight=1.0,
            failure_mean=2.0,
            per_eval_multiplier=0.5,
            cross_era_transfer=0.25,
        )
    )
    weighted_compiled = compile_beliefs(
        weighted_batch,
        registry,
        EraState(era_id="era_003"),
    )
    light_compiled = compile_beliefs(
        light_batch,
        registry,
        EraState(era_id="era_003"),
    )

    weighted_same_era = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=weighted_compiled,
        era_state=EraState(era_id="era_003", observation_count=0),
    )
    light_same_era = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=light_compiled,
        era_state=EraState(era_id="era_003", observation_count=0),
    )
    decayed_same_era = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=weighted_compiled,
        era_state=EraState(era_id="era_003", observation_count=6),
    )
    cross_era = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=weighted_compiled,
        era_state=EraState(era_id="era_004", observation_count=0),
    )
    weighted_feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=weighted_compiled,
        era_state=EraState(era_id="era_003", observation_count=0),
    )
    decayed_feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=weighted_compiled,
        era_state=EraState(era_id="era_004", observation_count=6),
    )

    feature_name = "main:parser.matcher=matcher_compiled"
    assert _feature_mean(
        weighted_same_era.features,
        feature_name,
    ) > _feature_mean(light_same_era.features, feature_name)
    assert _feature_mean(
        decayed_same_era.features,
        feature_name,
    ) < _feature_mean(weighted_same_era.features, feature_name)
    assert _feature_mean(
        cross_era.features,
        feature_name,
    ) < _feature_mean(weighted_same_era.features, feature_name)
    assert (
        weighted_feasibility.failure_mode_biases["oom"]
        > decayed_feasibility.failure_mode_biases["oom"]
    )


@covers("M2-002")
def test_cold_start_priors_shape_showcase_candidate_ranking() -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()
    beliefs = ingest_human_beliefs(
        load_serialized_payload(
            ROOT / "examples" / "live_exercises" / "bayes_complex" / "beliefs.yaml"
        )
    )
    compiled = compile_beliefs(
        beliefs,
        registry,
        EraState(era_id=beliefs.session_context.era_id),
    )

    objective = fit_objective_surrogate(
        (),
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(era_id=beliefs.session_context.era_id),
    )
    feasibility = fit_feasibility_surrogate(
        (),
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(era_id=beliefs.session_context.era_id),
    )
    ranked = rank_candidates(
        (
            ("cand_a_default", registry.default_genotype()),
            (
                "cand_b_compiled_matcher",
                _candidate_with_overrides(adapter, optim_lr="matcher_compiled"),
            ),
            (
                "cand_c_compiled_context_pair",
                _candidate_with_overrides(
                    adapter,
                    optim_lr="matcher_compiled",
                    model_depth="plan_context_pair",
                ),
            ),
            (
                "cand_d_wide_window_large_chunk",
                _candidate_with_overrides(
                    adapter,
                    model_width="window_wide",
                    batch_size="chunk_large",
                ),
            ),
            (
                "cand_e_jit_full_scan",
                _candidate_with_overrides(
                    adapter,
                    optim_lr="matcher_jit",
                    model_depth="plan_full_scan",
                ),
            ),
        ),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
    )
    ranked_by_id = {candidate.candidate_id: candidate for candidate in ranked}

    assert ranked[0].candidate_id == "cand_c_compiled_context_pair"
    assert (
        ranked_by_id["cand_c_compiled_context_pair"].predicted_utility
        > ranked_by_id["cand_b_compiled_matcher"].predicted_utility
    )
    assert (
        ranked_by_id["cand_d_wide_window_large_chunk"].valid_probability
        < ranked_by_id["cand_a_default"].valid_probability
    )
