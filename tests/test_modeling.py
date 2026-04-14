from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pytest import MonkeyPatch

from autoclanker.bayes_layer import (
    EraState,
    acquisition as acquisition_module,
    compile_beliefs,
    ingest_human_beliefs,
    load_serialized_payload,
    surrogate_objective as objective_module,
)
from autoclanker.bayes_layer.acquisition import generate_candidate_pool, rank_candidates
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.commit_policy import recommend_commit
from autoclanker.bayes_layer.config import load_bayes_layer_config
from autoclanker.bayes_layer.feature_encoder import aggregate_eval_results
from autoclanker.bayes_layer.posterior_graph import build_posterior_graph
from autoclanker.bayes_layer.query_policy import suggest_queries
from autoclanker.bayes_layer.surrogate_feasibility import fit_feasibility_surrogate
from autoclanker.bayes_layer.surrogate_objective import fit_objective_surrogate
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    FeasibilityPosterior,
    GeneStateRef,
    ObjectivePosterior,
    PosteriorFeature,
    RankedCandidate,
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


def _empty_bundle(*, era_id: str = "era_003") -> CompiledPriorBundle:
    return CompiledPriorBundle(
        era_id=era_id,
        main_effect_priors=(),
        pair_priors=(),
        feasibility_priors=(),
        vram_priors=(),
        hard_masks=(),
        preference_observations=(),
        candidate_generation_hints=(),
        linkage_hints=(),
        belief_previews=(),
    )


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
    assert objective.backend == "exact_joint_linear"
    assert objective.sampleable is True
    assert ranked
    assert ranked[0].candidate_id.startswith("cand_auto_")
    assert ranked[0].acquisition_backend == "constrained_thompson_sampling"
    assert queries
    assert queries[0].query_type == "pairwise_preference"
    assert queries[0].candidate_ids
    assert decision.session_id == "session_demo"
    assert graph.edges


@covers("M2-002")
def test_exact_objective_falls_back_to_heuristic_when_no_explicit_features_exist() -> (
    None
):
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()

    objective = fit_objective_surrogate(
        (),
        registry=registry,
        compiled_priors=_empty_bundle(),
        era_state=EraState(era_id="era_zero_features"),
    )

    assert objective.backend == "heuristic_independent_normal"
    assert objective.sampleable is False
    assert objective.feature_order == ()
    assert objective.features == ()
    assert objective.fallback_reason == "exact_joint_linear_no_features"


@covers("M3-003")
def test_generate_candidate_pool_skips_invalid_strategy_materialization_refs(
    monkeypatch: MonkeyPatch,
) -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()

    def _strategy_groups(
        _registry: object,
    ) -> tuple[tuple[str, ...], ...]:
        return (
            ("main:parser.matcher=matcher_compiled", "main:broken"),
            ("main:missing_state",),
            ("main:parser.matcher=matcher_compiled",),
        )

    monkeypatch.setattr(
        acquisition_module,
        "strategy_materialization_groups",
        _strategy_groups,
    )

    pool = generate_candidate_pool(registry, compiled_priors=_empty_bundle())
    candidate_ids = {
        tuple(ref.canonical_key for ref in genotype) for _candidate_id, genotype in pool
    }

    assert candidate_ids
    assert len(candidate_ids) == len(pool)


@covers("M4-001")
def test_query_policy_handles_disabled_and_non_pairwise_paths() -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()
    beliefs = ingest_human_beliefs(
        load_serialized_payload(ROOT / "examples/human_beliefs/expert_session.json")
    )
    objective = fit_objective_surrogate(
        (),
        registry=registry,
        compiled_priors=_empty_bundle(era_id=beliefs.session_context.era_id),
        era_state=EraState(era_id=beliefs.session_context.era_id),
    )
    config = load_bayes_layer_config()
    disabled_config = replace(
        config,
        query_policy=replace(config.query_policy, enabled=False),
    )

    assert suggest_queries(objective, beliefs=beliefs, config=disabled_config) == ()
    assert suggest_queries(objective, beliefs=beliefs, ranked_candidates=()) == ()


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


def _always_missing_cholesky(*args: object, **kwargs: object) -> None:
    del args, kwargs
    return None


def _always_missing_sample_factor(*args: object, **kwargs: object) -> None:
    del args, kwargs
    return None


def _sampled_draw_best_scores(
    encoded_candidates: object,
    *,
    objective_posterior: object,
    valid_probabilities: object,
) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
]:
    del encoded_candidates, objective_posterior, valid_probabilities
    return (
        (0.25, 0.9),
        (0.25, 0.9),
        (0.95, 0.1),
        (0.05, 0.05),
        (0.05, 0.95),
    )


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

    assert objective.backend == "exact_joint_linear"
    assert ranked[0].acquisition_backend == "optimistic_upper_confidence"
    assert ranked[0].acquisition_fallback_reason == (
        "insufficient_observations_for_sampled_acquisition"
    )
    assert ranked[0].candidate_id == "cand_c_compiled_context_pair"
    assert (
        ranked_by_id["cand_c_compiled_context_pair"].predicted_utility
        > ranked_by_id["cand_b_compiled_matcher"].predicted_utility
    )
    assert (
        ranked_by_id["cand_d_wide_window_large_chunk"].valid_probability
        < ranked_by_id["cand_a_default"].valid_probability
    )


def test_objective_surrogate_falls_back_to_heuristic_when_exact_fit_is_unsafe(
    monkeypatch: MonkeyPatch,
) -> None:
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
    )

    monkeypatch.setattr(
        objective_module,
        "_cholesky_factor",
        _always_missing_cholesky,
    )
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(
            era_id=beliefs.session_context.era_id,
            observation_count=len(observations),
        ),
    )

    assert objective.backend == "heuristic_independent_normal"
    assert objective.fallback_reason is not None


def test_rank_candidates_uses_sampled_draw_order_for_thompson_selection(
    monkeypatch: MonkeyPatch,
) -> None:
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
            candidate_id="pair",
            genotype=_candidate_with_overrides(
                adapter,
                optim_lr="matcher_compiled",
                model_depth="plan_context_pair",
            ),
            seed=2,
        ),
    )
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(era_id=beliefs.session_context.era_id),
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=EraState(era_id=beliefs.session_context.era_id),
    )
    candidate_pool = (
        ("cand_mean_best", registry.default_genotype()),
        (
            "cand_draw_best",
            _candidate_with_overrides(
                adapter,
                optim_lr="matcher_compiled",
                model_depth="plan_context_pair",
            ),
        ),
    )
    assert objective.backend == "exact_joint_linear"
    assert objective.sampleable is True

    monkeypatch.setattr(
        acquisition_module,
        "_thompson_scores",
        _sampled_draw_best_scores,
    )

    ranked = rank_candidates(
        candidate_pool,
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        compiled_priors=compiled,
    )

    assert ranked[0].candidate_id == "cand_draw_best"
    assert ranked[0].acquisition_score == 0.9
    assert ranked[0].sampled_score_mean == 0.1


def test_acquisition_falls_back_when_sampling_is_unavailable(
    monkeypatch: MonkeyPatch,
) -> None:
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
    )
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

    monkeypatch.setattr(
        acquisition_module,
        "_sample_factor",
        _always_missing_sample_factor,
    )
    ranked = rank_candidates(
        generate_candidate_pool(registry, compiled_priors=compiled),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        compiled_priors=compiled,
    )

    assert objective.backend == "exact_joint_linear"
    assert objective.sampleable is True
    assert ranked[0].acquisition_backend == "optimistic_upper_confidence"
    assert ranked[0].acquisition_fallback_reason == "sampling_factorization_failed"
    assert ranked[0].sampled_utility is None


def test_query_policy_prefers_candidate_scope_when_uncertainty_is_local_within_family() -> (
    None
):
    beliefs = ingest_human_beliefs(
        load_serialized_payload(ROOT / "examples/human_beliefs/expert_session.json")
    )
    objective = ObjectivePosterior(
        era_id=beliefs.session_context.era_id,
        baseline_utility=0.0,
        features=(),
        observation_count=0,
    )
    ranked = (
        RankedCandidate(
            candidate_id="cand_a",
            genotype=(),
            predicted_utility=1.0,
            uncertainty=0.7,
            valid_probability=0.95,
            acquisition_score=1.0,
            family_id="family_alpha",
        ),
        RankedCandidate(
            candidate_id="cand_b",
            genotype=(),
            predicted_utility=0.98,
            uncertainty=0.6,
            valid_probability=0.95,
            acquisition_score=0.96,
            family_id="family_alpha",
        ),
        RankedCandidate(
            candidate_id="cand_c",
            genotype=(),
            predicted_utility=0.5,
            uncertainty=0.1,
            valid_probability=0.95,
            acquisition_score=0.35,
            family_id="family_beta",
        ),
    )

    queries = suggest_queries(objective, beliefs=beliefs, ranked_candidates=ranked)

    assert queries[0].comparison_scope == "candidate"
    assert queries[0].candidate_ids == ("cand_a", "cand_b")


def test_query_policy_prefers_family_scope_when_cross_family_uncertainty_is_localized() -> (
    None
):
    beliefs = ingest_human_beliefs(
        load_serialized_payload(ROOT / "examples/human_beliefs/expert_session.json")
    )
    objective = ObjectivePosterior(
        era_id=beliefs.session_context.era_id,
        baseline_utility=0.0,
        features=(),
        observation_count=0,
    )
    ranked = (
        RankedCandidate(
            candidate_id="cand_a",
            genotype=(),
            predicted_utility=1.0,
            uncertainty=0.55,
            valid_probability=0.95,
            acquisition_score=1.0,
            family_id="family_alpha",
        ),
        RankedCandidate(
            candidate_id="cand_same_family",
            genotype=(),
            predicted_utility=0.8,
            uncertainty=0.1,
            valid_probability=0.95,
            acquisition_score=0.4,
            family_id="family_alpha",
        ),
        RankedCandidate(
            candidate_id="cand_family_beta",
            genotype=(),
            predicted_utility=0.95,
            uncertainty=0.6,
            valid_probability=0.95,
            acquisition_score=0.96,
            family_id="family_beta",
        ),
    )

    queries = suggest_queries(objective, beliefs=beliefs, ranked_candidates=ranked)

    assert queries[0].comparison_scope == "family"
    assert queries[0].candidate_ids == ("cand_a", "cand_family_beta")
    assert queries[0].family_ids == ("family_alpha", "family_beta")


def test_thompson_fallback_reason_variants() -> None:
    adapter = _make_fixture_adapter()
    registry = adapter.build_registry()
    candidate_pool = (("cand_reason", registry.default_genotype()),)
    feasibility = FeasibilityPosterior(
        era_id="era_reason",
        baseline_valid_probability=1.0,
        features=(),
        failure_mode_biases={},
        observation_count=0,
    )

    def _fallback_reason_for(objective: ObjectivePosterior) -> str | None:
        ranked = rank_candidates(
            candidate_pool,
            registry=registry,
            objective_posterior=objective,
            feasibility_posterior=feasibility,
        )
        return ranked[0].acquisition_fallback_reason

    assert (
        _fallback_reason_for(
            ObjectivePosterior(
                era_id="era_reason",
                baseline_utility=0.0,
                features=(),
                observation_count=1,
                backend="heuristic_independent_normal",
            )
        )
        == "objective_backend_not_exact_joint_linear"
    )
    assert (
        _fallback_reason_for(
            ObjectivePosterior(
                era_id="era_reason",
                baseline_utility=0.0,
                features=(),
                observation_count=1,
                backend="exact_joint_linear",
                sampleable=False,
                fallback_reason="exact_fit_abandoned",
            )
        )
        == "exact_fit_abandoned"
    )
    assert (
        _fallback_reason_for(
            ObjectivePosterior(
                era_id="era_reason",
                baseline_utility=0.0,
                features=(),
                observation_count=1,
                backend="exact_joint_linear",
                sampleable=True,
                feature_order=(),
                posterior_mean_vector=(0.0,),
                posterior_covariance=((1.0,),),
            )
        )
        == "objective_sampling_metadata_missing"
    )
