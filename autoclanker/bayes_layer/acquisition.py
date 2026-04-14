from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence

import numpy

from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.feature_encoder import strategy_materialization_groups
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.surrogate_feasibility import predict_valid_probability
from autoclanker.bayes_layer.surrogate_objective import encode_and_score_candidate
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    EncodedCandidate,
    FeasibilityPosterior,
    FrontierCandidate,
    GeneStateRef,
    ObjectivePosterior,
    RankedCandidate,
)

_THOMPSON_DRAW_COUNT = 32
_OPTIMISM_WEIGHT = 0.5
_SAMPLE_JITTER_STEPS = (0.0, 1.0e-10, 1.0e-8, 1.0e-6)


def generate_candidate_pool(
    registry: GeneRegistry,
    *,
    compiled_priors: CompiledPriorBundle,
    limit: int = 24,
) -> tuple[tuple[str, tuple[GeneStateRef, ...]], ...]:
    default_genotype = list(registry.default_genotype(materializable_only=True))
    default_by_gene = {ref.gene_id: ref for ref in default_genotype}
    candidates: list[tuple[str, tuple[GeneStateRef, ...]]] = []
    seen: set[tuple[str, ...]] = set()

    def add_candidate(genotype: Sequence[GeneStateRef]) -> None:
        ordered = tuple(sorted(genotype, key=registry.sort_key))
        signature = tuple(ref.canonical_key for ref in ordered)
        if signature in seen or len(candidates) >= limit:
            return
        seen.add(signature)
        candidates.append((f"cand_auto_{len(candidates) + 1:03d}", ordered))

    add_candidate(default_genotype)
    for gene_id in registry.materializable_gene_ids():
        definition = registry.genes[gene_id]
        for state_id in definition.states:
            if state_id == definition.default_state:
                continue
            genotype = dict(default_by_gene)
            genotype[gene_id] = GeneStateRef(gene_id=gene_id, state_id=state_id)
            add_candidate(tuple(genotype.values()))
    for spec in compiled_priors.pair_priors:
        pair_body = spec.item.target_ref.removeprefix("pair:")
        first, second = pair_body.split("+", maxsplit=1)
        first_gene, first_state = first.split("=", maxsplit=1)
        second_gene, second_state = second.split("=", maxsplit=1)
        if (
            first_gene not in default_by_gene
            or second_gene not in default_by_gene
            or not registry.genes[first_gene].materializable
            or not registry.genes[second_gene].materializable
        ):
            continue
        genotype = dict(default_by_gene)
        genotype[first_gene] = GeneStateRef(gene_id=first_gene, state_id=first_state)
        genotype[second_gene] = GeneStateRef(gene_id=second_gene, state_id=second_state)
        add_candidate(tuple(genotype.values()))
    for required_main_refs in strategy_materialization_groups(registry):
        genotype = dict(default_by_gene)
        valid_group = True
        for target_ref in required_main_refs:
            ref_body = target_ref.removeprefix("main:")
            if "=" not in ref_body:
                valid_group = False
                break
            gene_id, state_id = ref_body.split("=", maxsplit=1)
            definition = registry.genes.get(gene_id)
            if definition is None or not definition.materializable:
                valid_group = False
                break
            genotype[gene_id] = GeneStateRef(gene_id=gene_id, state_id=state_id)
        if valid_group:
            add_candidate(tuple(genotype.values()))
    return tuple(candidates[:limit])


def _candidate_influence_summary(
    encoded: EncodedCandidate,
    *,
    registry: GeneRegistry,
    feasibility_posterior: FeasibilityPosterior,
    compiled_priors: CompiledPriorBundle | None,
) -> tuple[str, ...]:
    if compiled_priors is None:
        return ()
    active_refs = set(encoded.main_effects) | set(encoded.pair_effects)
    active_refs.add(registry.pattern_ref(encoded.genotype))
    active_refs.update(registry.vram_ref(ref) for ref in encoded.genotype)
    active_refs.update(
        registry.feasibility_ref(ref, failure_mode)
        for ref in encoded.genotype
        for failure_mode in feasibility_posterior.failure_mode_biases
    )
    summaries: list[str] = []
    for spec in compiled_priors.all_items:
        if spec.item.target_ref not in active_refs:
            continue
        summaries.append(f"{spec.source_belief_id} -> {spec.item.target_ref}")
        if len(summaries) >= 4:
            break
    return tuple(summaries)


def _objective_feature_index(posterior: ObjectivePosterior) -> dict[str, int]:
    return {
        feature_name: index
        for index, feature_name in enumerate(posterior.feature_order)
    }


def _objective_pair_whitelist(posterior: ObjectivePosterior) -> set[str]:
    return {
        feature_name
        for feature_name in posterior.feature_order
        if feature_name.startswith("pair:")
    }


def _encoded_feature_vector(
    encoded: EncodedCandidate,
    *,
    posterior: ObjectivePosterior,
) -> numpy.ndarray:
    vector = numpy.zeros(len(posterior.feature_order), dtype=float)
    feature_index = _objective_feature_index(posterior)
    for feature_name in encoded.main_effects:
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = 1.0
    for feature_name in encoded.pair_effects:
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = 1.0
    for global_name, value in encoded.global_features.items():
        feature_name = f"global:{global_name}"
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = float(value)
    return vector


def _sample_factor(covariance: numpy.ndarray) -> numpy.ndarray | None:
    identity = numpy.eye(covariance.shape[0], dtype=float)
    for jitter in _SAMPLE_JITTER_STEPS:
        try:
            return numpy.linalg.cholesky(covariance + (jitter * identity))
        except numpy.linalg.LinAlgError:
            continue
    return None


def _thompson_seed(
    objective_posterior: ObjectivePosterior,
    candidate_ids: Sequence[str],
) -> int:
    digest = hashlib.sha256()
    digest.update(objective_posterior.era_id.encode("utf-8"))
    digest.update(objective_posterior.backend.encode("utf-8"))
    for candidate_id in candidate_ids:
        digest.update(candidate_id.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _thompson_scores(
    encoded_candidates: Sequence[EncodedCandidate],
    *,
    objective_posterior: ObjectivePosterior,
    valid_probabilities: Sequence[float],
) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    tuple[float, ...] | None,
]:
    if (
        objective_posterior.backend != "exact_joint_linear"
        or not objective_posterior.sampleable
        or not objective_posterior.posterior_covariance
        or not objective_posterior.posterior_mean_vector
        or not objective_posterior.feature_order
    ):
        return (None, None, None, None, None)

    covariance = numpy.asarray(objective_posterior.posterior_covariance, dtype=float)
    mean_vector = numpy.asarray(objective_posterior.posterior_mean_vector, dtype=float)
    factor = _sample_factor(covariance)
    if factor is None:
        return (None, None, None, None, None)

    candidate_matrix = numpy.vstack(
        [
            _encoded_feature_vector(candidate, posterior=objective_posterior)
            for candidate in encoded_candidates
        ]
    )
    rng = numpy.random.default_rng(
        _thompson_seed(
            objective_posterior,
            [candidate.candidate_id for candidate in encoded_candidates],
        )
    )
    draws = rng.standard_normal(
        (mean_vector.shape[0], _THOMPSON_DRAW_COUNT),
        dtype=float,
    )
    sampled_coefficients = mean_vector[:, None] + (factor @ draws)
    sampled_utilities = objective_posterior.baseline_utility + (
        candidate_matrix @ sampled_coefficients
    )
    valid_probability_vector = numpy.asarray(valid_probabilities, dtype=float)[:, None]
    sampled_scores = valid_probability_vector * sampled_utilities
    sampled_score_mean = sampled_scores.mean(axis=1)
    sampled_score_std = sampled_scores.std(axis=1)
    primary_scores = sampled_scores[:, 0]
    winner_indices = sampled_scores.argmax(axis=0)
    posterior_win_rate = numpy.array(
        [
            float((winner_indices == index).sum()) / float(_THOMPSON_DRAW_COUNT)
            for index in range(sampled_scores.shape[0])
        ],
        dtype=float,
    )
    sampled_utility = sampled_utilities[:, 0]
    return (
        tuple(float(item) for item in primary_scores),
        tuple(float(item) for item in sampled_utility),
        tuple(float(item) for item in sampled_score_mean),
        tuple(float(item) for item in sampled_score_std),
        tuple(float(item) for item in posterior_win_rate),
    )


def _thompson_fallback_reason(
    objective_posterior: ObjectivePosterior,
) -> str:
    if objective_posterior.observation_count <= 0:
        return "insufficient_observations_for_sampled_acquisition"
    if objective_posterior.backend != "exact_joint_linear":
        return "objective_backend_not_exact_joint_linear"
    if not objective_posterior.sampleable:
        return (
            objective_posterior.fallback_reason or "objective_posterior_not_sampleable"
        )
    if (
        not objective_posterior.posterior_covariance
        or not objective_posterior.posterior_mean_vector
        or not objective_posterior.feature_order
    ):
        return "objective_sampling_metadata_missing"
    return "sampling_factorization_failed"


def _optimistic_scores(
    predicted_utilities: Sequence[float],
    uncertainties: Sequence[float],
    valid_probabilities: Sequence[float],
) -> tuple[float, ...]:
    return tuple(
        float(valid_probability * (predicted + (_OPTIMISM_WEIGHT * uncertainty)))
        for predicted, uncertainty, valid_probability in zip(
            predicted_utilities,
            uncertainties,
            valid_probabilities,
            strict=True,
        )
    )


def rank_candidates(
    candidates: Sequence[tuple[str, Sequence[GeneStateRef]]],
    *,
    registry: GeneRegistry,
    objective_posterior: ObjectivePosterior,
    feasibility_posterior: FeasibilityPosterior,
    compiled_priors: CompiledPriorBundle | None = None,
    frontier_candidates: Mapping[str, FrontierCandidate] | None = None,
    config: BayesLayerConfig | None = None,
) -> tuple[RankedCandidate, ...]:
    active_config = config or load_bayes_layer_config()
    pair_whitelist = _objective_pair_whitelist(objective_posterior) or {
        feature.feature_name
        for feature in objective_posterior.features
        if feature.target_kind == "pair_effect"
    }

    encoded_candidates: list[EncodedCandidate] = []
    predicted_utilities: list[float] = []
    uncertainties: list[float] = []
    valid_probabilities: list[float] = []
    for candidate_id, genotype in candidates:
        encoded, predicted_utility, uncertainty = encode_and_score_candidate(
            candidate_id,
            genotype,
            registry=registry,
            posterior=objective_posterior,
            pair_whitelist=pair_whitelist or None,
        )
        encoded_candidates.append(encoded)
        predicted_utilities.append(predicted_utility)
        uncertainties.append(uncertainty)
        valid_probabilities.append(
            predict_valid_probability(encoded, posterior=feasibility_posterior)
        )

    sampled_payload = _thompson_scores(
        encoded_candidates,
        objective_posterior=objective_posterior,
        valid_probabilities=valid_probabilities,
    )
    (
        sampled_scores,
        sampled_utilities,
        sampled_score_mean,
        sampled_score_std,
        win_rates,
    ) = sampled_payload
    optimistic_scores = _optimistic_scores(
        predicted_utilities,
        uncertainties,
        valid_probabilities,
    )
    requested_thompson = (
        active_config.acquisition.kind == "constrained_thompson_sampling"
    )
    use_thompson = (
        requested_thompson
        and objective_posterior.observation_count > 0
        and sampled_scores is not None
        and sampled_utilities is not None
        and sampled_score_mean is not None
        and sampled_score_std is not None
        and win_rates is not None
    )
    acquisition_fallback_reason = (
        None
        if use_thompson or not requested_thompson
        else _thompson_fallback_reason(objective_posterior)
    )

    ranked: list[RankedCandidate] = []
    for index, encoded in enumerate(encoded_candidates):
        frontier_candidate = (
            None
            if frontier_candidates is None
            else frontier_candidates.get(encoded.candidate_id)
        )
        predicted_utility = predicted_utilities[index]
        uncertainty = uncertainties[index]
        valid_probability = valid_probabilities[index]
        acquisition_score = (
            sampled_scores[index]
            if use_thompson and sampled_scores is not None
            else optimistic_scores[index]
        )
        rationale: list[str] = []
        if valid_probability >= 0.8:
            rationale.append("high feasibility")
        if uncertainty >= 0.6:
            rationale.append("informative uncertainty")
        if predicted_utility >= objective_posterior.baseline_utility:
            rationale.append("above incumbent baseline")
        if use_thompson:
            rationale.append("ranked by sampled finite-pool posterior draw")
        elif acquisition_fallback_reason is not None:
            rationale.append(
                f"sampled acquisition unavailable: {acquisition_fallback_reason}"
            )
        ranked.append(
            RankedCandidate(
                candidate_id=encoded.candidate_id,
                genotype=encoded.genotype,
                predicted_utility=predicted_utility,
                uncertainty=uncertainty,
                valid_probability=valid_probability,
                acquisition_score=acquisition_score,
                objective_backend=objective_posterior.backend,
                acquisition_backend=(
                    "constrained_thompson_sampling"
                    if use_thompson
                    else "optimistic_upper_confidence"
                ),
                acquisition_fallback_reason=acquisition_fallback_reason,
                sampled_utility=(
                    None
                    if sampled_utilities is None or not use_thompson
                    else sampled_utilities[index]
                ),
                sampled_score_mean=(
                    None
                    if sampled_score_mean is None or not use_thompson
                    else sampled_score_mean[index]
                ),
                sampled_score_std=(
                    None
                    if sampled_score_std is None or not use_thompson
                    else sampled_score_std[index]
                ),
                posterior_win_rate=(
                    None if win_rates is None or not use_thompson else win_rates[index]
                ),
                rationale=tuple(rationale),
                influence_summary=_candidate_influence_summary(
                    encoded,
                    registry=registry,
                    feasibility_posterior=feasibility_posterior,
                    compiled_priors=compiled_priors,
                ),
                family_id=(
                    None if frontier_candidate is None else frontier_candidate.family_id
                ),
                origin_kind=(
                    None
                    if frontier_candidate is None
                    else frontier_candidate.origin_kind
                ),
                parent_candidate_ids=(
                    ()
                    if frontier_candidate is None
                    else frontier_candidate.parent_candidate_ids
                ),
                parent_belief_ids=(
                    ()
                    if frontier_candidate is None
                    else frontier_candidate.parent_belief_ids
                ),
                origin_query_ids=(
                    ()
                    if frontier_candidate is None
                    else frontier_candidate.origin_query_ids
                ),
                notes=None if frontier_candidate is None else frontier_candidate.notes,
                budget_weight=(
                    None
                    if frontier_candidate is None
                    else frontier_candidate.budget_weight
                ),
            )
        )

    return tuple(
        sorted(
            ranked,
            key=lambda candidate: (
                -candidate.acquisition_score,
                -candidate.valid_probability,
                -candidate.predicted_utility,
                candidate.candidate_id,
            ),
        )
    )
