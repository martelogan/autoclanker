from __future__ import annotations

import math
import re

from collections import Counter
from collections.abc import Sequence
from statistics import fmean

import numpy

from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.feature_encoder import (
    aggregate_eval_results,
    encode_candidate,
    iter_feature_memberships,
)
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    EncodedCandidate,
    EraState,
    GeneStateRef,
    ObjectivePosterior,
    PosteriorFeature,
    ValidEvalResult,
)

_DEFAULT_PRIOR_VARIANCE = 1.5**2
_MIN_PRIOR_VARIANCE = 0.05
_MIN_OBSERVATION_NOISE = 0.25
_BASE_PREDICTION_VARIANCE = 0.05
_FIT_JITTER_STEPS = (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
_SAMPLE_JITTER_STEPS = (0.0, 1.0e-10, 1.0e-8, 1.0e-6)
_MAX_CONDITION_NUMBER = 1.0e9


def _era_distance(source_era_id: str, target_era_id: str) -> int:
    if source_era_id == target_era_id:
        return 0
    pattern = re.compile(r"era_(\d+)$")
    source_match = pattern.fullmatch(source_era_id)
    target_match = pattern.fullmatch(target_era_id)
    if source_match is None or target_match is None:
        return 1
    return abs(int(source_match.group(1)) - int(target_match.group(1)))


def _prior_lookup(
    bundle: CompiledPriorBundle,
    *,
    era_state: EraState,
) -> dict[str, tuple[float, float, float]]:
    weighted_means: dict[str, float] = {}
    precision_totals: dict[str, float] = {}
    prior_variances: dict[str, float] = {}
    for spec in bundle.main_effect_priors + bundle.pair_priors:
        variance = max(spec.item.scale**2, _MIN_PRIOR_VARIANCE)
        base_precision = (spec.item.observation_weight or 1.0) / variance
        decay_factor = 1.0
        if spec.item.decay is not None:
            decay_factor *= (
                spec.item.decay.per_eval_multiplier**era_state.observation_count
            )
            if bundle.era_id != era_state.era_id:
                decay_factor *= spec.item.decay.cross_era_transfer ** _era_distance(
                    bundle.era_id,
                    era_state.era_id,
                )
        effective_precision = base_precision * decay_factor
        weighted_means.setdefault(spec.item.target_ref, 0.0)
        precision_totals.setdefault(spec.item.target_ref, 0.0)
        prior_variances.setdefault(spec.item.target_ref, variance)
        weighted_means[spec.item.target_ref] += spec.item.mean * effective_precision
        precision_totals[spec.item.target_ref] += effective_precision
    return {
        target_ref: (
            weighted_means[target_ref] / precision_totals[target_ref],
            precision_totals[target_ref],
            prior_variances[target_ref],
        )
        for target_ref in weighted_means
        if precision_totals[target_ref] > 0.0
    }


def _target_kind_for_feature(feature_name: str) -> str:
    if feature_name.startswith("pair:"):
        return "pair_effect"
    if feature_name.startswith("global:"):
        return "global_feature"
    return "main_effect"


def _pair_whitelist(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    compiled_priors: CompiledPriorBundle,
    config: BayesLayerConfig,
) -> set[str]:
    aggregates = aggregate_eval_results(observations, registry=registry)
    pair_support = Counter[str]()
    for _aggregate, encoded in iter_feature_memberships(aggregates, registry=registry):
        pair_support.update(encoded.pair_effects)
    whitelist = {
        name
        for name, _count in pair_support.most_common(
            config.objective_surrogate.max_pair_features,
        )
    }
    for spec in compiled_priors.pair_priors + compiled_priors.linkage_hints:
        whitelist.add(spec.item.target_ref)
    return whitelist


def _feature_prior(
    prior_lookup: dict[str, tuple[float, float, float]],
    feature_name: str,
) -> tuple[float, float, float]:
    prior_mean, prior_precision, prior_variance = prior_lookup.get(
        feature_name,
        (
            0.0,
            1.0 / max(_DEFAULT_PRIOR_VARIANCE, _MIN_PRIOR_VARIANCE),
            _DEFAULT_PRIOR_VARIANCE,
        ),
    )
    return prior_mean, prior_precision, prior_variance


def _build_feature_vector(
    encoded: EncodedCandidate,
    *,
    feature_index: dict[str, int],
) -> numpy.ndarray:
    vector = numpy.zeros(len(feature_index), dtype=float)
    for feature_name in encoded.main_effects:
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = 1.0
    for feature_name in encoded.pair_effects:
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = 1.0
    for name, value in encoded.global_features.items():
        feature_name = f"global:{name}"
        index = feature_index.get(feature_name)
        if index is not None:
            vector[index] = float(value)
    return vector


def _sorted_feature_names(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    prior_lookup: dict[str, tuple[float, float, float]],
    pair_whitelist: set[str],
    config: BayesLayerConfig,
) -> tuple[str, ...]:
    feature_names = set(prior_lookup)
    aggregates = aggregate_eval_results(observations, registry=registry)
    include_active_gene_count = False
    for aggregate, encoded in iter_feature_memberships(
        aggregates,
        registry=registry,
        pair_whitelist=pair_whitelist or None,
    ):
        del aggregate
        feature_names.update(encoded.main_effects)
        feature_names.update(encoded.pair_effects)
        active_gene_count = encoded.global_features.get("active_gene_count")
        if (
            config.objective_surrogate.include_active_gene_count
            and active_gene_count is not None
        ):
            include_active_gene_count = True
    if include_active_gene_count:
        feature_names.add("global:active_gene_count")
    return tuple(sorted(feature_names))


def _observation_precision(
    utility_variance: float,
    count: int,
) -> float:
    variance = max(
        _MIN_OBSERVATION_NOISE,
        float(utility_variance) + (0.05 / max(float(count), 1.0)),
    )
    return max(float(count), 1.0) / variance


def _heuristic_fit_objective_surrogate(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    compiled_priors: CompiledPriorBundle,
    era_state: EraState,
    config: BayesLayerConfig,
    fallback_reason: str | None = None,
) -> ObjectivePosterior:
    aggregates = aggregate_eval_results(observations, registry=registry)
    pair_whitelist = _pair_whitelist(
        observations,
        registry=registry,
        compiled_priors=compiled_priors,
        config=config,
    )
    baseline = fmean(item.utility_mean for item in aggregates) if aggregates else 0.0
    prior_lookup = _prior_lookup(compiled_priors, era_state=era_state)
    feature_samples: dict[str, list[float]] = {}
    feature_support: Counter[str] = Counter()
    for aggregate, encoded in iter_feature_memberships(
        aggregates,
        registry=registry,
        pair_whitelist=pair_whitelist or None,
    ):
        for feature_name in encoded.main_effects + encoded.pair_effects:
            feature_samples.setdefault(feature_name, []).append(aggregate.utility_mean)
            feature_support[feature_name] += aggregate.count

    if aggregates and config.objective_surrogate.include_active_gene_count:
        counts = [float(len(item.realized_genotype)) for item in aggregates]
        if len(set(counts)) > 1:
            mean_count = fmean(counts)
            numerator = sum(
                (count - mean_count) * (aggregate.utility_mean - baseline)
                for count, aggregate in zip(counts, aggregates, strict=True)
            )
            denominator = sum((count - mean_count) ** 2 for count in counts)
            slope = numerator / denominator if denominator else 0.0
        else:
            slope = 0.0
        feature_samples["global:active_gene_count"] = [slope + baseline]

    features: list[PosteriorFeature] = []
    feature_names = set(feature_samples) | set(prior_lookup)
    for feature_name in sorted(feature_names):
        samples = feature_samples.get(feature_name, [])
        support = feature_support.get(feature_name, len(samples))
        observed_delta = (fmean(samples) - baseline) if samples else 0.0
        prior_mean, prior_precision, prior_variance = _feature_prior(
            prior_lookup,
            feature_name,
        )
        if support == 0:
            posterior_precision = prior_precision
            posterior_mean = prior_mean
        else:
            observation_noise = max(
                _MIN_OBSERVATION_NOISE,
                fmean((sample - baseline) ** 2 for sample in samples),
            )
            posterior_precision = prior_precision + (support / observation_noise)
            posterior_mean = (
                (prior_mean * prior_precision)
                + (observed_delta * support / observation_noise)
            ) / posterior_precision
        posterior_variance = 1.0 / posterior_precision
        features.append(
            PosteriorFeature(
                feature_name=feature_name,
                target_kind=_target_kind_for_feature(feature_name),
                posterior_mean=posterior_mean,
                posterior_variance=posterior_variance,
                support=int(support),
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )
        )

    return ObjectivePosterior(
        era_id=era_state.era_id,
        baseline_utility=baseline,
        features=tuple(
            sorted(
                features,
                key=lambda feature: (
                    -abs(feature.posterior_mean),
                    -feature.support,
                    feature.feature_name,
                ),
            )
        ),
        observation_count=sum(item.count for item in aggregates),
        backend="heuristic_independent_normal",
        sampleable=False,
        feature_order=tuple(sorted(feature_names)),
        aggregate_count=len(aggregates),
        effective_observation_count=float(sum(item.count for item in aggregates)),
        fallback_reason=fallback_reason,
    )


def _cholesky_factor(
    matrix: numpy.ndarray,
    *,
    jitter_steps: Sequence[float],
) -> tuple[numpy.ndarray, float] | None:
    identity = numpy.eye(matrix.shape[0], dtype=float)
    for jitter in jitter_steps:
        try:
            return numpy.linalg.cholesky(matrix + (jitter * identity)), float(jitter)
        except numpy.linalg.LinAlgError:
            continue
    return None


def fit_objective_surrogate(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    compiled_priors: CompiledPriorBundle,
    era_state: EraState,
    config: BayesLayerConfig | None = None,
) -> ObjectivePosterior:
    active_config = config or load_bayes_layer_config()
    pair_whitelist = _pair_whitelist(
        observations,
        registry=registry,
        compiled_priors=compiled_priors,
        config=active_config,
    )
    aggregates = aggregate_eval_results(observations, registry=registry)
    baseline = fmean(item.utility_mean for item in aggregates) if aggregates else 0.0
    prior_lookup = _prior_lookup(compiled_priors, era_state=era_state)
    feature_order = _sorted_feature_names(
        observations,
        registry=registry,
        prior_lookup=prior_lookup,
        pair_whitelist=pair_whitelist,
        config=active_config,
    )
    if not feature_order:
        return _heuristic_fit_objective_surrogate(
            observations,
            registry=registry,
            compiled_priors=compiled_priors,
            era_state=era_state,
            config=active_config,
            fallback_reason="exact_joint_linear_no_features",
        )

    feature_index = {
        feature_name: index for index, feature_name in enumerate(feature_order)
    }
    prior_mean_vector = numpy.zeros(len(feature_order), dtype=float)
    prior_precision_vector = numpy.zeros(len(feature_order), dtype=float)
    prior_variances: dict[str, float] = {}
    for feature_name in feature_order:
        prior_mean, prior_precision, prior_variance = _feature_prior(
            prior_lookup,
            feature_name,
        )
        index = feature_index[feature_name]
        prior_mean_vector[index] = prior_mean
        prior_precision_vector[index] = prior_precision
        prior_variances[feature_name] = prior_variance

    design_rows: list[numpy.ndarray] = []
    centered_utilities: list[float] = []
    observation_precisions: list[float] = []
    support_counter: Counter[str] = Counter()
    observed_noises: list[float] = []
    for aggregate, encoded in iter_feature_memberships(
        aggregates,
        registry=registry,
        pair_whitelist=pair_whitelist or None,
    ):
        vector = _build_feature_vector(encoded, feature_index=feature_index)
        if not vector.any() and not feature_order:
            continue
        design_rows.append(vector)
        centered_utilities.append(float(aggregate.utility_mean - baseline))
        precision = _observation_precision(aggregate.utility_variance, aggregate.count)
        observation_precisions.append(precision)
        observed_noises.append(max(_MIN_OBSERVATION_NOISE, aggregate.count / precision))
        for feature_name, index in feature_index.items():
            if vector[index] != 0.0:
                support_counter[feature_name] += aggregate.count

    if not design_rows and not prior_lookup:
        return _heuristic_fit_objective_surrogate(
            observations,
            registry=registry,
            compiled_priors=compiled_priors,
            era_state=era_state,
            config=active_config,
            fallback_reason="exact_joint_linear_no_features",
        )

    design = (
        numpy.vstack(design_rows)
        if design_rows
        else numpy.zeros((0, len(feature_order)), dtype=float)
    )
    centered = numpy.asarray(centered_utilities, dtype=float)
    observation_precision_vector = numpy.asarray(observation_precisions, dtype=float)
    diagonal_prior = numpy.diag(prior_precision_vector)
    if design_rows:
        weighted_design = observation_precision_vector[:, None] * design
        precision_matrix = diagonal_prior + (design.T @ weighted_design)
        rhs = (prior_precision_vector * prior_mean_vector) + (
            design.T @ (observation_precision_vector * centered)
        )
    else:
        precision_matrix = diagonal_prior
        rhs = prior_precision_vector * prior_mean_vector

    condition_number = float(numpy.linalg.cond(precision_matrix))
    if not math.isfinite(condition_number) or condition_number > _MAX_CONDITION_NUMBER:
        return _heuristic_fit_objective_surrogate(
            observations,
            registry=registry,
            compiled_priors=compiled_priors,
            era_state=era_state,
            config=active_config,
            fallback_reason="exact_joint_linear_ill_conditioned",
        )

    precision_factor = _cholesky_factor(
        precision_matrix,
        jitter_steps=_FIT_JITTER_STEPS,
    )
    if precision_factor is None:
        return _heuristic_fit_objective_surrogate(
            observations,
            registry=registry,
            compiled_priors=compiled_priors,
            era_state=era_state,
            config=active_config,
            fallback_reason="exact_joint_linear_cholesky_failed",
        )
    precision_cholesky, used_jitter = precision_factor

    posterior_mean_vector = numpy.linalg.solve(precision_matrix, rhs)
    covariance = numpy.linalg.solve(
        precision_matrix,
        numpy.eye(len(feature_order), dtype=float),
    )
    sampleable = (
        _cholesky_factor(covariance, jitter_steps=_SAMPLE_JITTER_STEPS) is not None
    )
    if not sampleable:
        return _heuristic_fit_objective_surrogate(
            observations,
            registry=registry,
            compiled_priors=compiled_priors,
            era_state=era_state,
            config=active_config,
            fallback_reason="exact_joint_linear_sampling_unsafe",
        )
    del precision_cholesky

    posterior_variances = numpy.diag(covariance)
    features: list[PosteriorFeature] = []
    for feature_name in feature_order:
        index = feature_index[feature_name]
        features.append(
            PosteriorFeature(
                feature_name=feature_name,
                target_kind=_target_kind_for_feature(feature_name),
                posterior_mean=float(posterior_mean_vector[index]),
                posterior_variance=float(max(posterior_variances[index], 0.0)),
                support=int(support_counter.get(feature_name, 0)),
                prior_mean=float(prior_mean_vector[index]),
                prior_variance=prior_variances[feature_name],
            )
        )

    return ObjectivePosterior(
        era_id=era_state.era_id,
        baseline_utility=baseline,
        features=tuple(
            sorted(
                features,
                key=lambda feature: (
                    -abs(feature.posterior_mean),
                    -feature.support,
                    feature.feature_name,
                ),
            )
        ),
        observation_count=sum(item.count for item in aggregates),
        backend="exact_joint_linear",
        sampleable=True,
        feature_order=feature_order,
        posterior_mean_vector=tuple(float(item) for item in posterior_mean_vector),
        posterior_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance.tolist()
        ),
        aggregate_count=len(aggregates),
        effective_observation_count=float(sum(observation_precision_vector))
        if observation_precisions
        else 0.0,
        condition_number=condition_number,
        used_jitter=used_jitter,
        observation_noise=(
            fmean(observed_noises) if observed_noises else _MIN_OBSERVATION_NOISE
        ),
    )


def _posterior_feature_lookup(
    posterior: ObjectivePosterior,
) -> dict[str, PosteriorFeature]:
    return {feature.feature_name: feature for feature in posterior.features}


def _posterior_pair_whitelist(
    posterior: ObjectivePosterior,
) -> set[str]:
    if posterior.feature_order:
        return {
            feature_name
            for feature_name in posterior.feature_order
            if feature_name.startswith("pair:")
        }
    return {
        feature.feature_name
        for feature in posterior.features
        if feature.target_kind == "pair_effect"
    }


def predict_utility(
    candidate: EncodedCandidate,
    *,
    posterior: ObjectivePosterior,
) -> tuple[float, float]:
    if (
        posterior.backend == "exact_joint_linear"
        and posterior.sampleable
        and posterior.feature_order
        and posterior.posterior_covariance
    ):
        feature_index = {
            feature_name: index
            for index, feature_name in enumerate(posterior.feature_order)
        }
        vector = _build_feature_vector(candidate, feature_index=feature_index)
        mean_vector = numpy.asarray(posterior.posterior_mean_vector, dtype=float)
        covariance = numpy.asarray(posterior.posterior_covariance, dtype=float)
        predicted_delta = float(vector @ mean_vector)
        predictive_variance = float(vector @ covariance @ vector)
        return (
            posterior.baseline_utility + predicted_delta,
            math.sqrt(max(_BASE_PREDICTION_VARIANCE + predictive_variance, 0.0)),
        )

    feature_lookup = _posterior_feature_lookup(posterior)
    mean = posterior.baseline_utility
    variance = _BASE_PREDICTION_VARIANCE
    for feature_name in candidate.main_effects + candidate.pair_effects:
        feature = feature_lookup.get(feature_name)
        if feature is None:
            continue
        mean += feature.posterior_mean
        variance += feature.posterior_variance
    count_feature = feature_lookup.get("global:active_gene_count")
    if count_feature is not None:
        active_gene_count = candidate.global_features["active_gene_count"]
        mean += count_feature.posterior_mean * active_gene_count
        variance += count_feature.posterior_variance * (active_gene_count**2)
    return mean, math.sqrt(max(variance, 0.0))


def score_encoded_candidate(
    candidate: EncodedCandidate,
    *,
    posterior: ObjectivePosterior,
) -> tuple[float, float]:
    return predict_utility(candidate, posterior=posterior)


def encode_and_score_candidate(
    candidate_id: str,
    genotype: Sequence[GeneStateRef],
    *,
    registry: GeneRegistry,
    posterior: ObjectivePosterior,
    pair_whitelist: set[str] | None = None,
) -> tuple[EncodedCandidate, float, float]:
    active_pair_whitelist = (
        pair_whitelist
        if pair_whitelist is not None
        else (_posterior_pair_whitelist(posterior) or None)
    )
    encoded = encode_candidate(
        candidate_id,
        genotype,
        registry=registry,
        pair_whitelist=active_pair_whitelist,
    )
    mean, uncertainty = predict_utility(encoded, posterior=posterior)
    return encoded, mean, uncertainty
