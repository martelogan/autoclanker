from __future__ import annotations

import math
import re

from collections.abc import Sequence
from statistics import fmean

from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.feature_encoder import (
    aggregate_eval_results,
    iter_feature_memberships,
)
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    EncodedCandidate,
    EraState,
    FeasibilityPosterior,
    PosteriorFeature,
    ValidEvalResult,
)


def _clamp_probability(value: float) -> float:
    return min(max(value, 1e-3), 1.0 - 1e-3)


def _logit(probability: float) -> float:
    bounded = _clamp_probability(probability)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


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
) -> dict[str, tuple[float, float]]:
    weighted_means: dict[str, float] = {}
    precision_totals: dict[str, float] = {}
    for spec in bundle.feasibility_priors:
        variance = max(spec.item.scale**2, 0.05)
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
        weighted_means[spec.item.target_ref] += spec.item.mean * effective_precision
        precision_totals[spec.item.target_ref] += effective_precision
    return {
        target_ref: (
            weighted_means[target_ref] / precision_totals[target_ref],
            precision_totals[target_ref],
        )
        for target_ref in weighted_means
        if precision_totals[target_ref] > 0.0
    }


def _feature_prior_lookup(
    prior_lookup: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    weighted_means: dict[str, float] = {}
    precision_totals: dict[str, float] = {}
    for prior_ref, (mean, precision) in prior_lookup.items():
        feature_suffix = prior_ref.removeprefix("feasibility:").split("#", maxsplit=1)[
            0
        ]
        feature_name = f"main:{feature_suffix}"
        weighted_means.setdefault(feature_name, 0.0)
        precision_totals.setdefault(feature_name, 0.0)
        weighted_means[feature_name] += (-0.25 * mean) * precision
        precision_totals[feature_name] += precision
    return {
        feature_name: (
            weighted_means[feature_name] / precision_totals[feature_name],
            precision_totals[feature_name],
        )
        for feature_name in weighted_means
        if precision_totals[feature_name] > 0.0
    }


def fit_feasibility_surrogate(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    compiled_priors: CompiledPriorBundle,
    era_state: EraState,
    config: BayesLayerConfig | None = None,
) -> FeasibilityPosterior:
    active_config = config or load_bayes_layer_config()
    del active_config
    aggregates = aggregate_eval_results(observations, registry=registry)
    baseline_valid_probability = (
        fmean(item.valid_rate for item in aggregates) if aggregates else 0.5
    )
    baseline_logit = _logit(baseline_valid_probability)
    prior_lookup = _prior_lookup(compiled_priors, era_state=era_state)
    feature_prior_lookup = _feature_prior_lookup(prior_lookup)
    feature_samples: dict[str, list[float]] = {}
    for aggregate, encoded in iter_feature_memberships(aggregates, registry=registry):
        for feature_name in encoded.main_effects:
            feature_samples.setdefault(feature_name, []).append(aggregate.valid_rate)

    features: list[PosteriorFeature] = []
    feature_names = set(feature_samples) | set(feature_prior_lookup)
    for feature_name in sorted(feature_names):
        samples = feature_samples.get(feature_name, [])
        support = len(samples)
        observed_logit_delta = (
            (_logit(fmean(samples)) - baseline_logit) if samples else 0.0
        )
        prior_mean, prior_precision = feature_prior_lookup.get(feature_name, (0.0, 1.0))
        posterior_precision = support + prior_precision
        posterior_mean = (
            (support * observed_logit_delta) + (prior_mean * prior_precision)
        ) / posterior_precision
        posterior_variance = 1.0 / posterior_precision
        features.append(
            PosteriorFeature(
                feature_name=feature_name,
                target_kind="feasibility_logit",
                posterior_mean=posterior_mean,
                posterior_variance=posterior_variance,
                support=support,
                prior_mean=prior_mean,
            )
        )

    failure_mode_biases: dict[str, float] = {}
    for prior_ref, (mean, precision) in prior_lookup.items():
        failure_mode = prior_ref.rsplit("#", maxsplit=1)[-1]
        failure_mode_biases.setdefault(failure_mode, 0.0)
        failure_mode_biases[failure_mode] += mean * precision

    return FeasibilityPosterior(
        era_id=era_state.era_id,
        baseline_valid_probability=baseline_valid_probability,
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
        failure_mode_biases=dict(sorted(failure_mode_biases.items())),
        observation_count=sum(item.count for item in aggregates),
    )


def predict_valid_probability(
    candidate: EncodedCandidate,
    *,
    posterior: FeasibilityPosterior,
) -> float:
    baseline_logit = _logit(posterior.baseline_valid_probability)
    feature_lookup = {feature.feature_name: feature for feature in posterior.features}
    total_logit = baseline_logit
    for feature_name in candidate.main_effects:
        feature = feature_lookup.get(feature_name)
        if feature is not None:
            total_logit += feature.posterior_mean
    return _sigmoid(total_logit)
