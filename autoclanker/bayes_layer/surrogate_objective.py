from __future__ import annotations

import math
import re

from collections import Counter
from collections.abc import Sequence
from statistics import fmean

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
    for spec in bundle.main_effect_priors + bundle.pair_priors:
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


def _target_kind_for_feature(feature_name: str) -> str:
    if feature_name.startswith("pair:"):
        return "pair_effect"
    if feature_name.startswith("global:"):
        return "global_feature"
    return "main_effect"


def fit_objective_surrogate(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
    compiled_priors: CompiledPriorBundle,
    era_state: EraState,
    config: BayesLayerConfig | None = None,
) -> ObjectivePosterior:
    active_config = config or load_bayes_layer_config()
    aggregates = aggregate_eval_results(observations, registry=registry)
    pair_support = Counter[str]()
    for aggregate, encoded in iter_feature_memberships(aggregates, registry=registry):
        del aggregate
        pair_support.update(encoded.pair_effects)
    pair_whitelist = {
        name
        for name, _count in pair_support.most_common(
            active_config.objective_surrogate.max_pair_features,
        )
    }
    for spec in compiled_priors.pair_priors + compiled_priors.linkage_hints:
        pair_whitelist.add(spec.item.target_ref)

    baseline = fmean(item.utility_mean for item in aggregates) if aggregates else 0.0
    prior_lookup = _prior_lookup(compiled_priors, era_state=era_state)
    feature_samples: dict[str, list[float]] = {}
    for aggregate, encoded in iter_feature_memberships(
        aggregates,
        registry=registry,
        pair_whitelist=pair_whitelist,
    ):
        for feature_name in encoded.main_effects + encoded.pair_effects:
            feature_samples.setdefault(feature_name, []).append(aggregate.utility_mean)

    if aggregates and active_config.objective_surrogate.include_active_gene_count:
        counts = [float(len(item.realized_genotype)) for item in aggregates]
        if len(set(counts)) > 1:
            mean_count = fmean(counts)
            mean_utility = baseline
            numerator = sum(
                (count - mean_count) * (aggregate.utility_mean - mean_utility)
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
        support = len(samples)
        observed_delta = (fmean(samples) - baseline) if samples else 0.0
        prior_mean, prior_precision = prior_lookup.get(
            feature_name,
            (0.0, 1.0 / max(1.5**2, 0.05)),
        )
        if support == 0:
            posterior_precision = prior_precision
            posterior_mean = prior_mean
        else:
            observation_noise = max(
                0.25, fmean((sample - baseline) ** 2 for sample in samples)
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
                support=support,
                prior_mean=prior_mean,
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
    )


def predict_utility(
    candidate: EncodedCandidate,
    *,
    posterior: ObjectivePosterior,
) -> tuple[float, float]:
    feature_lookup = {feature.feature_name: feature for feature in posterior.features}
    mean = posterior.baseline_utility
    variance = 0.05
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
        variance += count_feature.posterior_variance
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
    encoded = encode_candidate(
        candidate_id,
        genotype,
        registry=registry,
        pair_whitelist=pair_whitelist,
    )
    mean, uncertainty = predict_utility(encoded, posterior=posterior)
    return encoded, mean, uncertainty
