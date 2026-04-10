from __future__ import annotations

import math

from collections.abc import Sequence

from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.feature_encoder import aggregate_eval_results
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    CommitDecision,
    InfluenceSummary,
    RankedCandidate,
    ValidEvalResult,
)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def recommend_commit(
    *,
    session_id: str,
    era_id: str,
    ranked_candidates: Sequence[RankedCandidate],
    observations: Sequence[ValidEvalResult],
    registry: GeneRegistry,
    config: BayesLayerConfig | None = None,
) -> CommitDecision:
    active_config = config or load_bayes_layer_config()
    aggregates = aggregate_eval_results(observations, registry=registry)
    incumbent = max(
        (
            aggregate.utility_mean
            for aggregate in aggregates
            if aggregate.status_counts.get("valid", 0) > 0
        ),
        default=0.0,
    )
    if not ranked_candidates:
        return CommitDecision(
            era_id=era_id,
            session_id=session_id,
            recommended=False,
            candidate_id=None,
            predicted_gain=0.0,
            gain_probability=0.0,
            valid_probability=0.0,
            reason="No ranked candidates were available.",
            thresholds={
                "posterior_gain_probability_threshold": active_config.commit.posterior_gain_probability_threshold,
                "min_valid_probability": active_config.commit.min_valid_probability,
                "epsilon_commit_floor": active_config.commit.epsilon_commit_floor,
            },
            influence_summary=(),
        )
    best = ranked_candidates[0]
    predicted_gain = best.predicted_utility - incumbent
    gain_probability = _sigmoid(
        predicted_gain / max(best.uncertainty, 0.2),
    )
    recommended = (
        predicted_gain >= active_config.commit.epsilon_commit_floor
        and gain_probability
        >= active_config.commit.posterior_gain_probability_threshold
        and best.valid_probability >= active_config.commit.min_valid_probability
    )
    reason = (
        f"Candidate {best.candidate_id} cleared commit thresholds."
        if recommended
        else f"Candidate {best.candidate_id} did not clear commit thresholds."
    )
    return CommitDecision(
        era_id=era_id,
        session_id=session_id,
        recommended=recommended,
        candidate_id=best.candidate_id,
        predicted_gain=predicted_gain,
        gain_probability=gain_probability,
        valid_probability=best.valid_probability,
        reason=reason,
        thresholds={
            "posterior_gain_probability_threshold": active_config.commit.posterior_gain_probability_threshold,
            "min_valid_probability": active_config.commit.min_valid_probability,
            "epsilon_commit_floor": active_config.commit.epsilon_commit_floor,
        },
        influence_summary=tuple(
            InfluenceSummary(
                source_belief_id=item.split(" -> ", maxsplit=1)[0],
                target_ref=item.split(" -> ", maxsplit=1)[1],
                summary=f"Commit recommendation considered {item}.",
            )
            for item in best.influence_summary
            if " -> " in item
        ),
    )
