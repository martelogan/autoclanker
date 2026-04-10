from __future__ import annotations

from collections.abc import Sequence

from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.types import (
    ExpertPriorBelief,
    GraphDirectiveBelief,
    IdeaBelief,
    MainEffectTarget,
    ObjectivePosterior,
    PairEffectTarget,
    QuerySuggestion,
    RankedCandidate,
    RelationBelief,
    ValidatedBeliefBatch,
)


def _main_feature_targets(beliefs: ValidatedBeliefBatch) -> set[str]:
    targets: set[str] = set()
    for belief in beliefs.beliefs:
        if isinstance(belief, IdeaBelief):
            gene = belief.gene
            targets.add(f"main:{gene.gene_id}={gene.state_id}")
        if isinstance(belief, ExpertPriorBelief) and isinstance(
            belief.target,
            MainEffectTarget,
        ):
            gene = belief.target.gene
            targets.add(f"main:{gene.gene_id}={gene.state_id}")
        if isinstance(belief, ExpertPriorBelief) and isinstance(
            belief.target,
            PairEffectTarget,
        ):
            members = sorted(
                belief.target.members,
                key=lambda ref: (ref.gene_id, ref.state_id),
            )
            left, right = members
            targets.add(
                f"pair:{left.gene_id}={left.state_id}+{right.gene_id}={right.state_id}"
            )
        if isinstance(belief, (RelationBelief, GraphDirectiveBelief)):
            members = sorted(
                belief.members,
                key=lambda ref: (ref.gene_id, ref.state_id),
            )
            if len(members) == 2:
                left, right = members
                targets.add(
                    f"pair:{left.gene_id}={left.state_id}+{right.gene_id}={right.state_id}"
                )
    return targets


def suggest_queries(
    objective_posterior: ObjectivePosterior,
    *,
    beliefs: ValidatedBeliefBatch,
    ranked_candidates: Sequence[RankedCandidate] = (),
    config: BayesLayerConfig | None = None,
) -> tuple[QuerySuggestion, ...]:
    active_config = config or load_bayes_layer_config()
    if not active_config.query_policy.enabled:
        return ()
    known_targets = _main_feature_targets(beliefs)
    queries: list[QuerySuggestion] = []
    next_index = 1
    for feature in sorted(
        objective_posterior.features,
        key=lambda item: (-item.posterior_variance, -abs(item.posterior_mean)),
    ):
        if feature.feature_name in known_targets:
            continue
        query_type = "effect_sign"
        prompt = (
            f"Confirm the sign and confidence of {feature.feature_name} for this era."
        )
        if feature.target_kind == "pair_effect":
            query_type = "relation_check"
            prompt = (
                f"Confirm whether {feature.feature_name} is synergistic or conflicting."
            )
        expected_value = feature.posterior_variance * max(
            abs(feature.posterior_mean), 0.1
        )
        if (
            expected_value
            < active_config.query_policy.min_expected_value_of_information
        ):
            continue
        queries.append(
            QuerySuggestion(
                query_id=f"query_{next_index:03d}",
                query_type=query_type,
                prompt=prompt,
                target_refs=(feature.feature_name,),
                expected_value=expected_value,
                confidence_gap=feature.posterior_variance,
            )
        )
        next_index += 1
        if len(queries) >= active_config.query_policy.max_queries_per_era:
            break
    if (
        not queries
        and "pairwise_preference" in active_config.query_policy.allowed_query_types
        and len(ranked_candidates) >= 2
    ):
        left = ranked_candidates[0]
        right = ranked_candidates[1]
        queries.append(
            QuerySuggestion(
                query_id="query_001",
                query_type="pairwise_preference",
                prompt=(
                    f"Choose between {left.candidate_id} and {right.candidate_id} "
                    "if a human preference is available."
                ),
                target_refs=(left.candidate_id, right.candidate_id),
                expected_value=abs(left.acquisition_score - right.acquisition_score)
                + 0.1,
                confidence_gap=left.uncertainty + right.uncertainty,
            )
        )
    return tuple(queries)
