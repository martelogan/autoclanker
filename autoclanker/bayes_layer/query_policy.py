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


def _pairwise_query(
    ranked_candidates: Sequence[RankedCandidate],
) -> QuerySuggestion | None:
    if len(ranked_candidates) < 2:
        return None
    left = ranked_candidates[0]
    best_pair: tuple[RankedCandidate, RankedCandidate] | None = None
    best_score = -1.0
    best_scope = "candidate"
    for candidate in ranked_candidates[1:6]:
        acquisition_gap = abs(left.acquisition_score - candidate.acquisition_score)
        localized_uncertainty = left.uncertainty + candidate.uncertainty
        if (
            left.family_id
            and candidate.family_id
            and left.family_id != candidate.family_id
        ):
            family_bonus = 0.25
            comparison_scope = "family"
        else:
            family_bonus = 0.0
            comparison_scope = "candidate"
        comparison_value = (
            localized_uncertainty / (acquisition_gap + 0.1)
        ) + family_bonus
        if comparison_value > best_score:
            best_score = comparison_value
            best_pair = (left, candidate)
            best_scope = comparison_scope
    if best_pair is None:
        return None
    first, second = best_pair
    expected_value = best_score
    return QuerySuggestion(
        query_id="query_001",
        query_type="pairwise_preference",
        prompt=(
            f"Compare {first.candidate_id} against {second.candidate_id} and decide "
            f"which {best_scope} should be evaluated or strengthened next."
        ),
        target_refs=(first.candidate_id, second.candidate_id),
        expected_value=expected_value,
        confidence_gap=first.uncertainty + second.uncertainty,
        candidate_ids=(first.candidate_id, second.candidate_id),
        family_ids=tuple(
            family_id
            for family_id in (first.family_id, second.family_id)
            if family_id is not None
        ),
        comparison_scope=best_scope,
    )


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

    pairwise_query = None
    if "pairwise_preference" in active_config.query_policy.allowed_query_types:
        pairwise_query = _pairwise_query(ranked_candidates)
    if (
        pairwise_query is not None
        and pairwise_query.expected_value
        >= active_config.query_policy.min_expected_value_of_information
    ):
        queries.append(pairwise_query)
        next_index += 1

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
    return tuple(queries[: active_config.query_policy.max_queries_per_era])
