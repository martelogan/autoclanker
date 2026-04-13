from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    del active_config
    ranked: list[RankedCandidate] = []
    pair_whitelist = {
        feature.feature_name
        for feature in objective_posterior.features
        if feature.target_kind == "pair_effect"
    }

    def candidate_influence_summary(
        encoded: EncodedCandidate,
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

    for candidate_id, genotype in candidates:
        frontier_candidate = (
            None
            if frontier_candidates is None
            else frontier_candidates.get(candidate_id)
        )
        encoded, predicted_utility, uncertainty = encode_and_score_candidate(
            candidate_id,
            genotype,
            registry=registry,
            posterior=objective_posterior,
            pair_whitelist=pair_whitelist or None,
        )
        valid_probability = predict_valid_probability(
            encoded,
            posterior=feasibility_posterior,
        )
        optimistic_utility = predicted_utility + (0.5 * uncertainty)
        acquisition_score = valid_probability * optimistic_utility
        rationale: list[str] = []
        if valid_probability >= 0.8:
            rationale.append("high feasibility")
        if uncertainty >= 0.6:
            rationale.append("informative uncertainty")
        if predicted_utility >= objective_posterior.baseline_utility:
            rationale.append("above incumbent baseline")
        ranked.append(
            RankedCandidate(
                candidate_id=candidate_id,
                genotype=encoded.genotype,
                predicted_utility=predicted_utility,
                uncertainty=uncertainty,
                valid_probability=valid_probability,
                acquisition_score=acquisition_score,
                rationale=tuple(rationale),
                influence_summary=candidate_influence_summary(encoded),
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
                -candidate.predicted_utility,
                candidate.candidate_id,
            ),
        )
    )
