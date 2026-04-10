from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean, variance
from typing import cast

from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AggregatedObservation,
    EncodedCandidate,
    GeneStateRef,
    ValidEvalResult,
)


def _metadata_string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in cast(list[object], value):
        if isinstance(item, str) and item.strip():
            items.append(item.strip())
    return tuple(items)


def _state_rule_refs(
    registry: GeneRegistry,
) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]], ...]:
    rules: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    for gene_id, definition in sorted(registry.genes.items()):
        if definition.materializable:
            continue
        metadata = definition.metadata or {}
        raw_state_rules = metadata.get("state_rules")
        if not isinstance(raw_state_rules, Mapping):
            continue
        for state_id in definition.states:
            if state_id == definition.default_state:
                continue
            raw_rule = cast(Mapping[object, object], raw_state_rules).get(state_id)
            if not isinstance(raw_rule, Mapping):
                continue
            rule = cast(Mapping[str, object], raw_rule)
            feature_ref = registry.main_ref(
                GeneStateRef(gene_id=gene_id, state_id=state_id)
            )
            rules.append(
                (
                    feature_ref,
                    _metadata_string_list(rule.get("implied_by_all")),
                    _metadata_string_list(rule.get("implied_by_any")),
                    _metadata_string_list(rule.get("blocked_by_any")),
                )
            )
    return tuple(rules)


def strategy_materialization_groups(
    registry: GeneRegistry,
) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for (
        _feature_ref,
        implied_by_all,
        _implied_by_any,
        _blocked_by_any,
    ) in _state_rule_refs(registry):
        main_refs = tuple(
            ref for ref in implied_by_all if ref.startswith("main:") and "=" in ref
        )
        if not main_refs:
            continue
        groups.append(main_refs)
    return tuple(dict.fromkeys(groups))


def inferred_surface_feature_refs(
    genotype: Sequence[GeneStateRef],
    *,
    registry: GeneRegistry,
) -> tuple[str, ...]:
    canonical = canonicalize_genotype(genotype, registry=registry)
    active_refs = {registry.main_ref(ref) for ref in canonical}
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            active_refs.add(registry.pair_ref((left, right)))

    inferred: list[str] = []
    changed = True
    while changed:
        changed = False
        for (
            feature_ref,
            implied_by_all,
            implied_by_any,
            blocked_by_any,
        ) in _state_rule_refs(registry):
            if feature_ref in active_refs:
                continue
            if blocked_by_any and any(ref in active_refs for ref in blocked_by_any):
                continue
            if implied_by_all and not all(ref in active_refs for ref in implied_by_all):
                continue
            if implied_by_any and not any(ref in active_refs for ref in implied_by_any):
                continue
            if not implied_by_all and not implied_by_any:
                continue
            active_refs.add(feature_ref)
            inferred.append(feature_ref)
            changed = True
    return tuple(dict.fromkeys(inferred))


def canonicalize_genotype(
    genotype: Sequence[GeneStateRef],
    *,
    registry: GeneRegistry,
) -> tuple[GeneStateRef, ...]:
    return registry.canonicalize_members(genotype)


def encode_candidate(
    candidate_id: str,
    genotype: Sequence[GeneStateRef],
    *,
    registry: GeneRegistry,
    pair_whitelist: set[str] | None = None,
) -> EncodedCandidate:
    canonical = canonicalize_genotype(genotype, registry=registry)
    concrete_main_effects = tuple(registry.main_ref(ref) for ref in canonical)
    all_pair_effects: list[str] = []
    for index, left in enumerate(canonical):
        for right in canonical[index + 1 :]:
            pair_ref = registry.pair_ref((left, right))
            all_pair_effects.append(pair_ref)
            if pair_whitelist is not None and pair_ref not in pair_whitelist:
                continue
    inferred_main_effects = inferred_surface_feature_refs(canonical, registry=registry)
    return EncodedCandidate(
        candidate_id=candidate_id,
        genotype=canonical,
        main_effects=tuple(
            dict.fromkeys(concrete_main_effects + inferred_main_effects)
        ),
        pair_effects=tuple(
            pair_ref
            for pair_ref in all_pair_effects
            if pair_whitelist is None or pair_ref in pair_whitelist
        ),
        global_features={"active_gene_count": float(len(canonical))},
    )


def aggregate_eval_results(
    observations: Sequence[ValidEvalResult],
    *,
    registry: GeneRegistry,
) -> tuple[AggregatedObservation, ...]:
    grouped: dict[str, list[ValidEvalResult]] = defaultdict(list)
    for observation in observations:
        grouped[observation.patch_hash].append(observation)

    aggregates: list[AggregatedObservation] = []
    for patch_hash, patch_observations in sorted(grouped.items()):
        utilities = [item.utility for item in patch_observations]
        delta_perfs = [item.delta_perf for item in patch_observations]
        runtimes = [item.runtime_sec for item in patch_observations]
        vrams = [item.peak_vram_mb for item in patch_observations]
        status_counts = Counter(item.status for item in patch_observations)
        canonical_genotype = canonicalize_genotype(
            patch_observations[0].realized_genotype,
            registry=registry,
        )
        aggregates.append(
            AggregatedObservation(
                patch_hash=patch_hash,
                candidate_ids=tuple(
                    sorted({item.candidate_id for item in patch_observations})
                ),
                realized_genotype=canonical_genotype,
                utility_mean=fmean(utilities),
                utility_variance=variance(utilities) if len(utilities) > 1 else 0.0,
                delta_perf_mean=fmean(delta_perfs),
                valid_rate=status_counts.get("valid", 0)
                / float(len(patch_observations)),
                runtime_sec_mean=fmean(runtimes),
                peak_vram_mb_mean=fmean(vrams),
                count=len(patch_observations),
                status_counts=dict(sorted(status_counts.items())),
            )
        )
    return tuple(aggregates)


def observed_feature_names(
    aggregates: Sequence[AggregatedObservation],
    *,
    registry: GeneRegistry,
    pair_whitelist: set[str] | None = None,
) -> tuple[str, ...]:
    feature_names: set[str] = set()
    for aggregate in aggregates:
        encoded = encode_candidate(
            aggregate.patch_hash,
            aggregate.realized_genotype,
            registry=registry,
            pair_whitelist=pair_whitelist,
        )
        feature_names.update(encoded.main_effects)
        feature_names.update(encoded.pair_effects)
    return tuple(sorted(feature_names))


def iter_feature_memberships(
    aggregates: Sequence[AggregatedObservation],
    *,
    registry: GeneRegistry,
    pair_whitelist: set[str] | None = None,
) -> Iterable[tuple[AggregatedObservation, EncodedCandidate]]:
    for aggregate in aggregates:
        yield (
            aggregate,
            encode_candidate(
                aggregate.patch_hash,
                aggregate.realized_genotype,
                registry=registry,
                pair_whitelist=pair_whitelist,
            ),
        )
