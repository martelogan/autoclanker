from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from autoclanker.bayes_layer.belief_io import (
    parse_gene_state_refs,
    validate_payload_against_schema,
)
from autoclanker.bayes_layer.types import (
    FrontierCandidate,
    FrontierDocument,
    FrontierFamilyRepresentative,
    FrontierOriginKind,
    FrontierSummary,
    GeneStateRef,
    MergeSuggestion,
    QuerySuggestion,
    RankedCandidate,
    ValidationFailure,
)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label} must be a mapping.")
    return cast(Mapping[str, object], value)


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationFailure(f"{key} must be a string.")
    normalized = value.strip()
    return normalized or None


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValidationFailure(f"{key} must be numeric.")
    return float(value)


def _optional_string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure(f"{key} must be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValidationFailure(f"{key} must be a list of strings.")
    items = cast(list[str], raw_items)
    return tuple(item.strip() for item in items if item.strip())


def parse_frontier_payload(payload: Mapping[str, object]) -> FrontierDocument:
    validate_payload_against_schema(payload, "frontier_input.schema.json")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValidationFailure("Frontier payload must contain a 'candidates' list.")
    default_family_id = (
        _optional_string(payload, "default_family_id") or "family_default"
    )
    frontier_id = _optional_string(payload, "frontier_id") or "frontier_default"
    candidates: list[FrontierCandidate] = []
    for index, raw_candidate in enumerate(cast(list[object], raw_candidates), start=1):
        mapping = _require_mapping(raw_candidate, f"candidates[{index - 1}]")
        candidate_id = (
            _optional_string(mapping, "candidate_id") or f"cand_input_{index:03d}"
        )
        origin_kind = cast(
            FrontierOriginKind,
            (_optional_string(mapping, "origin_kind") or "legacy_pool"),
        )
        if origin_kind not in {
            "legacy_pool",
            "manual",
            "belief",
            "query",
            "merge",
            "seed",
        }:
            raise ValidationFailure(
                f"candidates[{index - 1}].origin_kind must be one of legacy_pool, manual, belief, query, merge, seed."
            )
        budget_weight: float | None = None
        if mapping.get("budget_weight") is not None:
            budget_weight = _require_float(mapping, "budget_weight")
        candidates.append(
            FrontierCandidate(
                candidate_id=candidate_id,
                genotype=parse_gene_state_refs(
                    mapping.get("genotype"),
                    f"candidates[{index - 1}].genotype",
                    preserve_order=True,
                ),
                family_id=_optional_string(mapping, "family_id") or default_family_id,
                origin_kind=origin_kind,
                parent_candidate_ids=_optional_string_list(
                    mapping, "parent_candidate_ids"
                ),
                parent_belief_ids=_optional_string_list(mapping, "parent_belief_ids"),
                origin_query_ids=_optional_string_list(mapping, "origin_query_ids"),
                notes=_optional_string(mapping, "notes"),
                budget_weight=budget_weight,
            )
        )
    return FrontierDocument(
        candidates=tuple(candidates),
        frontier_id=frontier_id,
        default_family_id=default_family_id,
    )


def frontier_from_candidate_pairs(
    candidates: Sequence[tuple[str, Sequence[GeneStateRef]]],
) -> FrontierDocument:
    normalized: list[FrontierCandidate] = []
    for candidate_id, genotype in candidates:
        normalized.append(
            FrontierCandidate(
                candidate_id=candidate_id,
                genotype=tuple(genotype),
            )
        )
    return FrontierDocument(candidates=tuple(normalized))


def frontier_candidates_payload(
    frontier: FrontierDocument,
) -> tuple[tuple[str, tuple[GeneStateRef, ...]], ...]:
    return tuple(
        (candidate.candidate_id, tuple(candidate.genotype))
        for candidate in frontier.candidates
    )


def summarize_frontier(
    frontier: FrontierDocument,
    *,
    ranked_candidates: Sequence[RankedCandidate],
    queries: Sequence[QuerySuggestion],
) -> FrontierSummary:
    ranked_by_family: dict[str, list[RankedCandidate]] = defaultdict(list)
    family_budget_weights: dict[str, float] = defaultdict(float)
    for candidate in frontier.candidates:
        family_budget_weights[candidate.family_id] += candidate.budget_weight or 1.0
    for ranked in ranked_candidates:
        family_id = ranked.family_id or frontier.default_family_id
        ranked_by_family[family_id].append(ranked)

    representatives: list[FrontierFamilyRepresentative] = []
    for family_id in sorted(ranked_by_family):
        candidates = ranked_by_family[family_id]
        representative = candidates[0]
        representatives.append(
            FrontierFamilyRepresentative(
                family_id=family_id,
                representative_candidate_id=representative.candidate_id,
                representative_acquisition_score=representative.acquisition_score,
                candidate_count=len(candidates),
                compared_candidate_ids=tuple(item.candidate_id for item in candidates),
                budget_weight=family_budget_weights.get(
                    family_id, float(len(candidates))
                ),
            )
        )

    total_budget = (
        sum(representative.budget_weight for representative in representatives) or 1.0
    )
    budget_allocations = {
        representative.family_id: round(representative.budget_weight / total_budget, 3)
        for representative in representatives
    }

    dropped_family_reasons: dict[str, str] = {}
    if representatives:
        best_score = max(
            item.representative_acquisition_score for item in representatives
        )
        for representative in representatives:
            if len(
                representatives
            ) > 2 and representative.representative_acquisition_score < (
                best_score - 0.2
            ):
                dropped_family_reasons[representative.family_id] = (
                    "below frontier budget threshold after representative comparison"
                )

    pending_merge_suggestions: list[MergeSuggestion] = []
    if len(representatives) >= 2:
        ranked_reps = sorted(
            representatives,
            key=lambda item: (-item.representative_acquisition_score, item.family_id),
        )
        left = ranked_reps[0]
        right = ranked_reps[1]
        pending_merge_suggestions.append(
            MergeSuggestion(
                merge_id=f"merge_{left.family_id}_{right.family_id}",
                family_ids=(left.family_id, right.family_id),
                candidate_ids=(
                    left.representative_candidate_id,
                    right.representative_candidate_id,
                ),
                rationale=(
                    "Top frontier families remain distinct; compare or synthesize them before dropping either lane."
                ),
            )
        )

    return FrontierSummary(
        frontier_id=frontier.frontier_id,
        candidate_count=len(frontier.candidates),
        family_count=len({candidate.family_id for candidate in frontier.candidates}),
        family_representatives=tuple(representatives),
        dropped_family_reasons=dropped_family_reasons,
        pending_queries=tuple(queries),
        pending_merge_suggestions=tuple(pending_merge_suggestions),
        budget_allocations=budget_allocations,
    )
