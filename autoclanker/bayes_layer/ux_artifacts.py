from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime

from autoclanker.bayes_layer.types import (
    BeliefDeltaEntry,
    BeliefDeltaSummary,
    CommitDecision,
    CompiledPriorBundle,
    FrontierSummary,
    PosteriorFeature,
    PosteriorSummary,
    ProposalLedger,
    ProposalLedgerEntry,
    RankedCandidate,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _prior_specs_by_target(
    compiled: CompiledPriorBundle,
) -> dict[str, tuple[str, float]]:
    specs: dict[str, tuple[str, float]] = {}
    for spec in compiled.all_items:
        specs[spec.item.target_ref] = (spec.source_belief_id, spec.item.mean)
    return specs


def _change_kind(
    feature: PosteriorFeature,
    prior_mean: float,
) -> str:
    prior_mag = abs(prior_mean)
    posterior_mag = abs(feature.posterior_mean)
    if feature.posterior_variance >= max(0.25, posterior_mag * 0.5):
        return "uncertain"
    if posterior_mag >= prior_mag + 0.15:
        return "strengthened"
    return "weakened"


def _feature_summary(
    feature: PosteriorFeature,
    *,
    prior_mean: float,
    change_kind: str,
) -> str:
    direction = "supports" if feature.posterior_mean >= 0 else "pushes against"
    if change_kind == "uncertain":
        return (
            f"Evidence for {feature.feature_name} remains uncertain after "
            f"{feature.support} aggregated observations."
        )
    if change_kind == "strengthened":
        return (
            f"Observed evidence {direction} {feature.feature_name} more strongly than "
            f"the prior expectation ({prior_mean:.2f} -> {feature.posterior_mean:.2f})."
        )
    return (
        f"Observed evidence weakened confidence in {feature.feature_name} relative to "
        f"the prior ({prior_mean:.2f} -> {feature.posterior_mean:.2f})."
    )


def build_belief_delta_summary(
    *,
    compiled: CompiledPriorBundle,
    summary: PosteriorSummary,
    frontier_summary: FrontierSummary | None = None,
    ranked_candidates: Sequence[RankedCandidate] = (),
) -> BeliefDeltaSummary:
    prior_lookup = _prior_specs_by_target(compiled)
    strengthened: list[BeliefDeltaEntry] = []
    weakened: list[BeliefDeltaEntry] = []
    uncertain: list[BeliefDeltaEntry] = []

    for feature in summary.top_features[:8]:
        source_belief_id, prior_mean = prior_lookup.get(
            feature.feature_name,
            ("derived", 0.0),
        )
        change_kind = _change_kind(feature, prior_mean)
        entry = BeliefDeltaEntry(
            source_belief_id=source_belief_id,
            target_ref=feature.feature_name,
            target_kind=feature.target_kind,
            change_kind=change_kind,  # type: ignore[arg-type]
            prior_mean=prior_mean,
            posterior_mean=feature.posterior_mean,
            posterior_variance=feature.posterior_variance,
            support=feature.support,
            summary=_feature_summary(
                feature,
                prior_mean=prior_mean,
                change_kind=change_kind,
            ),
        )
        if change_kind == "strengthened":
            strengthened.append(entry)
        elif change_kind == "weakened":
            weakened.append(entry)
        else:
            uncertain.append(entry)

    promoted_candidate_ids = tuple(
        candidate.candidate_id for candidate in tuple(ranked_candidates)[:3]
    )
    dropped_family_ids = (
        ()
        if frontier_summary is None
        else tuple(sorted(frontier_summary.dropped_family_reasons.keys()))
    )
    notes = (
        f"Objective backend: {summary.objective_backend}.",
        f"Acquisition backend: {summary.acquisition_backend}.",
    )
    return BeliefDeltaSummary(
        era_id=summary.era_id,
        strengthened=tuple(strengthened),
        weakened=tuple(weakened),
        uncertain=tuple(uncertain),
        promoted_candidate_ids=promoted_candidate_ids,
        dropped_family_ids=dropped_family_ids,
        notes=notes,
    )


def _evidence_summary(
    candidate: RankedCandidate,
    decision: CommitDecision | None,
) -> str:
    if decision is not None and decision.candidate_id == candidate.candidate_id:
        return decision.reason
    if candidate.influence_summary:
        return candidate.influence_summary[0]
    if candidate.rationale:
        return candidate.rationale[0]
    return "Leading lane by current acquisition score."


def _entry_state(
    candidate: RankedCandidate,
    frontier_summary: FrontierSummary | None,
    decision: CommitDecision | None,
) -> str:
    if decision is not None and decision.candidate_id == candidate.candidate_id:
        return "recommended" if decision.recommended else "candidate"
    if candidate.valid_probability < 0.5:
        return "blocked"
    if frontier_summary is not None and frontier_summary.pending_queries:
        return "deferred"
    return "candidate"


def _blockers(
    candidate: RankedCandidate,
    frontier_summary: FrontierSummary | None,
    decision: CommitDecision | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if candidate.valid_probability < 0.5:
        blockers.append("Predicted valid probability is below 0.50.")
    if frontier_summary is not None and frontier_summary.pending_queries:
        blockers.append("Pending comparison queries remain before final approval.")
    if (
        frontier_summary is not None
        and frontier_summary.pending_merge_suggestions
        and candidate.origin_kind != "merge"
    ):
        blockers.append("Pending merge suggestions remain for the active frontier.")
    if decision is not None and not decision.recommended:
        blockers.append(decision.reason)
    return tuple(dict.fromkeys(blockers))


def build_proposal_ledger(
    *,
    session_id: str,
    era_id: str,
    ranked_candidates: Sequence[RankedCandidate],
    frontier_summary: FrontierSummary | None,
    decision: CommitDecision | None,
    previous: ProposalLedger | None = None,
    artifact_refs: Mapping[str, str] | None = None,
) -> ProposalLedger:
    now = _utc_now()
    previous_entries = (
        {entry.proposal_id: entry for entry in previous.entries} if previous else {}
    )
    entries: list[ProposalLedgerEntry] = []
    current_ids: list[str] = []

    top_candidates = tuple(ranked_candidates)[:3]
    for index, candidate in enumerate(top_candidates):
        proposal_id = f"proposal_{candidate.candidate_id}"
        current_ids.append(proposal_id)
        readiness_state = _entry_state(candidate, frontier_summary, decision)
        if index > 0 and readiness_state not in {"blocked", "deferred"}:
            readiness_state = "superseded"
        entry = ProposalLedgerEntry(
            proposal_id=proposal_id,
            session_id=session_id,
            era_id=era_id,
            candidate_id=candidate.candidate_id,
            family_id=candidate.family_id,
            readiness_state=readiness_state,  # type: ignore[arg-type]
            evidence_summary=_evidence_summary(candidate, decision),
            unresolved_risks=_blockers(candidate, frontier_summary, decision),
            approval_required=readiness_state == "recommended",
            updated_at=now,
            artifact_refs=None if artifact_refs is None else dict(artifact_refs),
            resume_token=f"{session_id}:{era_id}:{proposal_id}",
            source_candidate_ids=candidate.parent_candidate_ids,
            supersedes=()
            if index == 0
            else (f"proposal_{top_candidates[0].candidate_id}",),
            recommendation_reason=None if decision is None else decision.reason,
        )
        entries.append(entry)

    for proposal_id, previous_entry in previous_entries.items():
        if proposal_id in current_ids:
            continue
        entries.append(
            replace(
                previous_entry,
                readiness_state="superseded",
                updated_at=now,
                supersedes=tuple(
                    dict.fromkeys(
                        (
                            *previous_entry.supersedes,
                            *(tuple(current_ids[:1]) if current_ids else ()),
                        )
                    )
                ),
            )
        )

    current_proposal_id = (
        current_ids[0]
        if current_ids
        else previous.current_proposal_id
        if previous
        else None
    )
    return ProposalLedger(
        session_id=session_id,
        era_id=era_id,
        current_proposal_id=current_proposal_id,
        entries=tuple(entries),
        updated_at=now,
    )
