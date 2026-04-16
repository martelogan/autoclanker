from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer import load_serialized_payload
from autoclanker.bayes_layer.adapters import load_adapter
from autoclanker.bayes_layer.config import SessionArtifactConfig
from autoclanker.bayes_layer.eval_contract import (
    compare_eval_contracts,
    drift_status_for_contracts,
)
from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.types import (
    BeliefDeltaSummary,
    CommitDecision,
    ConstraintBelief,
    EvalContractSnapshot,
    ExpertPriorBelief,
    FeasibilityTarget,
    FrontierSummary,
    GraphDirectiveBelief,
    IdeaBelief,
    JsonValue,
    MainEffectTarget,
    ProposalBelief,
    ProposalLedger,
    ProposalLedgerEntry,
    QuerySuggestion,
    RelationBelief,
    SessionManifest,
    SessionStatus,
    ValidAdapterConfig,
    ValidatedBeliefBatch,
    ValidEvalResult,
    VramTarget,
    to_json_value,
)


def _summary_string(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _summary_list(value: object | None) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(cast(Sequence[object], value))


def _load_json_artifact(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    return cast(Mapping[str, object], load_serialized_payload(path))


def _brief_payload(
    *,
    summary: str,
    bullets: Sequence[str],
    extra: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {
        "summary": summary,
        "bullets": cast(list[JsonValue], to_json_value(tuple(bullets))),
    }
    if extra is not None:
        payload.update(extra)
    return payload


def _belief_ids(beliefs: ValidatedBeliefBatch | None) -> tuple[str, ...]:
    if beliefs is None:
        return ()
    return tuple(belief.id for belief in beliefs.beliefs)


def _proposal_beliefs(
    beliefs: ValidatedBeliefBatch | None,
) -> tuple[ProposalBelief, ...]:
    if beliefs is None:
        return ()
    return tuple(
        belief for belief in beliefs.beliefs if isinstance(belief, ProposalBelief)
    )


def normalized_session_status(
    *,
    store: FilesystemSessionStore,
    session_id: str,
    adapter_config: ValidAdapterConfig | None = None,
) -> tuple[SessionStatus, EvalContractSnapshot | None, EvalContractSnapshot | None]:
    status = store.status(session_id)
    expected_contract = store.load_eval_contract(session_id)
    current_contract: EvalContractSnapshot | None = None
    if adapter_config is not None:
        try:
            current_contract = load_adapter(adapter_config).capture_eval_contract()
        except Exception:  # pragma: no cover - status remains readable without live probe
            current_contract = None
    return (
        replace(
            status,
            current_eval_contract_digest=(
                None
                if current_contract is None
                else current_contract.contract_digest
            ),
            eval_contract_matches_current=(
                None
                if expected_contract is None or current_contract is None
                else not compare_eval_contracts(expected_contract, current_contract)
            ),
            eval_contract_drift_status=drift_status_for_contracts(
                expected_contract,
                current_contract,
            ),
        ),
        expected_contract,
        current_contract,
    )


def _query_comparison(query: QuerySuggestion | None) -> str | None:
    if query is None:
        return None
    if len(query.candidate_ids) >= 2:
        return f"{query.candidate_ids[0]} vs {query.candidate_ids[1]}"
    if len(query.family_ids) >= 2:
        return f"{query.family_ids[0]} vs {query.family_ids[1]}"
    if query.target_refs:
        return ", ".join(query.target_refs[:2])
    return None


def _query_reason(query: QuerySuggestion | None) -> str | None:
    if query is None:
        return None
    comparison = _query_comparison(query)
    if comparison is None:
        return "Resolving the next structured query would reduce uncertainty."
    if query.query_type == "pairwise_preference":
        return (
            f"{comparison} are currently close; choosing between them would reduce "
            "uncertainty most."
        )
    if query.query_type == "relation_check":
        return (
            f"{comparison} may reinforce or conflict with each other; clarifying "
            "that relation would change how the frontier branches next."
        )
    if query.query_type == "risk_triage":
        return (
            f"{comparison} is gated by risk tolerance; clarifying that would change "
            "ranking or proposal readiness."
        )
    return (
        f"{comparison} is the next useful comparison because the current frontier "
        "still cannot separate them cleanly."
    )


def _last_eval_summary_by_candidate(
    observations: Sequence[ValidEvalResult],
) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for observation in observations:
        summaries[observation.candidate_id] = (
            f"utility={observation.utility:.3f}, "
            f"delta_perf={observation.delta_perf:.3f}, "
            f"runtime={observation.runtime_sec:.2f}s"
        )
    return summaries


def _candidate_rationale(candidate: Mapping[str, object]) -> str | None:
    notes = _summary_string(candidate.get("notes"))
    if notes is not None:
        return notes
    rationale_items = tuple(
        item
        for item in _summary_list(candidate.get("rationale"))
        if _summary_string(item) is not None
    )
    if rationale_items:
        return _summary_string(rationale_items[0])
    influence_items = tuple(
        item
        for item in _summary_list(candidate.get("influence_summary"))
        if _summary_string(item) is not None
    )
    if influence_items:
        return _summary_string(influence_items[0])
    return None


def _proposal_entry_for_candidate(
    proposal_ledger: ProposalLedger | None,
    candidate_id: str,
) -> ProposalLedgerEntry | None:
    if proposal_ledger is None:
        return None
    return next(
        (
            entry
            for entry in proposal_ledger.entries
            if entry.candidate_id == candidate_id
        ),
        None,
    )


def _current_proposal(
    proposal_ledger: ProposalLedger | None,
) -> ProposalLedgerEntry | None:
    if proposal_ledger is None or proposal_ledger.current_proposal_id is None:
        return None
    return next(
        (
            entry
            for entry in proposal_ledger.entries
            if entry.proposal_id == proposal_ledger.current_proposal_id
        ),
        None,
    )


def _candidate_state_keys(candidate: Mapping[str, object]) -> frozenset[str]:
    raw_genotype = candidate.get("genotype")
    if not isinstance(raw_genotype, Sequence) or isinstance(raw_genotype, str | bytes):
        return frozenset()
    keys: list[str] = []
    for item in cast(Sequence[object], raw_genotype):
        if not isinstance(item, Mapping):
            continue
        mapping = cast(Mapping[str, object], item)
        gene_id = _summary_string(mapping.get("gene_id"))
        state_id = _summary_string(mapping.get("state_id"))
        if gene_id is None or state_id is None:
            continue
        keys.append(f"{gene_id}:{state_id}")
    return frozenset(keys)


def _feature_keys_for_belief(belief: object) -> tuple[str, ...]:
    if isinstance(belief, IdeaBelief):
        return (belief.gene.canonical_key,)
    if isinstance(belief, RelationBelief):
        return tuple(member.canonical_key for member in belief.members)
    if isinstance(belief, ConstraintBelief):
        return tuple(member.canonical_key for member in belief.scope)
    if isinstance(belief, GraphDirectiveBelief):
        return tuple(member.canonical_key for member in belief.members)
    if isinstance(belief, ExpertPriorBelief):
        target = belief.target
        if isinstance(target, MainEffectTarget | FeasibilityTarget | VramTarget):
            return (target.gene.canonical_key,)
        return tuple(member.canonical_key for member in target.members)
    return ()


def _belief_matches_candidate(
    *,
    belief: object,
    candidate_state_keys: frozenset[str],
) -> bool:
    if isinstance(belief, ProposalBelief):
        return False
    feature_keys = _feature_keys_for_belief(belief)
    return bool(feature_keys) and all(
        feature_key in candidate_state_keys for feature_key in feature_keys
    )


def _stable_unique_strings(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _candidate_lineage(
    *,
    candidate: Mapping[str, object],
    beliefs: ValidatedBeliefBatch | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    explicit_belief_ids = tuple(
        belief_id
        for belief_id in (
            _summary_string(item)
            for item in _summary_list(candidate.get("parent_belief_ids"))
        )
        if belief_id is not None
    )
    if beliefs is None:
        return (), explicit_belief_ids
    belief_by_id = {belief.id: belief for belief in beliefs.beliefs}
    inferred_belief_ids: list[str] = []
    candidate_state_keys = _candidate_state_keys(candidate)
    if candidate_state_keys:
        for belief in beliefs.beliefs:
            if _belief_matches_candidate(
                belief=belief,
                candidate_state_keys=candidate_state_keys,
            ):
                inferred_belief_ids.append(belief.id)
    source_belief_ids = _stable_unique_strings(
        explicit_belief_ids + tuple(inferred_belief_ids)
    )
    source_idea_ids = tuple(
        belief_id
        for belief_id in source_belief_ids
        if isinstance(belief_by_id.get(belief_id), IdeaBelief)
    )
    return source_idea_ids, source_belief_ids


def _lane_rows(
    *,
    ranked_candidates: Sequence[Mapping[str, object]],
    frontier_summary: FrontierSummary,
    proposal_ledger: ProposalLedger | None,
    trust_status: str,
    last_eval_by_candidate: Mapping[str, str],
    beliefs: ValidatedBeliefBatch | None,
) -> tuple[dict[str, JsonValue], ...]:
    pending_query_targets = {
        candidate_id
        for query in frontier_summary.pending_queries
        for candidate_id in query.candidate_ids
    }
    pending_merge_targets = {
        candidate_id
        for merge in frontier_summary.pending_merge_suggestions
        for candidate_id in merge.candidate_ids
    }
    rows: list[dict[str, JsonValue]] = []
    for index, candidate in enumerate(ranked_candidates, start=1):
        candidate_id = (
            _summary_string(candidate.get("candidate_id")) or f"cand_{index:03d}"
        )
        proposal_entry = _proposal_entry_for_candidate(proposal_ledger, candidate_id)
        decision_status = "hold"
        if (
            proposal_entry is not None
            and proposal_entry.readiness_state == "recommended"
        ):
            decision_status = "promote"
        elif proposal_entry is not None and proposal_entry.readiness_state == "blocked":
            decision_status = "blocked"
        elif candidate_id in pending_query_targets:
            decision_status = "query"
        elif candidate_id in pending_merge_targets:
            decision_status = "merge"
        elif index > 3:
            decision_status = "drop"
        source_idea_ids, source_belief_ids = _candidate_lineage(
            candidate=candidate,
            beliefs=beliefs,
        )
        score_summary: dict[str, JsonValue] = {}
        for key in (
            "predicted_utility",
            "valid_probability",
            "acquisition_score",
            "uncertainty",
            "objective_backend",
            "acquisition_backend",
            "acquisition_fallback_reason",
        ):
            if key in candidate:
                score_summary[key] = cast(JsonValue, candidate[key])
        lane_thesis = _candidate_rationale(candidate) or "Explicit candidate lane."
        evidence_summary = (
            proposal_entry.evidence_summary
            if proposal_entry is not None
            else (
                "Current leader lane by acquisition score."
                if index == 1
                else lane_thesis
            )
        )
        next_step = (
            "Answer the comparison query"
            if decision_status == "query"
            else (
                "Review the merge suggestion"
                if decision_status == "merge"
                else (
                    "Resolve the blocking issue"
                    if decision_status == "blocked"
                    else (
                        "Promote or recommend this lane"
                        if index == 1
                        else "Keep under review"
                    )
                )
            )
        )
        rows.append(
            {
                "lane_id": candidate_id,
                "family_id": _summary_string(candidate.get("family_id"))
                or "family_default",
                "source_idea_ids": cast(
                    list[JsonValue], to_json_value(source_idea_ids)
                ),
                "source_belief_ids": cast(
                    list[JsonValue], to_json_value(source_belief_ids)
                ),
                "lane_thesis": lane_thesis,
                "current_rank": index,
                "score_summary": cast(
                    dict[str, JsonValue], to_json_value(score_summary)
                ),
                "decision_status": decision_status,
                "proposal_status": (
                    proposal_entry.readiness_state
                    if proposal_entry is not None
                    else "not_ready"
                ),
                "trust_status": trust_status,
                "evidence_summary": evidence_summary,
                "next_step": next_step,
                "last_eval_summary": last_eval_by_candidate.get(
                    candidate_id,
                    "No eval recorded for this lane yet.",
                ),
            }
        )
    return tuple(rows)


def _proposal_rows(
    proposal_ledger: ProposalLedger | None,
    *,
    lane_lineage_by_id: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, JsonValue], ...]:
    if proposal_ledger is None:
        return ()
    return tuple(
        {
            "proposal_id": entry.proposal_id,
            "source_lane_ids": cast(
                list[JsonValue],
                to_json_value(
                    entry.source_candidate_ids
                    if entry.source_candidate_ids
                    else (entry.candidate_id,)
                ),
            ),
            "readiness": entry.readiness_state,
            "recommendation_text": entry.recommendation_reason
            or entry.evidence_summary,
            "evidence_basis": entry.evidence_summary,
            "unresolved_risks": cast(
                list[JsonValue], to_json_value(entry.unresolved_risks)
            ),
            "resume_hint": entry.resume_token,
            "updated_at": entry.updated_at,
            "source_lane_id": entry.candidate_id,
            "source_idea_ids": cast(
                list[JsonValue],
                to_json_value(
                    tuple(
                        item
                        for item in _summary_list(
                            lane_lineage_by_id.get(entry.candidate_id, {}).get(
                                "source_idea_ids"
                            )
                        )
                        if _summary_string(item) is not None
                    )
                ),
            ),
            "source_belief_ids": cast(
                list[JsonValue],
                to_json_value(
                    tuple(
                        item
                        for item in _summary_list(
                            lane_lineage_by_id.get(entry.candidate_id, {}).get(
                                "source_belief_ids"
                            )
                        )
                        if _summary_string(item) is not None
                    )
                ),
            ),
        }
        for entry in proposal_ledger.entries
    )


def _lineage_payload(
    *,
    lanes: Sequence[Mapping[str, object]],
    proposal_ledger: ProposalLedger | None,
    beliefs: ValidatedBeliefBatch | None,
) -> dict[str, JsonValue]:
    current_proposal = _current_proposal(proposal_ledger)
    lane_lineage = tuple(
        {
            "lane_id": _summary_string(lane.get("lane_id")) or "unknown",
            "family_id": _summary_string(lane.get("family_id")) or "family_default",
            "source_idea_ids": cast(
                Sequence[str],
                tuple(
                    item
                    for item in _summary_list(lane.get("source_idea_ids"))
                    if _summary_string(item) is not None
                ),
            ),
            "source_belief_ids": cast(
                Sequence[str],
                tuple(
                    item
                    for item in _summary_list(lane.get("source_belief_ids"))
                    if _summary_string(item) is not None
                ),
            ),
            "decision_status": _summary_string(lane.get("decision_status")) or "hold",
            "proposal_status": _summary_string(lane.get("proposal_status"))
            or "not_ready",
            "evidence_summary": _summary_string(lane.get("evidence_summary"))
            or "No evidence summary recorded yet.",
            "last_eval_summary": _summary_string(lane.get("last_eval_summary"))
            or "No eval recorded for this lane yet.",
        }
        for lane in lanes
    )
    return {
        "chain": cast(
            list[JsonValue],
            to_json_value(
                (
                    "initial ideas",
                    "canonical beliefs",
                    "seeded or derived lanes",
                    "eval evidence",
                    "lane decision",
                    "proposal recommendation",
                )
            ),
        ),
        "belief_ids": cast(list[JsonValue], to_json_value(_belief_ids(beliefs))),
        "lanes": cast(list[JsonValue], to_json_value(lane_lineage)),
        "recommended_proposal": cast(
            JsonValue,
            None
            if current_proposal is None
            else {
                "proposal_id": current_proposal.proposal_id,
                "candidate_id": current_proposal.candidate_id,
                "readiness": current_proposal.readiness_state,
                "source_lane_ids": (
                    current_proposal.source_candidate_ids
                    if current_proposal.source_candidate_ids
                    else (current_proposal.candidate_id,)
                ),
                "source_idea_ids": next(
                    (
                        lane["source_idea_ids"]
                        for lane in lane_lineage
                        if lane["lane_id"] == current_proposal.candidate_id
                    ),
                    (),
                ),
                "source_belief_ids": next(
                    (
                        lane["source_belief_ids"]
                        for lane in lane_lineage
                        if lane["lane_id"] == current_proposal.candidate_id
                    ),
                    (),
                ),
                "evidence_basis": current_proposal.evidence_summary,
                "unresolved_risks": current_proposal.unresolved_risks,
            },
        ),
    }


def _evidence_payload(
    *,
    status: SessionStatus,
    artifacts: SessionArtifactConfig,
    posterior_payload: Mapping[str, object] | None,
) -> dict[str, JsonValue]:
    artifact_paths = status.artifact_paths
    views: list[dict[str, JsonValue]] = []
    for artifact_id, path_key, label, description, filename in (
        (
            "results_markdown",
            "results_markdown",
            "Run summary",
            "Human-readable upstream session summary.",
            artifacts.results_markdown,
        ),
        (
            "candidate_rankings",
            "candidate_rankings_plot",
            "Candidate rankings",
            "Current lane ordering and acquisition scores.",
            artifacts.candidate_rankings_plot,
        ),
        (
            "belief_graph_prior",
            "prior_graph_plot",
            "Prior graph",
            "What the session believed before new eval evidence.",
            artifacts.prior_graph_plot,
        ),
        (
            "belief_graph_posterior",
            "posterior_graph_plot",
            "Posterior graph",
            "What the session still believes after eval evidence.",
            artifacts.posterior_graph_plot,
        ),
        (
            "convergence",
            "convergence_plot",
            "Convergence",
            "Whether new evals are still changing the picture.",
            artifacts.convergence_plot,
        ),
    ):
        path = Path(artifact_paths[path_key])
        views.append(
            {
                "id": artifact_id,
                "label": label,
                "description": description,
                "path": str(path),
                "exists": path.exists(),
                "filename": filename,
            }
        )
    notes = (
        "The belief graphs are evidence views over relations between settings and typed beliefs; they are not the frontier itself.",
        "The lane table is the frontier under comparison. Use it to understand what is being promoted, queried, merged, or dropped.",
    )
    return {
        "views": cast(list[JsonValue], to_json_value(tuple(views))),
        "notes": cast(list[JsonValue], to_json_value(notes)),
        "objective_backend": status.last_objective_backend,
        "acquisition_backend": status.last_acquisition_backend,
        "objective_fallback_reason": (
            None
            if posterior_payload is None
            else _summary_string(posterior_payload.get("objective_fallback_reason"))
        ),
        "acquisition_fallback_reason": (
            None
            if posterior_payload is None
            else _summary_string(posterior_payload.get("acquisition_fallback_reason"))
        ),
    }


def _prior_brief(
    *,
    beliefs: ValidatedBeliefBatch | None,
    manifest: SessionManifest,
    frontier_summary: FrontierSummary,
    eval_contract: EvalContractSnapshot | None,
) -> dict[str, JsonValue]:
    proposal_beliefs = _proposal_beliefs(beliefs)
    compiled_belief_count = (
        0 if beliefs is None else len(beliefs.beliefs) - len(proposal_beliefs)
    )
    summary = (
        f"{compiled_belief_count} typed belief(s) seeded the session; "
        f"{frontier_summary.family_count} frontier family/families were visible before the latest fit."
    )
    bullets = [
        (
            "Beliefs are still preview-pending."
            if manifest.beliefs_status == "preview_pending"
            else "Beliefs have been applied to the active session."
        ),
        (
            f"{len(proposal_beliefs)} proposal-style belief(s) remain metadata-only."
            if proposal_beliefs
            else "No proposal-only beliefs remain in the current batch."
        ),
        (
            f"Locked eval contract digest: {eval_contract.contract_digest}"
            if eval_contract is not None
            else "No locked eval contract is stored for this session."
        ),
        (
            f"Preview digest: {manifest.preview_digest}"
            if manifest.preview_digest is not None
            else "No preview digest was recorded."
        ),
    ]
    return _brief_payload(
        summary=summary,
        bullets=bullets,
        extra={
            "belief_ids": cast(list[JsonValue], to_json_value(_belief_ids(beliefs))),
            "metadata_only_belief_ids": cast(
                list[JsonValue],
                to_json_value(tuple(belief.id for belief in proposal_beliefs)),
            ),
            "locked_eval_contract_digest": (
                None if eval_contract is None else eval_contract.contract_digest
            ),
        },
    )


def _run_brief(
    *,
    status: SessionStatus,
    ranked_candidates: Sequence[Mapping[str, object]],
    next_action_summary: str | None,
    next_action_reason: str | None,
) -> dict[str, JsonValue]:
    leader = (
        None
        if not ranked_candidates
        else _summary_string(ranked_candidates[0].get("candidate_id"))
    )
    runner_up = (
        None
        if len(ranked_candidates) < 2
        else _summary_string(ranked_candidates[1].get("candidate_id"))
    )
    summary = (
        "No ranked leader lane exists yet."
        if leader is None
        else f"Leader lane: {leader}; runner-up: {runner_up or 'none'}."
    )
    bullets = [
        (
            f"Next action: {next_action_summary}"
            if next_action_summary is not None
            else "No concrete next action is queued."
        ),
        (
            f"Reason: {next_action_reason}"
            if next_action_reason is not None
            else "No uncertainty-reduction rationale is recorded yet."
        ),
        (
            f"Compared lanes: {status.frontier_candidate_count}; families: {status.frontier_family_count}."
        ),
        (
            f"Trust: {status.eval_contract_drift_status or 'unverified'}; "
            f"objective/acquisition: {status.last_objective_backend or 'unknown'} / "
            f"{status.last_acquisition_backend or 'unknown'}."
        ),
    ]
    return _brief_payload(summary=summary, bullets=bullets)


def _posterior_brief(
    *,
    belief_delta_summary: BeliefDeltaSummary | None,
) -> dict[str, JsonValue]:
    strengthened = (
        () if belief_delta_summary is None else belief_delta_summary.strengthened
    )
    weakened = () if belief_delta_summary is None else belief_delta_summary.weakened
    uncertain = () if belief_delta_summary is None else belief_delta_summary.uncertain
    summary = (
        "Posterior change is not recorded yet."
        if belief_delta_summary is None
        else (
            f"{len(strengthened)} belief(s) strengthened, "
            f"{len(weakened)} weakened, "
            f"and {len(uncertain)} uncertainty focus item(s) remain."
        )
    )
    bullets = [
        (
            "Strengthened: " + " | ".join(entry.summary for entry in strengthened[:2])
            if strengthened
            else "No strengthened belief summary is recorded."
        ),
        (
            "Weakened: " + " | ".join(entry.summary for entry in weakened[:2])
            if weakened
            else "No weakened belief summary is recorded."
        ),
        (
            "Promoted lanes: " + ", ".join(belief_delta_summary.promoted_candidate_ids)
            if belief_delta_summary is not None
            and belief_delta_summary.promoted_candidate_ids
            else "No promoted lanes were recorded."
        ),
        (
            "Dropped families: " + ", ".join(belief_delta_summary.dropped_family_ids)
            if belief_delta_summary is not None
            and belief_delta_summary.dropped_family_ids
            else "No dropped family reasons were recorded."
        ),
    ]
    return _brief_payload(summary=summary, bullets=bullets)


def _proposal_brief(
    *,
    proposal_ledger: ProposalLedger | None,
    decision: CommitDecision | None,
) -> dict[str, JsonValue]:
    current_proposal = _current_proposal(proposal_ledger)
    summary = (
        "No durable proposal has been recorded yet."
        if current_proposal is None
        else (
            f"Current proposal {current_proposal.proposal_id} is "
            f"{current_proposal.readiness_state} from lane "
            f"{current_proposal.candidate_id}."
        )
    )
    alternate_entries = (
        ()
        if proposal_ledger is None
        else tuple(
            entry
            for entry in proposal_ledger.entries
            if current_proposal is None
            or entry.proposal_id != current_proposal.proposal_id
        )
    )
    bullets = [
        (
            f"Evidence: {current_proposal.evidence_summary}"
            if current_proposal is not None
            else "Run suggest or recommend-commit to materialize a durable proposal."
        ),
        (
            "Unresolved risks: " + " | ".join(current_proposal.unresolved_risks[:3])
            if current_proposal is not None and current_proposal.unresolved_risks
            else "No unresolved proposal risks are recorded."
        ),
        (
            "Alternates: "
            + " | ".join(
                f"{entry.proposal_id} ({entry.readiness_state})"
                for entry in alternate_entries[:3]
            )
            if alternate_entries
            else "No alternate proposals are recorded."
        ),
        (
            f"Commit recommendation: {decision.reason}"
            if decision is not None
            else "No commit recommendation is recorded yet."
        ),
    ]
    return _brief_payload(summary=summary, bullets=bullets)


def build_review_bundle(
    *,
    store: FilesystemSessionStore,
    session_id: str,
    manifest: SessionManifest,
    adapter_config: ValidAdapterConfig | None = None,
) -> dict[str, JsonValue]:
    status, expected_contract, current_contract = normalized_session_status(
        store=store,
        session_id=session_id,
        adapter_config=adapter_config,
    )
    beliefs = (
        None if manifest.beliefs_status == "absent" else store.load_beliefs(session_id)
    )
    observations = store.read_observations(session_id)
    frontier_summary = store.load_frontier_status(session_id) or FrontierSummary(
        frontier_id="frontier_default",
        candidate_count=0,
        family_count=0,
        family_representatives=(),
        dropped_family_reasons={},
        pending_queries=(),
        pending_merge_suggestions=(),
        budget_allocations={},
    )
    belief_delta_summary = store.load_belief_delta_summary(session_id)
    proposal_ledger = store.load_proposal_ledger(session_id)
    decision = store.load_commit_decision(session_id)
    eval_contract = expected_contract
    query_payload = _load_json_artifact(
        store.artifact_path(session_id, store.artifact_filenames.query)
    )
    posterior_payload = _load_json_artifact(
        store.artifact_path(session_id, store.artifact_filenames.posterior_summary)
    )
    ranked_candidates = tuple(
        cast(Mapping[str, object], item)
        for item in _summary_list(
            None
            if query_payload is None
            else query_payload.get("ranked_candidates")
        )
        if isinstance(item, Mapping)
    )
    first_query = (
        frontier_summary.pending_queries[0]
        if frontier_summary.pending_queries
        else None
    )
    next_action_summary = (
        None
        if first_query is None
        else (
            f"Compare {_query_comparison(first_query)}."
            if _query_comparison(first_query) is not None
            else first_query.prompt
        )
    )
    next_action_reason = _query_reason(first_query)
    trust_status = status.eval_contract_drift_status or "unverified"
    lanes = _lane_rows(
        ranked_candidates=ranked_candidates,
        frontier_summary=frontier_summary,
        proposal_ledger=proposal_ledger,
        trust_status=trust_status,
        last_eval_by_candidate=_last_eval_summary_by_candidate(observations),
        beliefs=beliefs,
    )
    lane_lineage_by_id = {
        str(item["lane_id"]): cast(Mapping[str, object], item) for item in lanes
    }
    proposals = _proposal_rows(
        proposal_ledger,
        lane_lineage_by_id=lane_lineage_by_id,
    )
    payload = {
        "session": {
            "session_id": manifest.session_id,
            "era_id": manifest.era_id,
            "observation_count": status.observation_count,
            "beliefs_status": manifest.beliefs_status,
            "compiled_priors_active": manifest.compiled_priors_active,
            "adapter_execution_mode": manifest.adapter_execution_mode,
            "execution_status": (
                "drift"
                if trust_status == "drifted"
                else ("ok" if status.observation_count > 0 else "idle")
            ),
        },
        "prior_brief": _prior_brief(
            beliefs=beliefs,
            manifest=manifest,
            frontier_summary=frontier_summary,
            eval_contract=eval_contract,
        ),
        "run_brief": _run_brief(
            status=status,
            ranked_candidates=ranked_candidates,
            next_action_summary=next_action_summary,
            next_action_reason=next_action_reason,
        ),
        "posterior_brief": _posterior_brief(
            belief_delta_summary=belief_delta_summary,
        ),
        "proposal_brief": _proposal_brief(
            proposal_ledger=proposal_ledger,
            decision=decision,
        ),
        "lanes": cast(list[JsonValue], to_json_value(lanes)),
        "proposals": cast(list[JsonValue], to_json_value(proposals)),
        "lineage": _lineage_payload(
            lanes=lanes,
            proposal_ledger=proposal_ledger,
            beliefs=beliefs,
        ),
        "trust": {
            "status": trust_status,
            "locked_eval_contract_digest": status.eval_contract_digest,
            "current_eval_contract_digest": status.current_eval_contract_digest,
            "eval_contract_matches_current": status.eval_contract_matches_current,
            "current_eval_contract": (
                None if current_contract is None else to_json_value(current_contract)
            ),
            "last_eval_measurement_mode": status.last_eval_measurement_mode,
            "last_eval_stabilization_mode": status.last_eval_stabilization_mode,
            "last_eval_used_lease": status.last_eval_used_lease,
            "last_eval_noisy_system": status.last_eval_noisy_system,
        },
        "evidence": _evidence_payload(
            status=status,
            artifacts=store.artifact_filenames,
            posterior_payload=posterior_payload,
        ),
        "next_action": {
            "summary": next_action_summary,
            "reason": next_action_reason,
            "query_type": None if first_query is None else first_query.query_type,
            "comparison": _query_comparison(first_query),
            "pending_query_count": status.pending_query_count,
            "pending_merge_count": status.pending_merge_suggestion_count,
        },
    }
    return cast(dict[str, JsonValue], to_json_value(payload))
