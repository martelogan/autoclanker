from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer import (
    EraState,
    compile_beliefs,
    load_inline_ideas_payload,
    load_serialized_payload,
    load_serialized_payload_from_text,
    preview_compiled_beliefs,
    validate_adapter_config,
    validate_eval_result,
)
from autoclanker.bayes_layer.acquisition import generate_candidate_pool, rank_candidates
from autoclanker.bayes_layer.adapters import load_adapter
from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationMode,
    canonicalize_belief_input,
    default_canonicalization_mode,
    load_canonicalization_model,
)
from autoclanker.bayes_layer.commit_policy import recommend_commit
from autoclanker.bayes_layer.config import (
    DEFAULT_ADAPTER_CONFIG_PATH,
    load_bayes_layer_config,
)
from autoclanker.bayes_layer.eval_contract import (
    compare_eval_contracts,
    hardened_eval_result,
    isolated_execution_workspace,
    measured_execution_window,
)
from autoclanker.bayes_layer.frontier import (
    FrontierDocument,
    frontier_candidates_payload,
    frontier_from_candidate_pairs,
    parse_frontier_payload,
    summarize_frontier,
)
from autoclanker.bayes_layer.posterior_graph import build_posterior_graph
from autoclanker.bayes_layer.query_policy import suggest_queries
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.reporting import render_session_report_bundle
from autoclanker.bayes_layer.review_bundle import (
    build_review_bundle,
    normalized_session_status,
)
from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.surrogate_feasibility import fit_feasibility_surrogate
from autoclanker.bayes_layer.surrogate_objective import fit_objective_surrogate
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    EvalExecutionContext,
    FrontierCandidate,
    FrontierSummary,
    GeneStateRef,
    InfluenceSummary,
    JsonValue,
    PosteriorSummary,
    QuerySuggestion,
    RankedCandidate,
    SessionContext,
    SessionFailure,
    SessionManifest,
    ValidAdapterConfig,
    ValidatedBeliefBatch,
    ValidationFailure,
    ValidEvalResult,
    to_json_value,
)
from autoclanker.bayes_layer.ux_artifacts import (
    build_belief_delta_summary,
    build_proposal_ledger,
)


def _load_payload(input_path: str | None) -> dict[str, object]:
    if input_path and input_path != "-":
        return load_serialized_payload(Path(input_path))
    return load_serialized_payload_from_text(sys.stdin.read())


def _load_init_belief_payload(args: argparse.Namespace) -> dict[str, object] | None:
    if args.ideas_json is not None:
        if args.beliefs_input is not None:
            raise ValidationFailure(
                "Use either --beliefs-input or --ideas-json, not both."
            )
        return load_inline_ideas_payload(args.ideas_json)
    if args.beliefs_input is not None:
        return _load_payload(args.beliefs_input)
    return None


def _load_adapter_config(config_path: str | None) -> ValidAdapterConfig:
    path = Path(config_path) if config_path else DEFAULT_ADAPTER_CONFIG_PATH
    return validate_adapter_config(load_serialized_payload(path), base_dir=path.parent)


def _registry_from_config(config: ValidAdapterConfig) -> GeneRegistry:
    return load_adapter(config).build_registry()


def _canonicalization_mode(args: argparse.Namespace) -> CanonicalizationMode | None:
    raw_mode = cast(str | None, getattr(args, "canonicalization_mode", None))
    return cast(CanonicalizationMode | None, raw_mode)


def _store(
    session_root: str | None, adapter_config: ValidAdapterConfig
) -> FilesystemSessionStore:
    return FilesystemSessionStore(root=session_root or adapter_config.session_root)


def _empty_beliefs(session_id: str, era_id: str) -> ValidatedBeliefBatch:
    context = SessionContext(session_id=session_id, era_id=era_id, user_profile="basic")
    return ValidatedBeliefBatch(
        session_context=context,
        beliefs=(),
        canonical_payload=cast(
            dict[str, JsonValue],
            to_json_value({"session_context": context, "beliefs": []}),
        ),
    )


def _empty_bundle(era_id: str) -> CompiledPriorBundle:
    return CompiledPriorBundle(
        era_id=era_id,
        main_effect_priors=(),
        pair_priors=(),
        feasibility_priors=(),
        vram_priors=(),
        hard_masks=(),
        preference_observations=(),
        candidate_generation_hints=(),
        linkage_hints=(),
        belief_previews=(),
    )


def _belief_fallback_context(args: argparse.Namespace) -> SessionContext | None:
    if args.era_id is None and args.session_id is None:
        return None
    if args.era_id is None:
        raise ValidationFailure(
            "--era-id is required when beginner ideas omit session_context."
        )
    return SessionContext(
        era_id=args.era_id,
        session_id=args.session_id,
        user_profile="basic",
    )


def _payload_digest(payload: dict[str, JsonValue]) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _json_object(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], to_json_value(value))


def _candidate_input_to_refs(
    payload: dict[str, object],
) -> tuple[tuple[str, tuple[GeneStateRef, ...]], ...]:
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValidationFailure("Candidate payload must contain a 'candidates' list.")
    raw_candidate_items = cast(list[object], raw_candidates)
    parsed: list[tuple[str, tuple[GeneStateRef, ...]]] = []
    for index, raw_candidate in enumerate(raw_candidate_items, start=1):
        if not isinstance(raw_candidate, dict):
            raise ValidationFailure("Each candidate must be an object.")
        candidate_mapping = cast(dict[str, object], raw_candidate)
        candidate_id = str(
            candidate_mapping.get("candidate_id", f"cand_input_{index:03d}")
        )
        raw_genotype = candidate_mapping.get("genotype")
        if not isinstance(raw_genotype, list):
            raise ValidationFailure("Each candidate must provide a genotype list.")
        raw_genotype_items = cast(list[object], raw_genotype)
        genotype: list[GeneStateRef] = []
        for genotype_index, raw_ref in enumerate(raw_genotype_items, start=1):
            if not isinstance(raw_ref, dict):
                raise ValidationFailure("Genotype entries must be objects.")
            raw_ref_mapping = cast(dict[str, object], raw_ref)
            gene_id = raw_ref_mapping.get("gene_id")
            state_id = raw_ref_mapping.get("state_id")
            if not isinstance(gene_id, str) or not gene_id.strip():
                raise ValidationFailure(
                    f"Genotype entry {genotype_index} is missing a non-empty gene_id."
                )
            if not isinstance(state_id, str) or not state_id.strip():
                raise ValidationFailure(
                    f"Genotype entry {genotype_index} is missing a non-empty state_id."
                )
            genotype.append(
                GeneStateRef(
                    gene_id=gene_id.strip(),
                    state_id=state_id.strip(),
                )
            )
        parsed.append((candidate_id, tuple(genotype)))
    return tuple(parsed)


def _frontier_from_payload(payload: dict[str, object]) -> FrontierDocument:
    if "frontier_id" in payload or "default_family_id" in payload:
        return parse_frontier_payload(payload)
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValidationFailure("Candidate payload must contain a 'candidates' list.")
    metadata_keys = {
        "family_id",
        "origin_kind",
        "parent_candidate_ids",
        "parent_belief_ids",
        "origin_query_ids",
        "notes",
        "budget_weight",
    }
    if any(
        isinstance(item, dict) and any(key in item for key in metadata_keys)
        for item in cast(list[object], raw_candidates)
    ):
        return parse_frontier_payload(payload)
    return frontier_from_candidate_pairs(_candidate_input_to_refs(payload))


def _load_frontier(
    candidates_input: str | None,
    *,
    registry: GeneRegistry,
    compiled: CompiledPriorBundle,
) -> FrontierDocument:
    if candidates_input is None:
        return frontier_from_candidate_pairs(
            generate_candidate_pool(registry, compiled_priors=compiled)
        )
    return _frontier_from_payload(_load_payload(candidates_input))


def _frontier_lookup(
    frontier: FrontierDocument,
) -> dict[str, FrontierCandidate]:
    return {candidate.candidate_id: candidate for candidate in frontier.candidates}


def _load_active_beliefs_and_bundle(
    *,
    store: FilesystemSessionStore,
    manifest: SessionManifest,
    registry: GeneRegistry,
) -> tuple[ValidatedBeliefBatch, CompiledPriorBundle]:
    if manifest.beliefs_status == "preview_pending":
        raise SessionFailure(
            "Beliefs were previewed but not applied. Run `autoclanker session apply-beliefs` with the preview digest."
        )
    if manifest.beliefs_status == "absent":
        return (
            _empty_beliefs(manifest.session_id, manifest.era_id),
            _empty_bundle(manifest.era_id),
        )
    overlay = store.load_surface_overlay(manifest.session_id)
    active_registry = (
        registry.with_overlay(overlay) if overlay is not None else registry
    )
    beliefs = store.load_beliefs(manifest.session_id)
    compiled = compile_beliefs(
        beliefs,
        active_registry,
        EraState(era_id=manifest.era_id),
    )
    return beliefs, compiled


def _compute_summary(
    *,
    era_state: EraState,
    registry: GeneRegistry,
    observations: tuple[ValidEvalResult, ...],
    compiled: CompiledPriorBundle,
) -> PosteriorSummary:
    active_config = load_bayes_layer_config()
    started = time.perf_counter()
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
        config=active_config,
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
        config=active_config,
    )
    ranked = rank_candidates(
        generate_candidate_pool(registry, compiled_priors=compiled),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        config=active_config,
    )
    fit_runtime_ms = (time.perf_counter() - started) * 1000.0
    graph = build_posterior_graph(objective, compiled)
    acquisition_backend = (
        ranked[0].acquisition_backend
        if ranked
        else (
            "constrained_thompson_sampling"
            if (
                active_config.acquisition.kind == "constrained_thompson_sampling"
                and objective.backend == "exact_joint_linear"
                and objective.sampleable
            )
            else "optimistic_upper_confidence"
        )
    )
    main_feature_count = sum(
        1 for feature in objective.features if feature.target_kind == "main_effect"
    )
    pair_feature_count = sum(
        1 for feature in objective.features if feature.target_kind == "pair_effect"
    )
    acquisition_fallback_reason = (
        ranked[0].acquisition_fallback_reason if ranked else None
    )
    return PosteriorSummary(
        era_id=era_state.era_id,
        observation_count=objective.observation_count,
        aggregate_count=len({item.patch_hash for item in observations}),
        objective_baseline=objective.baseline_utility,
        valid_baseline=feasibility.baseline_valid_probability,
        top_features=objective.features[:8],
        top_candidates=ranked[:5],
        graph=graph,
        objective_backend=objective.backend,
        acquisition_backend=acquisition_backend,
        acquisition_fallback_reason=acquisition_fallback_reason,
        objective_sampleable=objective.sampleable,
        objective_effective_observation_count=objective.effective_observation_count,
        objective_feature_count=len(objective.features),
        objective_main_feature_count=main_feature_count,
        objective_pair_feature_count=pair_feature_count,
        objective_condition_number=objective.condition_number,
        objective_used_jitter=objective.used_jitter,
        objective_observation_noise=objective.observation_noise,
        objective_fallback_reason=objective.fallback_reason,
        fit_runtime_ms=fit_runtime_ms,
        influence_summary=tuple(
            InfluenceSummary(
                source_belief_id=spec.source_belief_id,
                target_ref=spec.item.target_ref,
                summary=f"{spec.source_belief_id} contributes prior mass to {spec.item.target_ref}.",
            )
            for spec in compiled.all_items[:8]
        ),
    )


def _status_payload(
    *,
    store: FilesystemSessionStore,
    manifest: SessionManifest,
    adapter_config: ValidAdapterConfig,
) -> dict[str, JsonValue]:
    status, expected_contract, current_contract = normalized_session_status(
        store=store,
        session_id=manifest.session_id,
        adapter_config=adapter_config,
    )
    payload = cast(dict[str, JsonValue], to_json_value(status))
    if expected_contract is not None:
        payload["eval_contract"] = cast(
            dict[str, JsonValue], to_json_value(expected_contract)
        )
    if current_contract is not None:
        payload["current_eval_contract"] = cast(
            dict[str, JsonValue], to_json_value(current_contract)
        )
        payload["current_eval_contract_digest"] = current_contract.contract_digest
    return payload


def _append_hardened_eval_result(
    *,
    store: FilesystemSessionStore,
    manifest: SessionManifest,
    adapter_config: ValidAdapterConfig,
    candidate: FrontierCandidate,
    seed: int,
    replication_index: int,
) -> ValidEvalResult:
    adapter = load_adapter(adapter_config)
    expected_contract = store.load_eval_contract(manifest.session_id)
    if expected_contract is None:
        expected_contract = adapter.capture_eval_contract()
    with (
        isolated_execution_workspace(expected_contract) as (
            isolation_mode,
            workspace_root,
        ),
        measured_execution_window(expected_contract) as measurement,
    ):
        result = adapter.evaluate_candidate(
            era_id=manifest.era_id,
            candidate_id=candidate.candidate_id,
            genotype=candidate.genotype,
            seed=seed,
            replication_index=replication_index,
            execution_context=EvalExecutionContext(
                session_id=manifest.session_id,
                era_id=manifest.era_id,
                contract=expected_contract,
                isolation_mode=isolation_mode,
                workspace_root=workspace_root,
                seed=seed,
                replication_index=replication_index,
                measurement_mode=measurement.measurement_mode,
                stabilization_mode=measurement.stabilization_mode,
                lease_scope=measurement.lease_scope,
            ),
        )
        hardened = hardened_eval_result(
            result,
            contract=expected_contract,
            isolation_mode=isolation_mode,
            workspace_root=workspace_root,
            measurement=measurement,
        )
    store.append_observation(manifest.session_id, hardened)
    store.write_eval_run_record(
        manifest.session_id,
        candidate_id=candidate.candidate_id,
        payload={
            "candidate": cast(dict[str, JsonValue], to_json_value(candidate)),
            "result": cast(dict[str, JsonValue], to_json_value(hardened)),
        },
    )
    return hardened


def _suggest_from_frontier(
    *,
    store: FilesystemSessionStore,
    manifest: SessionManifest,
    registry: GeneRegistry,
    beliefs: ValidatedBeliefBatch,
    compiled: CompiledPriorBundle,
    frontier: FrontierDocument,
    observations: tuple[ValidEvalResult, ...],
) -> tuple[tuple[RankedCandidate, ...], tuple[QuerySuggestion, ...], FrontierSummary]:
    active_config = load_bayes_layer_config()
    era_state = EraState(
        era_id=manifest.era_id,
        observation_count=len(observations),
    )
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
        config=active_config,
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
        config=active_config,
    )
    ranked = rank_candidates(
        frontier_candidates_payload(frontier),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        compiled_priors=compiled,
        frontier_candidates=_frontier_lookup(frontier),
        config=active_config,
    )
    queries = suggest_queries(
        objective,
        beliefs=beliefs,
        ranked_candidates=ranked,
        config=active_config,
    )
    frontier_summary = summarize_frontier(
        frontier,
        ranked_candidates=ranked[:8],
        queries=queries,
    )
    ranked_payload = tuple(_json_object(item) for item in ranked[:8])
    store.write_query_artifact(
        manifest.session_id,
        ranked_candidates=ranked_payload,
        queries=queries,
        frontier_summary=frontier_summary,
    )
    store.write_frontier_status(manifest.session_id, frontier_summary)
    return ranked[:8], queries, frontier_summary


def _proposal_artifact_refs(
    store: FilesystemSessionStore,
    session_id: str,
) -> dict[str, str]:
    artifact_paths = store.status(session_id).artifact_paths
    return {
        key: artifact_paths[key]
        for key in (
            "posterior_summary",
            "belief_delta_summary",
            "query",
            "frontier_status",
            "commit_decision",
            "results_markdown",
        )
    }


def handle_init(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    adapter = load_adapter(adapter_config)
    probe = adapter.probe()
    if not probe.available:
        raise SessionFailure(probe.detail)
    registry = adapter.build_registry()
    store = _store(args.session_root, adapter_config)
    beliefs: ValidatedBeliefBatch | None = None
    preview_payload: dict[str, JsonValue] | None = None
    preview_digest: str | None = None
    belief_payload = _load_init_belief_payload(args)
    canonicalization_summary_payload: dict[str, JsonValue] | None = None
    surface_overlay_payload: dict[str, JsonValue] | None = None
    canonicalization_model = load_canonicalization_model(
        cast(str | None, getattr(args, "canonicalization_model", None))
    )
    effective_canonicalization_mode = default_canonicalization_mode(
        requested_mode=_canonicalization_mode(args),
        model=canonicalization_model,
    )
    if belief_payload is not None:
        outcome = canonicalize_belief_input(
            belief_payload,
            fallback_session_context=_belief_fallback_context(args),
            registry=registry,
            mode=effective_canonicalization_mode,
            model=canonicalization_model,
        )
        beliefs = outcome.beliefs
        registry = outcome.registry
        if outcome.summary is not None:
            canonicalization_summary_payload = cast(
                dict[str, JsonValue], to_json_value(outcome.summary)
            )
        surface_overlay_payload = outcome.surface_overlay_payload
    session_id = args.session_id or (
        beliefs.session_context.session_id if beliefs else None
    )
    era_id = args.era_id or (beliefs.session_context.era_id if beliefs else None)
    if session_id is None or era_id is None:
        raise ValidationFailure(
            "session init requires a session id and era id, directly or via beliefs."
        )
    eval_contract = adapter.capture_eval_contract()
    if beliefs is not None:
        preview = preview_compiled_beliefs(beliefs, registry, EraState(era_id=era_id))
        preview_payload = cast(dict[str, JsonValue], to_json_value(preview))
        preview_digest = _payload_digest(preview_payload)
    manifest = SessionManifest(
        session_id=session_id,
        era_id=era_id,
        adapter_kind=adapter_config.kind,
        adapter_execution_mode=str(
            (probe.metadata or {}).get("execution_mode", adapter_config.mode)
        ),
        session_root=str(store.root),
        created_at=datetime.now(tz=UTC).isoformat(),
        preview_required=load_bayes_layer_config().interface_preview_required,
        beliefs_status="preview_pending" if beliefs is not None else "absent",
        preview_digest=preview_digest,
        compiled_priors_active=False,
        user_profile=beliefs.session_context.user_profile if beliefs else None,
        canonicalization_mode=(
            effective_canonicalization_mode
            if beliefs is not None
            and belief_payload is not None
            and "ideas" in belief_payload
            else None
        ),
        surface_overlay_active=surface_overlay_payload is not None,
        eval_contract_digest=eval_contract.contract_digest,
        eval_contract_required=True,
        workspace_snapshot_mode=eval_contract.workspace_snapshot_mode,
    )
    store.init_session(manifest)
    store.write_eval_contract(session_id, eval_contract)
    store.write_surface_snapshot(
        session_id,
        {
            "registry": cast(dict[str, JsonValue], to_json_value(registry.to_dict())),
            "surface_summary": cast(
                dict[str, JsonValue], to_json_value(registry.surface_summary())
            ),
            "eval_contract": cast(dict[str, JsonValue], to_json_value(eval_contract)),
        },
    )
    if beliefs is not None and preview_payload is not None:
        store.write_beliefs(session_id, beliefs)
        store.write_preview(session_id, preview_payload)
        if surface_overlay_payload is not None:
            store.write_surface_overlay(session_id, surface_overlay_payload)
        if canonicalization_summary_payload is not None:
            store.write_canonicalization_summary(
                session_id,
                canonicalization_summary_payload,
            )
    payload = _status_payload(
        store=store,
        manifest=manifest,
        adapter_config=adapter_config,
    )
    if canonicalization_summary_payload is not None:
        payload["canonicalization_summary"] = canonicalization_summary_payload
    if surface_overlay_payload is not None:
        payload["surface_overlay"] = surface_overlay_payload
    return payload


def handle_apply_beliefs(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    if manifest.beliefs_status == "absent":
        raise SessionFailure(
            f"Session {args.session_id!r} does not have stored beliefs."
        )
    if manifest.preview_digest is None:
        raise SessionFailure(
            f"Session {args.session_id!r} did not record a preview digest."
        )
    if args.preview_digest != manifest.preview_digest:
        raise SessionFailure(
            "Preview digest mismatch. Re-run `autoclanker session status` and apply the recorded digest."
        )
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    beliefs = store.load_beliefs(args.session_id)
    compiled = compile_beliefs(
        beliefs,
        registry,
        EraState(era_id=manifest.era_id),
    )
    store.write_compiled_priors(
        args.session_id,
        cast(dict[str, JsonValue], to_json_value(compiled)),
    )
    store.write_manifest(
        replace(
            manifest,
            beliefs_status="applied",
            compiled_priors_active=True,
        )
    )
    updated_manifest = store.load_manifest(args.session_id)
    return _status_payload(
        store=store,
        manifest=updated_manifest,
        adapter_config=adapter_config,
    )


def handle_ingest_eval(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    result = validate_eval_result(_load_payload(args.input))
    if result.era_id != manifest.era_id:
        raise SessionFailure(
            f"Eval result era {result.era_id!r} did not match session era {manifest.era_id!r}."
        )
    if manifest.eval_contract_required:
        expected_contract = store.load_eval_contract(args.session_id)
        if expected_contract is None:
            raise SessionFailure(
                f"Session {args.session_id!r} requires an eval contract, but none was stored."
            )
        if result.eval_contract is None:
            raise SessionFailure(
                "Eval result did not include eval_contract for this hardened session."
            )
        mismatches = compare_eval_contracts(expected_contract, result.eval_contract)
        if mismatches:
            raise SessionFailure(
                "Eval result contract did not match the locked session contract: "
                + ", ".join(mismatches)
            )
        current_contract = load_adapter(adapter_config).capture_eval_contract()
        current_mismatches = compare_eval_contracts(expected_contract, current_contract)
        if current_mismatches:
            raise SessionFailure(
                "Current evaluation surface drifted from the locked session contract: "
                + ", ".join(current_mismatches)
            )
    store.append_observation(args.session_id, result)
    return {
        "session_id": args.session_id,
        "observation_count": len(store.read_observations(args.session_id)),
        "patch_hash": result.patch_hash,
    }


def handle_fit(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    observations = store.read_observations(args.session_id)
    beliefs, compiled = _load_active_beliefs_and_bundle(
        store=store,
        manifest=manifest,
        registry=registry,
    )
    del beliefs
    era_state = EraState(
        era_id=manifest.era_id,
        observation_count=len(observations),
    )
    summary = _compute_summary(
        era_state=era_state,
        registry=registry,
        observations=observations,
        compiled=compiled,
    )
    store.write_posterior_summary(args.session_id, summary)
    store.write_belief_delta_summary(
        args.session_id,
        build_belief_delta_summary(
            compiled=compiled,
            summary=summary,
        ),
    )
    store.write_influence_summary(
        args.session_id,
        payload={
            "influence_summary": cast(
                list[JsonValue], to_json_value(summary.influence_summary)
            )
        },
    )
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        adapter_config=adapter_config,
        compiled=compiled,
        observations=observations,
        summary=summary,
    )
    payload = cast(dict[str, JsonValue], to_json_value(summary))
    payload["report_artifacts"] = cast(
        dict[str, JsonValue], to_json_value(report_artifacts)
    )
    return payload


def handle_suggest(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    observations = store.read_observations(args.session_id)
    beliefs, compiled = _load_active_beliefs_and_bundle(
        store=store,
        manifest=manifest,
        registry=registry,
    )
    frontier = _load_frontier(
        args.candidates_input,
        registry=registry,
        compiled=compiled,
    )
    ranked, queries, frontier_summary = _suggest_from_frontier(
        store=store,
        manifest=manifest,
        registry=registry,
        beliefs=beliefs,
        compiled=compiled,
        frontier=frontier,
        observations=observations,
    )
    summary = _compute_summary(
        era_state=EraState(
            era_id=manifest.era_id,
            observation_count=len(observations),
        ),
        registry=registry,
        observations=observations,
        compiled=compiled,
    )
    store.write_posterior_summary(args.session_id, summary)
    store.write_belief_delta_summary(
        args.session_id,
        build_belief_delta_summary(
            compiled=compiled,
            summary=summary,
            frontier_summary=frontier_summary,
            ranked_candidates=ranked,
        ),
    )
    store.write_proposal_ledger(
        args.session_id,
        build_proposal_ledger(
            session_id=args.session_id,
            era_id=manifest.era_id,
            ranked_candidates=ranked,
            frontier_summary=frontier_summary,
            decision=None,
            previous=store.load_proposal_ledger(args.session_id),
            artifact_refs=_proposal_artifact_refs(store, args.session_id),
        ),
    )
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        adapter_config=adapter_config,
        compiled=compiled,
        observations=observations,
        summary=summary,
        ranked_candidates=ranked[:8],
        queries=queries,
    )
    return {
        "session_id": args.session_id,
        "ranked_candidates": cast(list[JsonValue], to_json_value(ranked)),
        "queries": cast(list[JsonValue], to_json_value(queries)),
        "frontier_summary": cast(dict[str, JsonValue], to_json_value(frontier_summary)),
        "influence_summary": cast(
            list[JsonValue],
            to_json_value(
                tuple(
                    {
                        "candidate_id": item.candidate_id,
                        "influence_summary": item.influence_summary,
                    }
                    for item in ranked[:8]
                )
            ),
        ),
        "report_artifacts": cast(dict[str, JsonValue], to_json_value(report_artifacts)),
    }


def handle_run_eval(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    raw_payload = _load_payload(args.candidate_input)
    frontier = _frontier_from_payload(
        raw_payload if "candidates" in raw_payload else {"candidates": [raw_payload]}
    )
    if len(frontier.candidates) != 1:
        raise ValidationFailure("run-eval requires exactly one candidate.")
    candidate = frontier.candidates[0]
    result = _append_hardened_eval_result(
        store=store,
        manifest=manifest,
        adapter_config=adapter_config,
        candidate=candidate,
        seed=args.seed,
        replication_index=args.replication_index,
    )
    return {
        "session_id": args.session_id,
        "result": cast(dict[str, JsonValue], to_json_value(result)),
        "observation_count": len(store.read_observations(args.session_id)),
    }


def handle_run_frontier(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    beliefs, compiled = _load_active_beliefs_and_bundle(
        store=store,
        manifest=manifest,
        registry=registry,
    )
    frontier = _frontier_from_payload(_load_payload(args.frontier_input))
    results: list[ValidEvalResult] = []
    for index, candidate in enumerate(frontier.candidates):
        results.append(
            _append_hardened_eval_result(
                store=store,
                manifest=manifest,
                adapter_config=adapter_config,
                candidate=candidate,
                seed=args.seed_base + index,
                replication_index=0,
            )
        )
    observations = store.read_observations(args.session_id)
    ranked, queries, frontier_summary = _suggest_from_frontier(
        store=store,
        manifest=manifest,
        registry=registry,
        beliefs=beliefs,
        compiled=compiled,
        frontier=frontier,
        observations=observations,
    )
    return {
        "session_id": args.session_id,
        "results": cast(list[JsonValue], to_json_value(tuple(results))),
        "ranked_candidates": cast(list[JsonValue], to_json_value(ranked)),
        "queries": cast(list[JsonValue], to_json_value(queries)),
        "frontier_summary": cast(dict[str, JsonValue], to_json_value(frontier_summary)),
    }


def handle_frontier_status(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    frontier_summary = store.load_frontier_status(args.session_id)
    payload = _status_payload(
        store=store,
        manifest=manifest,
        adapter_config=adapter_config,
    )
    payload["frontier_summary"] = to_json_value(
        frontier_summary
        or FrontierSummary(
            frontier_id="frontier_default",
            candidate_count=0,
            family_count=0,
            family_representatives=(),
            dropped_family_reasons={},
            pending_queries=(),
            pending_merge_suggestions=(),
            budget_allocations={},
        )
    )
    return payload


def handle_recommend_commit(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    observations = store.read_observations(args.session_id)
    _beliefs, compiled = _load_active_beliefs_and_bundle(
        store=store,
        manifest=manifest,
        registry=registry,
    )
    era_state = EraState(
        era_id=manifest.era_id,
        observation_count=len(observations),
    )
    objective = fit_objective_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
    )
    feasibility = fit_feasibility_surrogate(
        observations,
        registry=registry,
        compiled_priors=compiled,
        era_state=era_state,
    )
    ranked = rank_candidates(
        generate_candidate_pool(registry, compiled_priors=compiled),
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        compiled_priors=compiled,
    )
    decision = recommend_commit(
        session_id=args.session_id,
        era_id=manifest.era_id,
        ranked_candidates=ranked,
        observations=observations,
        registry=registry,
    )
    store.write_commit_decision(args.session_id, decision)
    frontier_summary = store.load_frontier_status(args.session_id)
    store.write_influence_summary(
        args.session_id,
        payload={
            "influence_summary": cast(
                list[JsonValue], to_json_value(decision.influence_summary)
            )
        },
    )
    summary = _compute_summary(
        era_state=era_state,
        registry=registry,
        observations=observations,
        compiled=compiled,
    )
    store.write_posterior_summary(args.session_id, summary)
    store.write_belief_delta_summary(
        args.session_id,
        build_belief_delta_summary(
            compiled=compiled,
            summary=summary,
            frontier_summary=frontier_summary,
            ranked_candidates=ranked,
        ),
    )
    store.write_proposal_ledger(
        args.session_id,
        build_proposal_ledger(
            session_id=args.session_id,
            era_id=manifest.era_id,
            ranked_candidates=ranked,
            frontier_summary=frontier_summary,
            decision=decision,
            previous=store.load_proposal_ledger(args.session_id),
            artifact_refs=_proposal_artifact_refs(store, args.session_id),
        ),
    )
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        adapter_config=adapter_config,
        compiled=compiled,
        observations=observations,
        summary=summary,
        ranked_candidates=ranked[:8],
        decision=decision,
    )
    payload = cast(dict[str, JsonValue], to_json_value(decision))
    payload["report_artifacts"] = cast(
        dict[str, JsonValue], to_json_value(report_artifacts)
    )
    return payload


def handle_render_report(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    registry = _registry_from_config(adapter_config)
    overlay = store.load_surface_overlay(args.session_id)
    if overlay is not None:
        registry = registry.with_overlay(overlay)
    observations = store.read_observations(args.session_id)
    _beliefs, compiled = _load_active_beliefs_and_bundle(
        store=store,
        manifest=manifest,
        registry=registry,
    )
    summary = _compute_summary(
        era_state=EraState(
            era_id=manifest.era_id,
            observation_count=len(observations),
        ),
        registry=registry,
        observations=observations,
        compiled=compiled,
    )
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        adapter_config=adapter_config,
        compiled=compiled,
        observations=observations,
        summary=summary,
    )
    return {
        "session_id": args.session_id,
        "report_artifacts": cast(dict[str, JsonValue], to_json_value(report_artifacts)),
    }


def handle_review_bundle(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    return build_review_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        adapter_config=adapter_config,
    )


def handle_status(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    return _status_payload(
        store=store,
        manifest=manifest,
        adapter_config=adapter_config,
    )


def register_session_commands(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "session",
        help="Manage autoclanker filesystem sessions.",
    )
    session_subparsers = parser.add_subparsers(dest="session_command", required=True)

    init_parser = session_subparsers.add_parser(
        "init",
        help="Create a new session manifest.",
    )
    init_parser.add_argument("--session-id", help="Session identifier.")
    init_parser.add_argument("--era-id", help="Era identifier.")
    init_parser.add_argument(
        "--beliefs-input",
        help="Optional beliefs or beginner ideas file. Use '-' to read from stdin.",
    )
    init_parser.add_argument(
        "--ideas-json",
        help="Inline beginner ideas as JSON. Accepts one string idea, one idea object, a list of idea strings or objects, or an object with top-level 'ideas'. String ideas and omitted confidence values default to confidence 2.",
    )
    init_parser.add_argument(
        "--canonicalization-mode",
        choices=("deterministic", "hybrid", "llm"),
        help="Override the beginner canonicalization pipeline mode. deterministic uses only registry semantics, hybrid uses deterministic resolution first and then the model for unresolved ideas, and llm uses the model first with deterministic fallback when the model returns no typed belief. Defaults to hybrid only when a canonicalization model is configured, otherwise deterministic.",
    )
    init_parser.add_argument(
        "--canonicalization-model",
        help="Optional provider-agnostic canonicalization model identifier. Use 'stub' for the built-in test model, 'anthropic' for the bundled Anthropic provider, or an import path exposing build_autoclanker_canonicalization_model().",
    )
    init_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    init_parser.add_argument("--session-root", help="Override session root.")
    init_parser.set_defaults(handler=handle_init)

    apply_parser = session_subparsers.add_parser(
        "apply-beliefs",
        help="Activate stored beliefs after verifying the preview digest.",
    )
    apply_parser.add_argument("--session-id", required=True, help="Target session id.")
    apply_parser.add_argument(
        "--preview-digest",
        required=True,
        help="Preview digest returned by session init/status.",
    )
    apply_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    apply_parser.add_argument("--session-root", help="Override session root.")
    apply_parser.set_defaults(handler=handle_apply_beliefs)

    ingest_parser = session_subparsers.add_parser(
        "ingest-eval",
        help="Append one eval result.",
    )
    ingest_parser.add_argument("--session-id", required=True, help="Target session id.")
    ingest_parser.add_argument("--input", help="Path to a JSON eval result file.")
    ingest_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    ingest_parser.add_argument("--session-root", help="Override session root.")
    ingest_parser.set_defaults(handler=handle_ingest_eval)

    run_eval_parser = session_subparsers.add_parser(
        "run-eval",
        help="Execute one candidate under the locked eval contract and append the result.",
    )
    run_eval_parser.add_argument(
        "--session-id", required=True, help="Target session id."
    )
    run_eval_parser.add_argument(
        "--candidate-input",
        required=True,
        help="Path to one candidate object or one-candidate frontier payload.",
    )
    run_eval_parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed to pass through to the adapter.",
    )
    run_eval_parser.add_argument(
        "--replication-index",
        type=int,
        default=0,
        help="Replication index to pass through to the adapter.",
    )
    run_eval_parser.add_argument(
        "--adapter-config", help="Optional adapter config path."
    )
    run_eval_parser.add_argument("--session-root", help="Override session root.")
    run_eval_parser.set_defaults(handler=handle_run_eval)

    run_frontier_parser = session_subparsers.add_parser(
        "run-frontier",
        help="Execute a frontier batch under the locked eval contract.",
    )
    run_frontier_parser.add_argument(
        "--session-id", required=True, help="Target session id."
    )
    run_frontier_parser.add_argument(
        "--frontier-input",
        required=True,
        help="Path to a frontier payload.",
    )
    run_frontier_parser.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Base seed; candidates increment from this value.",
    )
    run_frontier_parser.add_argument(
        "--adapter-config", help="Optional adapter config path."
    )
    run_frontier_parser.add_argument("--session-root", help="Override session root.")
    run_frontier_parser.set_defaults(handler=handle_run_frontier)

    fit_parser = session_subparsers.add_parser(
        "fit", help="Refresh posterior summaries."
    )
    fit_parser.add_argument("--session-id", required=True, help="Target session id.")
    fit_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    fit_parser.add_argument("--session-root", help="Override session root.")
    fit_parser.set_defaults(handler=handle_fit)

    suggest_parser = session_subparsers.add_parser(
        "suggest",
        help="Rank candidates and emit bounded query suggestions.",
    )
    suggest_parser.add_argument(
        "--session-id", required=True, help="Target session id."
    )
    suggest_parser.add_argument(
        "--candidates-input",
        help="Optional candidate pool or frontier payload.",
    )
    suggest_parser.add_argument(
        "--adapter-config", help="Optional adapter config path."
    )
    suggest_parser.add_argument("--session-root", help="Override session root.")
    suggest_parser.set_defaults(handler=handle_suggest)

    commit_parser = session_subparsers.add_parser(
        "recommend-commit",
        help="Emit a commit or no-commit recommendation.",
    )
    commit_parser.add_argument("--session-id", required=True, help="Target session id.")
    commit_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    commit_parser.add_argument("--session-root", help="Override session root.")
    commit_parser.set_defaults(handler=handle_recommend_commit)

    status_parser = session_subparsers.add_parser(
        "status",
        help="Read session artifact status.",
    )
    status_parser.add_argument("--session-id", required=True, help="Target session id.")
    status_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    status_parser.add_argument("--session-root", help="Override session root.")
    status_parser.set_defaults(handler=handle_status)

    frontier_status_parser = session_subparsers.add_parser(
        "frontier-status",
        help="Read the persisted frontier summary and current trust state.",
    )
    frontier_status_parser.add_argument(
        "--session-id", required=True, help="Target session id."
    )
    frontier_status_parser.add_argument(
        "--adapter-config", help="Optional adapter config path."
    )
    frontier_status_parser.add_argument("--session-root", help="Override session root.")
    frontier_status_parser.set_defaults(handler=handle_frontier_status)

    review_bundle_parser = session_subparsers.add_parser(
        "review-bundle",
        help="Derive a normalized review bundle from the current session artifacts.",
    )
    review_bundle_parser.add_argument(
        "--session-id", required=True, help="Target session id."
    )
    review_bundle_parser.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="Output format for the derived bundle.",
    )
    review_bundle_parser.add_argument(
        "--adapter-config", help="Optional adapter config path."
    )
    review_bundle_parser.add_argument("--session-root", help="Override session root.")
    review_bundle_parser.set_defaults(handler=handle_review_bundle)

    render_parser = session_subparsers.add_parser(
        "render-report",
        help="Write a human-readable session summary and visual report artifacts.",
    )
    render_parser.add_argument("--session-id", required=True, help="Target session id.")
    render_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    render_parser.add_argument("--session-root", help="Override session root.")
    render_parser.set_defaults(handler=handle_render_report)
