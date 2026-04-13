from __future__ import annotations

import argparse
import hashlib
import json
import sys

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
from autoclanker.bayes_layer.posterior_graph import build_posterior_graph
from autoclanker.bayes_layer.query_policy import suggest_queries
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.reporting import render_session_report_bundle
from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.surrogate_feasibility import fit_feasibility_surrogate
from autoclanker.bayes_layer.surrogate_objective import fit_objective_surrogate
from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    GeneStateRef,
    InfluenceSummary,
    JsonValue,
    PosteriorSummary,
    SessionContext,
    SessionFailure,
    SessionManifest,
    ValidAdapterConfig,
    ValidatedBeliefBatch,
    ValidationFailure,
    ValidEvalResult,
    to_json_value,
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


def _load_candidates(
    candidates_input: str | None,
    *,
    registry: GeneRegistry,
    compiled: CompiledPriorBundle,
) -> tuple[tuple[str, tuple[GeneStateRef, ...]], ...]:
    if candidates_input is None:
        return generate_candidate_pool(registry, compiled_priors=compiled)
    return _candidate_input_to_refs(_load_payload(candidates_input))


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
    )
    graph = build_posterior_graph(objective, compiled)
    return PosteriorSummary(
        era_id=era_state.era_id,
        observation_count=objective.observation_count,
        aggregate_count=len({item.patch_hash for item in observations}),
        objective_baseline=objective.baseline_utility,
        valid_baseline=feasibility.baseline_valid_probability,
        top_features=objective.features[:8],
        top_candidates=ranked[:5],
        graph=graph,
        influence_summary=tuple(
            InfluenceSummary(
                source_belief_id=spec.source_belief_id,
                target_ref=spec.item.target_ref,
                summary=f"{spec.source_belief_id} contributes prior mass to {spec.item.target_ref}.",
            )
            for spec in compiled.all_items[:8]
        ),
    )


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
    )
    store.init_session(manifest)
    store.write_surface_snapshot(
        session_id,
        {
            "registry": cast(dict[str, JsonValue], to_json_value(registry.to_dict())),
            "surface_summary": cast(
                dict[str, JsonValue], to_json_value(registry.surface_summary())
            ),
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
    payload = cast(dict[str, JsonValue], to_json_value(store.status(session_id)))
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
        SessionManifest(
            session_id=manifest.session_id,
            era_id=manifest.era_id,
            adapter_kind=manifest.adapter_kind,
            adapter_execution_mode=manifest.adapter_execution_mode,
            session_root=manifest.session_root,
            created_at=manifest.created_at,
            preview_required=manifest.preview_required,
            beliefs_status="applied",
            preview_digest=manifest.preview_digest,
            compiled_priors_active=True,
            user_profile=manifest.user_profile,
            canonicalization_mode=manifest.canonicalization_mode,
            surface_overlay_active=manifest.surface_overlay_active,
        )
    )
    return cast(dict[str, JsonValue], to_json_value(store.status(args.session_id)))


def handle_ingest_eval(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    manifest = store.load_manifest(args.session_id)
    result = validate_eval_result(_load_payload(args.input))
    if result.era_id != manifest.era_id:
        raise SessionFailure(
            f"Eval result era {result.era_id!r} did not match session era {manifest.era_id!r}."
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
    candidates = _load_candidates(
        args.candidates_input,
        registry=registry,
        compiled=compiled,
    )
    ranked = rank_candidates(
        candidates,
        registry=registry,
        objective_posterior=objective,
        feasibility_posterior=feasibility,
        compiled_priors=compiled,
    )
    queries = suggest_queries(
        objective,
        beliefs=beliefs,
        ranked_candidates=ranked,
    )
    ranked_payload = tuple(_json_object(item) for item in ranked[:8])
    store.write_query_artifact(
        args.session_id,
        ranked_candidates=ranked_payload,
        queries=queries,
    )
    summary = _compute_summary(
        era_state=era_state,
        registry=registry,
        observations=observations,
        compiled=compiled,
    )
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
        compiled=compiled,
        observations=observations,
        summary=summary,
        ranked_candidates=ranked[:8],
        queries=queries,
    )
    return {
        "session_id": args.session_id,
        "ranked_candidates": cast(list[JsonValue], list(ranked_payload)),
        "queries": cast(list[JsonValue], to_json_value(queries)),
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
    report_artifacts = render_session_report_bundle(
        store=store,
        session_id=args.session_id,
        manifest=manifest,
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
        compiled=compiled,
        observations=observations,
        summary=summary,
    )
    return {
        "session_id": args.session_id,
        "report_artifacts": cast(dict[str, JsonValue], to_json_value(report_artifacts)),
    }


def handle_status(args: argparse.Namespace) -> dict[str, JsonValue]:
    adapter_config = _load_adapter_config(args.adapter_config)
    store = _store(args.session_root, adapter_config)
    status = store.status(args.session_id)
    return cast(dict[str, JsonValue], to_json_value(status))


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
        "--candidates-input", help="Optional candidate pool payload."
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

    render_parser = session_subparsers.add_parser(
        "render-report",
        help="Write a human-readable session summary and visual report artifacts.",
    )
    render_parser.add_argument("--session-id", required=True, help="Target session id.")
    render_parser.add_argument("--adapter-config", help="Optional adapter config path.")
    render_parser.add_argument("--session-root", help="Override session root.")
    render_parser.set_defaults(handler=handle_render_report)
