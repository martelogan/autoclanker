from __future__ import annotations

import importlib
import os

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, cast

from autoclanker.bayes_layer.belief_io import (
    idea_joint_effect_strength,
    idea_relation_strength,
    ingest_belief_input,
    ingest_human_beliefs,
    merge_session_context,
    normalize_beginner_idea_payload,
    optional_risk_name_list,
    optional_string,
    optional_string_list,
    require_int,
    require_mapping,
    require_string,
    validate_payload_against_schema,
)
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    Belief,
    BeliefContext,
    CanonicalizationMode,
    CanonicalizationSource,
    GeneStateRef,
    IdeaBelief,
    JsonValue,
    ProposalBelief,
    RelationBelief,
    RelationType,
    SemanticLevel,
    SessionContext,
    SurfaceKind,
    ValidatedBeliefBatch,
    ValidationFailure,
    to_json_value,
)


@dataclass(frozen=True, slots=True)
class SurfaceOverlayGene:
    gene_id: str
    states: tuple[str, ...]
    default_state: str
    description: str
    aliases: tuple[str, ...] = ()
    state_descriptions: dict[str, str] | None = None
    state_aliases: dict[str, tuple[str, ...]] | None = None
    surface_kind: SurfaceKind = "mutation_family"
    semantic_level: SemanticLevel = "strategy"
    materializable: bool = False
    code_scopes: tuple[str, ...] = ()
    risk_hints: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class CanonicalizationIdea:
    belief_id: str
    rationale: str
    confidence_level: int
    relation: str
    effect: str | None
    risk_names: tuple[str, ...]
    scope: str | None
    evidence_sources: tuple[str, ...]
    raw_mapping: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class CanonicalizationSuggestion:
    belief_id: str
    belief: Belief | None
    source: CanonicalizationSource
    summary: str
    matched_evidence: tuple[str, ...] = ()
    overlay_genes: tuple[SurfaceOverlayGene, ...] = ()
    confidence_score: float | None = None
    needs_review: bool = False


@dataclass(frozen=True, slots=True)
class CanonicalizationRecord:
    belief_id: str
    source: CanonicalizationSource
    status: str
    belief_kind: str
    summary: str
    target_refs: tuple[str, ...] = ()
    overlay_gene_ids: tuple[str, ...] = ()
    matched_evidence: tuple[str, ...] = ()
    confidence_score: float | None = None


@dataclass(frozen=True, slots=True)
class CanonicalizationSummary:
    mode: CanonicalizationMode
    model_name: str | None
    records: tuple[CanonicalizationRecord, ...]


@dataclass(frozen=True, slots=True)
class CanonicalizationRequest:
    session_context: SessionContext
    registry: GeneRegistry
    ideas: tuple[CanonicalizationIdea, ...]


class CanonicalizationModel(Protocol):
    name: str

    def canonicalize(
        self,
        request: CanonicalizationRequest,
    ) -> tuple[CanonicalizationSuggestion, ...]: ...


@dataclass(frozen=True, slots=True)
class CanonicalizationOutcome:
    beliefs: ValidatedBeliefBatch
    registry: GeneRegistry
    summary: CanonicalizationSummary | None
    surface_overlay_payload: dict[str, JsonValue] | None = None


def _overlay_registry(
    overlay_genes: Sequence[SurfaceOverlayGene],
) -> GeneRegistry | None:
    if not overlay_genes:
        return None
    mapping = {gene.gene_id: gene.states for gene in overlay_genes}
    return GeneRegistry.from_mapping(
        mapping,
        defaults={gene.gene_id: gene.default_state for gene in overlay_genes},
        descriptions={gene.gene_id: gene.description for gene in overlay_genes},
        aliases={gene.gene_id: gene.aliases for gene in overlay_genes},
        state_descriptions={
            gene.gene_id: dict(gene.state_descriptions or {}) for gene in overlay_genes
        },
        state_aliases={
            gene.gene_id: dict(gene.state_aliases or {}) for gene in overlay_genes
        },
        surface_kinds={gene.gene_id: gene.surface_kind for gene in overlay_genes},
        semantic_levels={gene.gene_id: gene.semantic_level for gene in overlay_genes},
        materializable={gene.gene_id: gene.materializable for gene in overlay_genes},
        code_scopes={gene.gene_id: gene.code_scopes for gene in overlay_genes},
        risk_hints={gene.gene_id: gene.risk_hints for gene in overlay_genes},
        origins={gene.gene_id: "overlay" for gene in overlay_genes},
        metadata={
            gene.gene_id: dict(gene.metadata or {})
            for gene in overlay_genes
            if gene.metadata is not None
        }
        or None,
    )


def _overlay_payload(
    overlay_genes: Sequence[SurfaceOverlayGene],
) -> dict[str, JsonValue] | None:
    if not overlay_genes:
        return None
    overlay = _overlay_registry(overlay_genes)
    if overlay is None:
        return None
    return {
        "registry": cast(dict[str, JsonValue], to_json_value(overlay.to_dict())),
        "surface_summary": cast(
            dict[str, JsonValue],
            to_json_value(overlay.surface_summary()),
        ),
    }


def _target_refs_for_belief(belief: Belief) -> tuple[str, ...]:
    if isinstance(belief, IdeaBelief):
        return (f"{belief.gene.gene_id}={belief.gene.state_id}",)
    if isinstance(belief, RelationBelief):
        return tuple(f"{member.gene_id}={member.state_id}" for member in belief.members)
    return ()


def _record_for_belief(
    belief: Belief,
    *,
    source: CanonicalizationSource,
    status: str,
    summary: str,
    matched_evidence: Sequence[str] = (),
    overlay_gene_ids: Sequence[str] = (),
    confidence_score: float | None = None,
) -> CanonicalizationRecord:
    return CanonicalizationRecord(
        belief_id=belief.id,
        source=source,
        status=status,
        belief_kind=belief.kind,
        summary=summary,
        target_refs=_target_refs_for_belief(belief),
        overlay_gene_ids=tuple(overlay_gene_ids),
        matched_evidence=tuple(matched_evidence),
        confidence_score=confidence_score,
    )


def _with_context_metadata(belief: Belief, metadata: Mapping[str, JsonValue]) -> Belief:
    merged = (
        dict(belief.context.metadata or {})
        if belief.context is not None and belief.context.metadata is not None
        else {}
    )
    merged.update(metadata)
    context = belief.context or BeliefContext()
    return replace(belief, context=replace(context, metadata=merged))


def _validate_known_belief(belief: Belief, *, registry: GeneRegistry) -> Belief:
    if isinstance(belief, IdeaBelief):
        registry.canonicalize_ref(belief.gene)
        return belief
    if isinstance(belief, RelationBelief):
        registry.canonicalize_members(belief.members)
        return belief
    return belief


def _deterministic_summary(
    belief: Belief,
    *,
    unresolved_hint: bool = False,
) -> CanonicalizationRecord:
    metadata = (
        belief.context.metadata if belief.context and belief.context.metadata else {}
    )
    source: CanonicalizationSource = "deterministic"
    match_score = metadata.get("canonicalization_score")
    confidence_score = (
        float(match_score) if isinstance(match_score, (int, float)) else None
    )
    matched_phrases_raw = metadata.get("matched_phrases")
    matched_phrases = (
        tuple(str(item) for item in matched_phrases_raw)
        if isinstance(matched_phrases_raw, list)
        else ()
    )
    if isinstance(belief, ProposalBelief):
        summary = "High-level idea remains a proposal until it is canonicalized."
        if unresolved_hint:
            summary = "Deterministic canonicalization could not resolve this idea confidently."
        return _record_for_belief(
            belief,
            source=source,
            status="needs_review",
            summary=summary,
            matched_evidence=matched_phrases,
            confidence_score=confidence_score,
        )
    if isinstance(belief, RelationBelief):
        return _record_for_belief(
            belief,
            source=source,
            status="resolved",
            summary="Resolved to a typed relation belief using registry semantics.",
            matched_evidence=matched_phrases,
            confidence_score=confidence_score,
        )
    if isinstance(belief, IdeaBelief):
        return _record_for_belief(
            belief,
            source=source,
            status="resolved",
            summary="Resolved to a typed idea belief using registry semantics.",
            matched_evidence=matched_phrases,
            confidence_score=confidence_score,
        )
    return _record_for_belief(
        belief,
        source=source,
        status="resolved",
        summary="Resolved deterministically.",
        matched_evidence=matched_phrases,
        confidence_score=confidence_score,
    )


def _idea_request_items(
    payload: Mapping[str, object],
    *,
    fallback_session_context: SessionContext | None,
) -> tuple[SessionContext, tuple[CanonicalizationIdea, ...]]:
    normalized_payload = normalize_beginner_idea_payload(
        payload,
        source_name="idea payload",
    )
    validate_payload_against_schema(normalized_payload, "idea_belief.schema.json")
    raw_session_context = normalized_payload.get("session_context")
    session_context = merge_session_context(
        (
            None
            if raw_session_context is None
            else require_mapping(raw_session_context, "session_context")
        ),
        fallback=fallback_session_context,
        default_user_profile="basic",
    )
    raw_ideas = normalized_payload.get("ideas")
    if not isinstance(raw_ideas, list):
        raise ValidationFailure("ideas must be a list.")
    items: list[CanonicalizationIdea] = []
    for index, raw_item in enumerate(cast(list[object], raw_ideas)):
        mapping = require_mapping(raw_item, f"ideas[{index}]")
        idea_id = optional_string(mapping, "id") or f"idea_{index + 1:03d}"
        items.append(
            CanonicalizationIdea(
                belief_id=idea_id,
                rationale=require_string(mapping, "idea"),
                confidence_level=require_int(mapping, "confidence"),
                relation=optional_string(mapping, "relation") or "synergy",
                effect=optional_string(mapping, "effect"),
                risk_names=optional_risk_name_list(mapping, "risks"),
                scope=optional_string(mapping, "scope"),
                evidence_sources=optional_string_list(mapping, "evidence_sources")
                or ("intuition",),
                raw_mapping=cast(dict[str, JsonValue], to_json_value(dict(mapping))),
            )
        )
    return session_context, tuple(items)


def _deterministic_ingest(
    payload: Mapping[str, object],
    *,
    fallback_session_context: SessionContext | None,
    registry: GeneRegistry | None,
) -> ValidatedBeliefBatch:
    return ingest_belief_input(
        payload,
        fallback_session_context=fallback_session_context,
        registry=registry,
    )


class StubCanonicalizationModel:
    name = "stub"

    def canonicalize(
        self,
        request: CanonicalizationRequest,
    ) -> tuple[CanonicalizationSuggestion, ...]:
        suggestions: list[CanonicalizationSuggestion] = []
        for idea in request.ideas:
            lowered = idea.rationale.lower()
            if (
                "cache" in lowered
                or "fast path" in lowered
                or "repeated format" in lowered
            ):
                overlay_gene = SurfaceOverlayGene(
                    gene_id="search.repeated_format_fast_path",
                    states=("path_default", "path_compiled_context"),
                    default_state="path_default",
                    description="Higher-level search angle for repeated log formats that combines compiled matching with paired context reconstruction.",
                    aliases=(
                        "fast path",
                        "repeated format fast path",
                        "compiled context path",
                    ),
                    state_descriptions={
                        "path_default": "Do not bias toward a specialized repeated-format path.",
                        "path_compiled_context": "Bias toward the combined compiled-matcher plus context-pair path.",
                    },
                    state_aliases={
                        "path_default": ("default path",),
                        "path_compiled_context": (
                            "compiled context path",
                            "repeated format fast path",
                        ),
                    },
                    surface_kind="search_angle",
                    semantic_level="strategy",
                    materializable=False,
                    code_scopes=("parser.matcher", "parser.plan"),
                    risk_hints=("metric_instability",),
                    metadata={
                        "model_name": self.name,
                        "state_rules": {
                            "path_compiled_context": {
                                "implied_by_all": [
                                    "main:parser.matcher=matcher_compiled",
                                    "main:parser.plan=plan_context_pair",
                                ]
                            }
                        },
                    },
                )
                belief = _with_context_metadata(
                    IdeaBelief(
                        id=idea.belief_id,
                        confidence_level=idea.confidence_level,
                        evidence_sources=idea.evidence_sources,
                        rationale=idea.rationale,
                        gene=GeneStateRef(
                            gene_id="search.repeated_format_fast_path",
                            state_id="path_compiled_context",
                        ),
                        effect_strength=2 if idea.effect != "hurt" else -2,
                    ),
                    {
                        "canonicalization_mode": "llm_overlay",
                        "canonicalization_source": "llm",
                        "original_idea": idea.rationale,
                    },
                )
                suggestions.append(
                    CanonicalizationSuggestion(
                        belief_id=idea.belief_id,
                        belief=belief,
                        source="llm",
                        summary="Stub model proposed a session-local repeated-format fast-path search feature.",
                        matched_evidence=("fast path", "repeated format"),
                        overlay_genes=(overlay_gene,),
                        confidence_score=0.84,
                    )
                )
                continue
            if "together" in lowered and "summary" in lowered:
                belief = _with_context_metadata(
                    RelationBelief(
                        id=idea.belief_id,
                        confidence_level=idea.confidence_level,
                        evidence_sources=idea.evidence_sources,
                        rationale=idea.rationale,
                        members=(
                            GeneStateRef(
                                gene_id="parser.matcher",
                                state_id="matcher_compiled",
                            ),
                            GeneStateRef(
                                gene_id="emit.summary",
                                state_id="summary_streaming",
                            ),
                        ),
                        relation=cast(RelationType, "synergy"),
                        strength=idea_relation_strength(idea.confidence_level),
                        joint_effect_strength=idea_joint_effect_strength(
                            confidence_level=idea.confidence_level,
                            relation="synergy",
                        ),
                    ),
                    {
                        "canonicalization_mode": "llm_relation",
                        "canonicalization_source": "llm",
                        "original_idea": idea.rationale,
                    },
                )
                suggestions.append(
                    CanonicalizationSuggestion(
                        belief_id=idea.belief_id,
                        belief=belief,
                        source="llm",
                        summary="Stub model proposed a typed relation between existing surface entries.",
                        matched_evidence=("together", "summary"),
                        confidence_score=0.71,
                    )
                )
                continue
            suggestions.append(
                CanonicalizationSuggestion(
                    belief_id=idea.belief_id,
                    belief=None,
                    source="llm",
                    summary="Stub model could not confidently propose a typed belief.",
                    matched_evidence=(),
                    confidence_score=0.22,
                    needs_review=True,
                )
            )
        return tuple(suggestions)


def load_canonicalization_model(model_name: str | None) -> CanonicalizationModel | None:
    configured = model_name or os.environ.get("AUTOCLANKER_CANONICALIZATION_MODEL")
    if configured is None or not configured.strip():
        return None
    normalized = configured.strip()
    if normalized == "stub":
        return StubCanonicalizationModel()
    if normalized == "anthropic":
        normalized = "autoclanker.bayes_layer.providers.anthropic_canonicalizer"
    if normalized in {"openai", "openai-compatible", "openai_compatible"}:
        normalized = (
            "autoclanker.bayes_layer.providers.openai_compatible_canonicalizer"
        )
    module = importlib.import_module(normalized)
    builder = getattr(module, "build_autoclanker_canonicalization_model", None)
    if not callable(builder):
        raise ValidationFailure(
            f"Canonicalization model module {normalized!r} must export build_autoclanker_canonicalization_model()."
        )
    built = builder()
    if not hasattr(built, "canonicalize") or not hasattr(built, "name"):
        raise ValidationFailure(
            f"Canonicalization model module {normalized!r} returned an invalid model object."
        )
    return cast(CanonicalizationModel, built)


def default_canonicalization_mode(
    *,
    requested_mode: CanonicalizationMode | None,
    model: CanonicalizationModel | None,
) -> CanonicalizationMode:
    if requested_mode is not None:
        return requested_mode
    return "hybrid" if model is not None else "deterministic"


def canonicalize_belief_input(
    payload: Mapping[str, object],
    *,
    fallback_session_context: SessionContext | None = None,
    registry: GeneRegistry | None = None,
    mode: CanonicalizationMode = "deterministic",
    model: CanonicalizationModel | None = None,
) -> CanonicalizationOutcome:
    if "beliefs" in payload:
        beliefs = ingest_human_beliefs(payload)
        return CanonicalizationOutcome(
            beliefs=beliefs,
            registry=registry or GeneRegistry(genes={}),
            summary=None,
            surface_overlay_payload=None,
        )
    if "ideas" not in payload:
        raise ValidationFailure("Expected top-level 'beliefs' or 'ideas'.")
    if mode == "llm" and model is None:
        raise ValidationFailure(
            "canonicalization mode 'llm' requires --canonicalization-model or AUTOCLANKER_CANONICALIZATION_MODEL."
        )

    base_registry = registry
    deterministic = _deterministic_ingest(
        payload,
        fallback_session_context=fallback_session_context,
        registry=base_registry,
    )
    if base_registry is None or mode == "deterministic" or model is None:
        deterministic_records: tuple[CanonicalizationRecord, ...] = tuple(
            _deterministic_summary(belief) for belief in deterministic.beliefs
        )
        return CanonicalizationOutcome(
            beliefs=deterministic,
            registry=base_registry or GeneRegistry(genes={}),
            summary=CanonicalizationSummary(
                mode="deterministic",
                model_name=None,
                records=deterministic_records,
            ),
            surface_overlay_payload=None,
        )

    session_context, idea_items = _idea_request_items(
        payload,
        fallback_session_context=fallback_session_context,
    )
    unresolved_ids = {
        belief.id
        for belief in deterministic.beliefs
        if isinstance(belief, ProposalBelief)
    }
    pending_ideas = (
        idea_items
        if mode == "llm"
        else tuple(item for item in idea_items if item.belief_id in unresolved_ids)
    )
    if not pending_ideas:
        deterministic_records: tuple[CanonicalizationRecord, ...] = tuple(
            _deterministic_summary(belief) for belief in deterministic.beliefs
        )
        return CanonicalizationOutcome(
            beliefs=deterministic,
            registry=base_registry,
            summary=CanonicalizationSummary(
                mode=mode,
                model_name=model.name,
                records=deterministic_records,
            ),
            surface_overlay_payload=None,
        )

    suggestions = model.canonicalize(
        CanonicalizationRequest(
            session_context=session_context,
            registry=base_registry,
            ideas=pending_ideas,
        )
    )
    suggestion_by_id = {item.belief_id: item for item in suggestions}
    overlay_genes: list[SurfaceOverlayGene] = []
    for suggestion in suggestions:
        overlay_genes.extend(suggestion.overlay_genes)
    overlay_registry = _overlay_registry(overlay_genes)
    augmented_registry = (
        base_registry.with_overlay(overlay_registry)
        if overlay_registry is not None
        else base_registry
    )
    replaced_beliefs: list[Belief] = []
    records: list[CanonicalizationRecord] = []
    for belief in deterministic.beliefs:
        suggestion = suggestion_by_id.get(belief.id)
        if suggestion is None or suggestion.belief is None:
            replaced_beliefs.append(belief)
            if suggestion is not None:
                if isinstance(belief, ProposalBelief):
                    records.append(
                        _record_for_belief(
                            belief,
                            source="llm" if mode == "llm" else "hybrid",
                            status="needs_review",
                            summary=suggestion.summary,
                            matched_evidence=suggestion.matched_evidence,
                            confidence_score=suggestion.confidence_score,
                        )
                    )
                else:
                    records.append(
                        _record_for_belief(
                            belief,
                            source="deterministic",
                            status="resolved",
                            summary="Model returned no typed replacement; kept deterministic resolution.",
                            matched_evidence=suggestion.matched_evidence,
                            confidence_score=suggestion.confidence_score,
                        )
                    )
            else:
                records.append(
                    _deterministic_summary(
                        belief,
                        unresolved_hint=isinstance(belief, ProposalBelief),
                    )
                )
            continue
        validated = _validate_known_belief(
            suggestion.belief, registry=augmented_registry
        )
        hybrid_metadata: dict[str, JsonValue] = {
            "canonicalization_mode": mode,
            "canonicalization_source": suggestion.source,
            "original_idea": getattr(validated, "rationale", ""),
        }
        if suggestion.matched_evidence:
            hybrid_metadata["matched_evidence"] = list(suggestion.matched_evidence)
        if suggestion.confidence_score is not None:
            hybrid_metadata["confidence_score"] = suggestion.confidence_score
        validated = _with_context_metadata(validated, hybrid_metadata)
        replaced_beliefs.append(validated)
        records.append(
            _record_for_belief(
                validated,
                source="llm" if mode == "llm" else "hybrid",
                status="resolved",
                summary=suggestion.summary,
                matched_evidence=suggestion.matched_evidence,
                overlay_gene_ids=[gene.gene_id for gene in suggestion.overlay_genes],
                confidence_score=suggestion.confidence_score,
            )
        )

    canonical_payload = cast(
        dict[str, JsonValue],
        to_json_value(
            {
                "session_context": deterministic.session_context,
                "beliefs": tuple(replaced_beliefs),
            }
        ),
    )
    return CanonicalizationOutcome(
        beliefs=ValidatedBeliefBatch(
            session_context=deterministic.session_context,
            beliefs=tuple(replaced_beliefs),
            canonical_payload=canonical_payload,
        ),
        registry=augmented_registry,
        summary=CanonicalizationSummary(
            mode=mode,
            model_name=model.name,
            records=tuple(records),
        ),
        surface_overlay_payload=_overlay_payload(overlay_genes),
    )
