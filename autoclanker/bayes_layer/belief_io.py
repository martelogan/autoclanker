from __future__ import annotations

import importlib.util
import json
import re
import shutil

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import jsonschema
import yaml

from autoclanker.bayes_layer.config import repo_relative_path
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AdapterKind,
    AdapterMode,
    Belief,
    BeliefContext,
    CandidatePattern,
    ConstraintBelief,
    ConstraintType,
    DecayOverride,
    EvalStatus,
    ExpertPriorBelief,
    FailureMode,
    FeasibilityTarget,
    GeneStateRef,
    GraphDirectiveBelief,
    GraphDirectiveType,
    IdeaBelief,
    JsonValue,
    MainEffectTarget,
    PairEffectTarget,
    PreferenceBelief,
    PreferenceDirection,
    ProposalBelief,
    RelationBelief,
    RelationType,
    RiskVector,
    SessionContext,
    UserProfile,
    ValidAdapterConfig,
    ValidatedBeliefBatch,
    ValidationFailure,
    ValidEvalResult,
    VramTarget,
    to_json_value,
)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label} must be a mapping.")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _require_mapping(value, label)


def _require_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"Expected {key!r} to be a non-empty string.")
    return value.strip()


def _optional_string(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationFailure(f"Expected {key!r} to be a string.")
    normalized = value.strip()
    return normalized or None


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValidationFailure(f"Expected {key!r} to be an integer.")
    return value


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValidationFailure(f"Expected {key!r} to be numeric.")
    return float(value)


def _optional_string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    items = cast(list[str], raw_items)
    return tuple(item.strip() for item in items if item.strip())


def _optional_path_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    items = cast(list[str], raw_items)
    return tuple(item for item in items)


def _normalize_gene_state_ref(mapping: Mapping[str, object]) -> GeneStateRef:
    return GeneStateRef(
        gene_id=_require_string(mapping, "gene_id"),
        state_id=_require_string(mapping, "state_id"),
    )


def _parse_gene_state_ref_text(value: str, *, label: str) -> GeneStateRef:
    raw = value.strip()
    if "=" not in raw:
        raise ValidationFailure(
            f"{label} must use the form 'gene_id=state_id', got {value!r}."
        )
    gene_id, state_id = raw.split("=", 1)
    if not gene_id.strip() or not state_id.strip():
        raise ValidationFailure(
            f"{label} must use the form 'gene_id=state_id', got {value!r}."
        )
    return GeneStateRef(
        gene_id=gene_id.strip(),
        state_id=state_id.strip(),
    )


def _optional_risk_name_list(
    mapping: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    allowed = {
        "compile_fail",
        "runtime_fail",
        "oom",
        "timeout",
        "metric_instability",
    }
    items = tuple(item.strip() for item in cast(list[str], raw_items) if item.strip())
    invalid = sorted(item for item in items if item not in allowed)
    if invalid:
        raise ValidationFailure(
            f"Unsupported risk names in {key!r}: {', '.join(invalid)}."
        )
    return items


def parse_gene_state_refs(
    items: object,
    label: str,
    *,
    preserve_order: bool = False,
) -> tuple[GeneStateRef, ...]:
    if not isinstance(items, list):
        raise ValidationFailure(f"{label} must be a list.")
    raw_items = cast(list[object], items)
    refs: list[GeneStateRef] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_items):
        ref = _normalize_gene_state_ref(_require_mapping(item, f"{label}[{index}]"))
        if ref.canonical_key in seen:
            raise ValidationFailure(
                f"{label} contains duplicate member {ref.canonical_key!r}."
            )
        seen.add(ref.canonical_key)
        refs.append(ref)
    if preserve_order:
        return tuple(refs)
    return tuple(sorted(refs, key=lambda ref: (ref.gene_id, ref.state_id)))


def _parse_risk_vector(value: object, label: str) -> RiskVector | None:
    mapping = _optional_mapping(value, label)
    if mapping is None:
        return None
    return RiskVector(
        compile_fail=cast(int | None, mapping.get("compile_fail")),
        runtime_fail=cast(int | None, mapping.get("runtime_fail")),
        oom=cast(int | None, mapping.get("oom")),
        timeout=cast(int | None, mapping.get("timeout")),
        metric_instability=cast(int | None, mapping.get("metric_instability")),
    )


def _parse_context(value: object, label: str) -> BeliefContext | None:
    mapping = _optional_mapping(value, label)
    if mapping is None:
        return None
    tags = ()
    raw_tags = mapping.get("tags")
    if raw_tags is not None:
        if not isinstance(raw_tags, list):
            raise ValidationFailure(f"{label}.tags must be a list of strings.")
        raw_tag_items = cast(list[object], raw_tags)
        if any(not isinstance(item, str) for item in raw_tag_items):
            raise ValidationFailure(f"{label}.tags must be a list of strings.")
        tag_items = cast(list[str], raw_tag_items)
        tags = tuple(item.strip() for item in tag_items if item.strip())
    metadata: dict[str, JsonValue] = {}
    for key, item in mapping.items():
        if key in {"hardware_profile_id", "budget_profile_id", "tags"}:
            continue
        metadata[key] = to_json_value(item)
    return BeliefContext(
        hardware_profile_id=_optional_string(mapping, "hardware_profile_id"),
        budget_profile_id=_optional_string(mapping, "budget_profile_id"),
        tags=tags,
        metadata=metadata or None,
    )


def load_serialized_payload_from_text(
    text: str,
    *,
    source_name: str = "<stdin>",
) -> dict[str, object]:
    stripped = text.lstrip()
    if not stripped:
        raise ValidationFailure(f"{source_name} was empty.")
    try:
        if stripped.startswith("{") or stripped.startswith("["):
            payload = json.loads(text)
        else:
            payload = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValidationFailure(f"Failed to parse {source_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(f"{source_name} must contain a top-level mapping.")
    return cast(dict[str, object], payload)


def load_serialized_payload(path: Path) -> dict[str, object]:
    return load_serialized_payload_from_text(
        path.read_text(encoding="utf-8"),
        source_name=str(path),
    )


def load_inline_ideas_payload(
    text: str,
    *,
    source_name: str = "--ideas-json",
) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        raise ValidationFailure(f"{source_name} was empty.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationFailure(f"Failed to parse {source_name}: {exc}") from exc
    if isinstance(payload, str):
        idea_text = payload.strip()
        if not idea_text:
            raise ValidationFailure(f"{source_name} string idea was empty.")
        return {
            "ideas": [
                {
                    "idea": idea_text,
                    "confidence": 2,
                }
            ]
        }
    if isinstance(payload, list):
        return _normalize_beginner_idea_payload(
            {"ideas": cast(list[object], payload)},
            source_name=source_name,
        )
    if not isinstance(payload, dict):
        raise ValidationFailure(
            f"{source_name} must be a JSON string idea, one idea object, a JSON array of idea strings or objects, or an object with top-level 'ideas'."
        )
    mapping = cast(dict[str, object], payload)
    if "ideas" in mapping or "beliefs" in mapping or "session_context" in mapping:
        if "ideas" in mapping:
            return _normalize_beginner_idea_payload(mapping, source_name=source_name)
        return mapping
    if "idea" in mapping:
        return _normalize_beginner_idea_payload(
            {"ideas": [mapping]},
            source_name=source_name,
        )
    raise ValidationFailure(
        f"{source_name} must be one idea object, a list of idea strings or objects, or an object with top-level 'ideas'."
    )


def _normalize_beginner_idea_item(item: object, *, label: str) -> dict[str, object]:
    if isinstance(item, str):
        idea_text = item.strip()
        if not idea_text:
            raise ValidationFailure(f"{label} must not be empty.")
        return {
            "idea": idea_text,
            "confidence": 2,
        }
    mapping = dict(_require_mapping(item, label))
    if "idea" in mapping and "confidence" not in mapping:
        mapping["confidence"] = 2
    return mapping


def _normalize_beginner_idea_payload(
    payload: Mapping[str, object],
    *,
    source_name: str,
) -> dict[str, object]:
    normalized = dict(payload)
    raw_ideas = payload.get("ideas")
    if raw_ideas is None:
        return normalized
    if not isinstance(raw_ideas, list):
        raise ValidationFailure(f"{source_name}.ideas must be a list.")
    raw_idea_items = cast(list[object], raw_ideas)
    normalized["ideas"] = [
        _normalize_beginner_idea_item(
            item,
            label=f"{source_name}.ideas[{index}]",
        )
        for index, item in enumerate(raw_idea_items)
    ]
    return normalized


def load_schema_document(schema_filename: str) -> dict[str, object]:
    schema_path = repo_relative_path("schemas", schema_filename)
    return load_serialized_payload(schema_path)


def validate_payload_against_schema(
    payload: Mapping[str, object],
    schema_filename: str,
) -> None:
    schema = load_schema_document(schema_filename)
    validator = jsonschema.Draft202012Validator(schema)
    payload_for_validation = cast(Any, dict(payload))
    validator_any = cast(Any, validator)
    errors: list[Any]
    errors = sorted(
        validator_any.iter_errors(payload_for_validation),
        key=lambda error: list(error.path),
    )
    if errors:
        messages: list[str] = []
        for error in errors:
            json_path = "$"
            for part in error.path:
                json_path = f"{json_path}.{part}"
            messages.append(f"{json_path}: {error.message}")
        raise ValidationFailure("; ".join(messages))


def _parse_session_context(payload: Mapping[str, object]) -> SessionContext:
    return SessionContext(
        session_id=_optional_string(payload, "session_id"),
        era_id=_require_string(payload, "era_id"),
        hardware_profile_id=_optional_string(payload, "hardware_profile_id"),
        budget_profile_id=_optional_string(payload, "budget_profile_id"),
        author=_optional_string(payload, "author"),
        user_profile=cast(
            UserProfile | None, _optional_string(payload, "user_profile")
        ),
    )


def _merge_session_context(
    payload: Mapping[str, object] | None,
    *,
    fallback: SessionContext | None = None,
    default_user_profile: UserProfile | None = None,
) -> SessionContext:
    mapping = cast(Mapping[str, object], {} if payload is None else payload)
    era_id = _optional_string(mapping, "era_id")
    if era_id is None and fallback is not None:
        era_id = fallback.era_id
    if era_id is None:
        raise ValidationFailure(
            "session_context.era_id is required, directly or via --era-id."
        )
    session_id = _optional_string(mapping, "session_id")
    if session_id is None and fallback is not None:
        session_id = fallback.session_id
    hardware_profile_id = _optional_string(mapping, "hardware_profile_id")
    if hardware_profile_id is None and fallback is not None:
        hardware_profile_id = fallback.hardware_profile_id
    budget_profile_id = _optional_string(mapping, "budget_profile_id")
    if budget_profile_id is None and fallback is not None:
        budget_profile_id = fallback.budget_profile_id
    author = _optional_string(mapping, "author")
    if author is None and fallback is not None:
        author = fallback.author
    user_profile = cast(UserProfile | None, _optional_string(mapping, "user_profile"))
    if user_profile is None and fallback is not None:
        user_profile = fallback.user_profile
    if user_profile is None:
        user_profile = default_user_profile
    return SessionContext(
        era_id=era_id,
        session_id=session_id,
        hardware_profile_id=hardware_profile_id,
        budget_profile_id=budget_profile_id,
        author=author,
        user_profile=user_profile,
    )


def _idea_effect_strength(
    *,
    confidence_level: int,
    effect: str | None,
) -> int:
    magnitude = min(3, max(1, confidence_level))
    return -magnitude if effect == "hurt" else magnitude


def _idea_relation_strength(confidence_level: int) -> int:
    return min(3, max(1, confidence_level))


def _idea_joint_effect_strength(
    *,
    confidence_level: int,
    relation: str,
) -> int:
    magnitude = min(3, max(1, confidence_level))
    if relation in {"conflict", "exclusion"}:
        return -magnitude
    return magnitude


def _idea_risk_vector(
    *,
    risk_names: tuple[str, ...],
    confidence_level: int,
) -> RiskVector | None:
    if not risk_names:
        return None
    level = min(3, max(1, confidence_level))
    return RiskVector(**{name: level for name in risk_names})


_AUTO_CANONICALIZE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "here",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "probably",
        "should",
        "that",
        "the",
        "this",
        "to",
        "usually",
        "when",
        "with",
    }
)
_AUTO_RELATION_CUES = (
    "work best together",
    "works best together",
    "works best with",
    "best with",
    "pair with",
    "paired with",
    "together",
    "alongside",
    "plus",
    "combo",
    "combination",
)


def _normalize_freeform_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _freeform_tokens(value: str) -> tuple[str, ...]:
    normalized = _normalize_freeform_text(value)
    if not normalized:
        return ()
    return tuple(
        token
        for token in normalized.split()
        if len(token) > 1 and token not in _AUTO_CANONICALIZE_STOPWORDS
    )


def _idea_text_has_relation_signal(value: str, mapping: Mapping[str, object]) -> bool:
    if "relation" in mapping:
        return True
    normalized = _normalize_freeform_text(value)
    return any(cue in normalized for cue in _AUTO_RELATION_CUES)


def _state_match_score(
    *,
    idea_text: str,
    idea_tokens: set[str],
    ref: GeneStateRef,
    registry: GeneRegistry,
) -> tuple[float, tuple[str, ...]]:
    definition = registry.genes[ref.gene_id]
    matched_phrases: list[str] = []
    score = 0.0

    gene_phrases = (
        ref.gene_id,
        definition.description,
        *definition.aliases,
    )
    state_phrases = (
        ref.state_id,
        registry.state_description(ref),
        *registry.state_aliases(ref),
    )

    for phrase in gene_phrases:
        normalized_phrase = _normalize_freeform_text(phrase)
        if not normalized_phrase:
            continue
        phrase_tokens = set(_freeform_tokens(phrase))
        if normalized_phrase in idea_text:
            score += 1.5
            matched_phrases.append(phrase)
        elif phrase_tokens:
            overlap = idea_tokens & phrase_tokens
            if overlap:
                score += 0.45 * float(len(overlap))

    for phrase in state_phrases:
        normalized_phrase = _normalize_freeform_text(phrase)
        if not normalized_phrase:
            continue
        phrase_tokens = set(_freeform_tokens(phrase))
        if normalized_phrase in idea_text:
            score += 3.0
            matched_phrases.append(phrase)
        elif phrase_tokens and phrase_tokens.issubset(idea_tokens):
            score += 2.2
            matched_phrases.append(phrase)
        elif phrase_tokens:
            overlap = idea_tokens & phrase_tokens
            if overlap:
                score += 0.9 * float(len(overlap))

    if ref.state_id == definition.default_state:
        baseline_words = {"default", "baseline", "keep", "unchanged", "leave"}
        if not (baseline_words & idea_tokens):
            score -= 0.8
    return score, tuple(dict.fromkeys(matched_phrases))


def _top_state_matches(
    *,
    idea_text: str,
    registry: GeneRegistry,
) -> tuple[tuple[GeneStateRef, float, tuple[str, ...]], ...]:
    tokens = set(_freeform_tokens(idea_text))
    if not tokens:
        return ()
    matches: list[tuple[GeneStateRef, float, tuple[str, ...]]] = []
    normalized_text = _normalize_freeform_text(idea_text)
    for gene_id in registry.known_gene_ids():
        definition = registry.genes[gene_id]
        for state_id in definition.states:
            ref = GeneStateRef(gene_id=gene_id, state_id=state_id)
            score, phrases = _state_match_score(
                idea_text=normalized_text,
                idea_tokens=tokens,
                ref=ref,
                registry=registry,
            )
            matches.append((ref, score, phrases))
    return tuple(
        sorted(
            matches,
            key=lambda item: (
                -item[1],
                item[0].gene_id,
                item[0].state_id,
            ),
        )
    )


def _suggested_option_strings(
    matches: Sequence[tuple[GeneStateRef, float, tuple[str, ...]]],
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    suggestions: list[str] = []
    for ref, score, _phrases in matches:
        if score <= 0.0:
            continue
        suggestions.append(f"{ref.gene_id}={ref.state_id}")
        if len(suggestions) >= limit:
            break
    return tuple(suggestions)


def _auto_context_metadata(
    *,
    mode: str,
    original_idea: str,
    match_score: float | None = None,
    matched_phrases: Sequence[str] = (),
    suggested_options: Sequence[str] = (),
) -> BeliefContext:
    metadata: dict[str, JsonValue] = {
        "canonicalization_mode": mode,
        "original_idea": original_idea,
    }
    if match_score is not None:
        metadata["canonicalization_score"] = round(match_score, 3)
    if matched_phrases:
        metadata["matched_phrases"] = list(matched_phrases)
    if suggested_options:
        metadata["suggested_options"] = list(suggested_options)
    return BeliefContext(metadata=metadata)


def _parse_candidate_pattern(value: object, label: str) -> CandidatePattern:
    mapping = _require_mapping(value, label)
    return CandidatePattern(
        members=parse_gene_state_refs(mapping.get("members"), f"{label}.members")
    )


def _parse_expert_target(
    value: object,
) -> MainEffectTarget | PairEffectTarget | FeasibilityTarget | VramTarget:
    mapping = _require_mapping(value, "target")
    target_kind = _require_string(mapping, "target_kind")
    if target_kind == "main_effect":
        return MainEffectTarget(
            target_kind="main_effect",
            gene=_normalize_gene_state_ref(
                _require_mapping(mapping.get("gene"), "target.gene")
            ),
        )
    if target_kind == "pair_effect":
        members = parse_gene_state_refs(mapping.get("members"), "target.members")
        if len(members) != 2:
            raise ValidationFailure(
                "Expert pair_effect targets must contain exactly two members."
            )
        return PairEffectTarget(
            target_kind="pair_effect",
            members=(members[0], members[1]),
        )
    if target_kind == "feasibility_logit":
        failure_mode = _require_string(mapping, "failure_mode")
        return FeasibilityTarget(
            target_kind="feasibility_logit",
            gene=_normalize_gene_state_ref(
                _require_mapping(mapping.get("gene"), "target.gene")
            ),
            failure_mode=cast(FailureMode, failure_mode),
        )
    if target_kind == "vram_effect":
        return VramTarget(
            target_kind="vram_effect",
            gene=_normalize_gene_state_ref(
                _require_mapping(mapping.get("gene"), "target.gene")
            ),
        )
    raise ValidationFailure(f"Unsupported expert target kind {target_kind!r}.")


def _parse_belief(raw_belief: Mapping[str, object]) -> Belief:
    kind = _require_string(raw_belief, "kind")
    belief_id = _require_string(raw_belief, "id")
    confidence_level = _require_int(raw_belief, "confidence_level")
    evidence_sources = _optional_string_list(raw_belief, "evidence_sources")
    context = _parse_context(raw_belief.get("context"), "context")
    rationale = _optional_string(raw_belief, "rationale") or ""
    if kind == "proposal":
        return ProposalBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            proposal_text=_require_string(raw_belief, "proposal_text"),
            suggested_scope=_optional_string(raw_belief, "suggested_scope"),
            risk_hints=_parse_risk_vector(raw_belief.get("risk_hints"), "risk_hints"),
        )
    if kind == "idea":
        return IdeaBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            gene=_normalize_gene_state_ref(
                _require_mapping(raw_belief.get("gene"), "gene")
            ),
            effect_strength=_require_int(raw_belief, "effect_strength"),
            risk=_parse_risk_vector(raw_belief.get("risk"), "risk"),
            complexity_delta=cast(int | None, raw_belief.get("complexity_delta")),
        )
    if kind == "relation":
        return RelationBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            members=parse_gene_state_refs(raw_belief.get("members"), "members"),
            relation=cast(RelationType, _require_string(raw_belief, "relation")),
            strength=_require_int(raw_belief, "strength"),
            joint_effect_strength=cast(
                int | None, raw_belief.get("joint_effect_strength")
            ),
        )
    if kind == "preference":
        left_pattern = _parse_candidate_pattern(
            raw_belief.get("left_pattern"), "left_pattern"
        )
        right_pattern = _parse_candidate_pattern(
            raw_belief.get("right_pattern"), "right_pattern"
        )
        if left_pattern.members == right_pattern.members:
            raise ValidationFailure(
                "Preference beliefs must compare two distinct candidate patterns."
            )
        return PreferenceBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            left_pattern=left_pattern,
            right_pattern=right_pattern,
            preference=cast(
                PreferenceDirection,
                _require_string(raw_belief, "preference"),
            ),
            strength=_require_int(raw_belief, "strength"),
        )
    if kind == "constraint":
        return ConstraintBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            constraint_type=cast(
                ConstraintType, _require_string(raw_belief, "constraint_type")
            ),
            severity=_require_int(raw_belief, "severity"),
            scope=parse_gene_state_refs(raw_belief.get("scope"), "scope"),
        )
    if kind == "expert_prior":
        return ExpertPriorBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            target=_parse_expert_target(raw_belief.get("target")),
            prior_family=cast(
                Literal["normal", "logit_normal"],
                _require_string(raw_belief, "prior_family"),
            ),
            mean=_require_float(raw_belief, "mean"),
            scale=_require_float(raw_belief, "scale"),
            observation_weight=cast(float | None, raw_belief.get("observation_weight")),
            decay_override=(
                None
                if raw_belief.get("decay_override") is None
                else DecayOverride(
                    per_eval_multiplier=cast(
                        float | None,
                        _require_mapping(
                            raw_belief.get("decay_override"), "decay_override"
                        ).get("per_eval_multiplier"),
                    ),
                    cross_era_transfer=cast(
                        float | None,
                        _require_mapping(
                            raw_belief.get("decay_override"), "decay_override"
                        ).get("cross_era_transfer"),
                    ),
                )
            ),
        )
    if kind == "graph_directive":
        members = parse_gene_state_refs(raw_belief.get("members"), "members")
        if len(members) != 2:
            raise ValidationFailure(
                "graph_directive beliefs require exactly two members."
            )
        return GraphDirectiveBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            context=context,
            rationale=rationale,
            members=(members[0], members[1]),
            directive=cast(
                GraphDirectiveType, _require_string(raw_belief, "directive")
            ),
            strength=_require_int(raw_belief, "strength"),
        )
    raise ValidationFailure(f"Unsupported belief kind {kind!r}.")


def ingest_human_beliefs(payload: Mapping[str, object]) -> ValidatedBeliefBatch:
    validate_payload_against_schema(payload, "human_belief.schema.json")
    session_context = _parse_session_context(
        _require_mapping(payload.get("session_context"), "session_context"),
    )
    raw_beliefs = payload.get("beliefs")
    if not isinstance(raw_beliefs, list):
        raise ValidationFailure("beliefs must be a list.")
    raw_belief_items = cast(list[object], raw_beliefs)
    beliefs = tuple(
        _parse_belief(_require_mapping(item, f"beliefs[{index}]"))
        for index, item in enumerate(raw_belief_items)
    )
    belief_ids = [belief.id for belief in beliefs]
    if len(belief_ids) != len(set(belief_ids)):
        raise ValidationFailure("Belief ids must be unique within a batch.")
    canonical_payload = cast(
        dict[str, JsonValue],
        to_json_value(
            {
                "session_context": session_context,
                "beliefs": beliefs,
            }
        ),
    )
    return ValidatedBeliefBatch(
        session_context=session_context,
        beliefs=beliefs,
        canonical_payload=canonical_payload,
    )


def _auto_canonicalize_beginner_idea(
    *,
    belief_id: str,
    rationale: str,
    confidence_level: int,
    evidence_sources: tuple[str, ...],
    relation: str,
    effect: str | None,
    risk_names: tuple[str, ...],
    scope: str | None,
    registry: GeneRegistry | None,
    mapping: Mapping[str, object],
) -> Belief:
    if registry is None:
        return ProposalBelief(
            id=belief_id,
            confidence_level=confidence_level,
            evidence_sources=evidence_sources,
            rationale=rationale,
            proposal_text=rationale,
            suggested_scope=scope or "patch",
            risk_hints=_idea_risk_vector(
                risk_names=risk_names,
                confidence_level=confidence_level,
            ),
        )

    matches = _top_state_matches(idea_text=rationale, registry=registry)
    suggestions = _suggested_option_strings(matches)
    if _idea_text_has_relation_signal(rationale, mapping):
        relation_members: list[tuple[GeneStateRef, float, tuple[str, ...]]] = []
        seen_genes: set[str] = set()
        for ref, score, phrases in matches:
            if score < 2.2 or ref.gene_id in seen_genes:
                continue
            relation_members.append((ref, score, phrases))
            seen_genes.add(ref.gene_id)
            if len(relation_members) == 2:
                break
        if len(relation_members) == 2:
            best_score = relation_members[0][1] + relation_members[1][1]
            matched_phrases = relation_members[0][2] + relation_members[1][2]
            return RelationBelief(
                id=belief_id,
                confidence_level=confidence_level,
                evidence_sources=evidence_sources,
                context=_auto_context_metadata(
                    mode="heuristic_relation",
                    original_idea=rationale,
                    match_score=best_score,
                    matched_phrases=matched_phrases,
                ),
                rationale=rationale,
                members=tuple(
                    sorted(
                        (relation_members[0][0], relation_members[1][0]),
                        key=lambda ref: (ref.gene_id, ref.state_id),
                    )
                ),
                relation=cast(RelationType, relation),
                strength=_idea_relation_strength(confidence_level),
                joint_effect_strength=_idea_joint_effect_strength(
                    confidence_level=confidence_level,
                    relation=relation,
                ),
            )

    if matches:
        best_ref, best_score, best_phrases = matches[0]
        second_score = matches[1][1] if len(matches) > 1 else -1.0
        if best_score >= 2.6 and (best_score - second_score) >= 0.6:
            return IdeaBelief(
                id=belief_id,
                confidence_level=confidence_level,
                evidence_sources=evidence_sources,
                context=_auto_context_metadata(
                    mode="heuristic_single",
                    original_idea=rationale,
                    match_score=best_score,
                    matched_phrases=best_phrases,
                ),
                rationale=rationale,
                gene=best_ref,
                effect_strength=_idea_effect_strength(
                    confidence_level=confidence_level,
                    effect=effect,
                ),
                risk=_idea_risk_vector(
                    risk_names=risk_names,
                    confidence_level=confidence_level,
                ),
            )

    return ProposalBelief(
        id=belief_id,
        confidence_level=confidence_level,
        evidence_sources=evidence_sources,
        context=_auto_context_metadata(
            mode="needs_review",
            original_idea=rationale,
            suggested_options=suggestions,
        ),
        rationale=rationale,
        proposal_text=rationale,
        suggested_scope=scope or "patch",
        risk_hints=_idea_risk_vector(
            risk_names=risk_names,
            confidence_level=confidence_level,
        ),
    )


def _ingest_idea_beliefs(
    payload: Mapping[str, object],
    *,
    fallback_session_context: SessionContext | None = None,
    registry: GeneRegistry | None = None,
) -> ValidatedBeliefBatch:
    normalized_payload = _normalize_beginner_idea_payload(
        payload,
        source_name="idea payload",
    )
    validate_payload_against_schema(normalized_payload, "idea_belief.schema.json")
    raw_session_context = normalized_payload.get("session_context")
    session_context = _merge_session_context(
        (
            None
            if raw_session_context is None
            else _require_mapping(raw_session_context, "session_context")
        ),
        fallback=fallback_session_context,
        default_user_profile="basic",
    )

    raw_ideas = normalized_payload.get("ideas")
    if not isinstance(raw_ideas, list):
        raise ValidationFailure("ideas must be a list.")
    raw_idea_items = cast(list[object], raw_ideas)
    beliefs: list[Belief] = []
    for index, item in enumerate(raw_idea_items):
        mapping = _require_mapping(item, f"ideas[{index}]")
        belief_id = _optional_string(mapping, "id") or f"idea_{index + 1:03d}"
        rationale = _require_string(mapping, "idea")
        confidence_level = _require_int(mapping, "confidence")
        evidence_sources = _optional_string_list(mapping, "evidence_sources") or (
            "intuition",
        )
        target = _optional_string(mapping, "target")
        option = _optional_string(mapping, "option")
        raw_members = mapping.get("members")
        raw_options = mapping.get("options")
        relation = _optional_string(mapping, "relation") or "synergy"
        effect = _optional_string(mapping, "effect")
        risk_names = _optional_risk_name_list(mapping, "risks")
        scope = _optional_string(mapping, "scope")

        if target is not None and option is not None:
            raise ValidationFailure(
                f"Idea {belief_id!r} cannot define both target and option."
            )
        if raw_members is not None and raw_options is not None:
            raise ValidationFailure(
                f"Idea {belief_id!r} cannot define both members and options."
            )
        resolved_target = option or target
        raw_group = raw_options if raw_options is not None else raw_members

        if resolved_target is not None and raw_group is not None:
            raise ValidationFailure(
                f"Idea {belief_id!r} cannot define both option/target and options/members."
            )

        if raw_group is not None:
            if not isinstance(raw_group, list):
                raise ValidationFailure(
                    f"Idea {belief_id!r} options must be a list of strings."
                )
            raw_member_items = cast(list[object], raw_group)
            if any(not isinstance(member, str) for member in raw_member_items):
                raise ValidationFailure(
                    f"Idea {belief_id!r} options must be a list of strings."
                )
            members = tuple(
                _parse_gene_state_ref_text(
                    cast(str, member),
                    label=f"{belief_id}.options[{member_index}]",
                )
                for member_index, member in enumerate(raw_member_items)
            )
            beliefs.append(
                RelationBelief(
                    id=belief_id,
                    confidence_level=confidence_level,
                    evidence_sources=evidence_sources,
                    context=_auto_context_metadata(
                        mode="explicit_relation",
                        original_idea=rationale,
                    ),
                    rationale=rationale,
                    members=tuple(
                        sorted(members, key=lambda ref: (ref.gene_id, ref.state_id))
                    ),
                    relation=cast(RelationType, relation),
                    strength=_idea_relation_strength(confidence_level),
                    joint_effect_strength=_idea_joint_effect_strength(
                        confidence_level=confidence_level,
                        relation=relation,
                    ),
                )
            )
            continue

        if resolved_target is not None:
            beliefs.append(
                IdeaBelief(
                    id=belief_id,
                    confidence_level=confidence_level,
                    evidence_sources=evidence_sources,
                    context=_auto_context_metadata(
                        mode="explicit_option",
                        original_idea=rationale,
                    ),
                    rationale=rationale,
                    gene=_parse_gene_state_ref_text(
                        resolved_target,
                        label=f"{belief_id}.option",
                    ),
                    effect_strength=_idea_effect_strength(
                        confidence_level=confidence_level,
                        effect=effect,
                    ),
                    risk=_idea_risk_vector(
                        risk_names=risk_names,
                        confidence_level=confidence_level,
                    ),
                )
            )
            continue

        beliefs.append(
            _auto_canonicalize_beginner_idea(
                belief_id=belief_id,
                rationale=rationale,
                confidence_level=confidence_level,
                evidence_sources=evidence_sources,
                relation=relation,
                effect=effect,
                risk_names=risk_names,
                scope=scope,
                registry=registry,
                mapping=mapping,
            )
        )

    belief_ids = [belief.id for belief in beliefs]
    if len(belief_ids) != len(set(belief_ids)):
        raise ValidationFailure("Belief ids must be unique within a batch.")
    canonical_payload = cast(
        dict[str, JsonValue],
        to_json_value(
            {
                "session_context": session_context,
                "beliefs": tuple(beliefs),
            }
        ),
    )
    return ValidatedBeliefBatch(
        session_context=session_context,
        beliefs=tuple(beliefs),
        canonical_payload=canonical_payload,
    )


def ingest_belief_input(
    payload: Mapping[str, object],
    *,
    fallback_session_context: SessionContext | None = None,
    registry: GeneRegistry | None = None,
) -> ValidatedBeliefBatch:
    if "beliefs" in payload:
        return ingest_human_beliefs(payload)
    if "ideas" in payload:
        return _ingest_idea_beliefs(
            payload,
            fallback_session_context=fallback_session_context,
            registry=registry,
        )
    raise ValidationFailure("Expected top-level 'beliefs' or 'ideas'.")


require_mapping = _require_mapping
require_string = _require_string
optional_string = _optional_string
require_int = _require_int
optional_string_list = _optional_string_list
parse_gene_state_ref_text = _parse_gene_state_ref_text
optional_risk_name_list = _optional_risk_name_list
normalize_beginner_idea_payload = _normalize_beginner_idea_payload
merge_session_context = _merge_session_context
idea_relation_strength = _idea_relation_strength
idea_joint_effect_strength = _idea_joint_effect_strength


def validate_eval_result(payload: Mapping[str, object]) -> ValidEvalResult:
    validate_payload_against_schema(payload, "eval_result.schema.json")
    intended = parse_gene_state_refs(
        payload.get("intended_genotype"),
        "intended_genotype",
        preserve_order=True,
    )
    realized = parse_gene_state_refs(
        payload.get("realized_genotype"),
        "realized_genotype",
        preserve_order=True,
    )
    raw_metrics_mapping = _require_mapping(payload.get("raw_metrics"), "raw_metrics")
    raw_metrics = {
        key: to_json_value(value) for key, value in raw_metrics_mapping.items()
    }
    failure_metadata_raw = _optional_mapping(
        payload.get("failure_metadata"), "failure_metadata"
    )
    failure_metadata = (
        None
        if failure_metadata_raw is None
        else {key: to_json_value(value) for key, value in failure_metadata_raw.items()}
    )
    return ValidEvalResult(
        era_id=_require_string(payload, "era_id"),
        candidate_id=_require_string(payload, "candidate_id"),
        intended_genotype=intended,
        realized_genotype=realized,
        patch_hash=_require_string(payload, "patch_hash"),
        status=cast(EvalStatus, _require_string(payload, "status")),
        seed=_require_int(payload, "seed"),
        runtime_sec=_require_float(payload, "runtime_sec"),
        peak_vram_mb=_require_float(payload, "peak_vram_mb"),
        raw_metrics=raw_metrics,
        delta_perf=_require_float(payload, "delta_perf"),
        utility=_require_float(payload, "utility"),
        replication_index=_require_int(payload, "replication_index"),
        stdout_digest=_optional_string(payload, "stdout_digest"),
        stderr_digest=_optional_string(payload, "stderr_digest"),
        artifact_paths=_optional_path_list(payload, "artifact_paths"),
        failure_metadata=failure_metadata,
    )


def resolve_relative_path(path_value: str, *, base_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute() or base_dir is None:
        return path
    return (base_dir / path).resolve()


def validate_adapter_config(
    payload: Mapping[str, object],
    *,
    base_dir: Path | None = None,
) -> ValidAdapterConfig:
    validate_payload_against_schema(payload, "adapter_config.schema.json")
    raw_adapter = _require_mapping(payload.get("adapter"), "adapter")
    kind = cast(AdapterKind, _require_string(raw_adapter, "kind"))
    mode = cast(AdapterMode, _require_string(raw_adapter, "mode"))
    allow_missing = bool(raw_adapter.get("allow_missing", False))
    repo_path = _optional_string(raw_adapter, "repo_path")
    python_module = _optional_string(raw_adapter, "python_module")
    command = _optional_path_list(raw_adapter, "command")
    config_path = _optional_string(raw_adapter, "config_path")
    metadata_raw = _optional_mapping(raw_adapter.get("metadata"), "metadata")
    metadata = (
        None
        if metadata_raw is None
        else {key: to_json_value(value) for key, value in metadata_raw.items()}
    )

    if mode == "fixture" and kind != "fixture":
        raise ValidationFailure("Only the fixture adapter may use mode='fixture'.")
    if kind == "fixture" and mode != "fixture":
        raise ValidationFailure("The fixture adapter only supports mode='fixture'.")
    if mode == "auto" and kind not in {"autoresearch", "cevolve"}:
        raise ValidationFailure(
            "auto mode is only supported by the first-party autoresearch and cevolve adapters."
        )
    if mode == "local_repo_path" and repo_path is None:
        raise ValidationFailure("local_repo_path mode requires repo_path.")
    if mode == "installed_module" and python_module is None:
        raise ValidationFailure("installed_module mode requires python_module.")
    if mode == "subprocess_cli" and not command:
        raise ValidationFailure("subprocess_cli mode requires command.")
    if (
        mode == "auto"
        and kind in {"autoresearch", "cevolve"}
        and repo_path is None
        and python_module is None
        and not command
        and not allow_missing
    ):
        raise ValidationFailure(
            "auto mode for first-party adapters requires at least one of repo_path, python_module, or command unless allow_missing=true."
        )

    resolved_path = (
        None
        if repo_path is None
        else resolve_relative_path(repo_path, base_dir=base_dir)
    )
    repo_path_available = resolved_path is not None and resolved_path.exists()
    python_module_available = (
        python_module is not None
        and importlib.util.find_spec(python_module) is not None
    )
    command_available = bool(command) and shutil.which(command[0]) is not None

    if mode == "local_repo_path" and not repo_path_available and not allow_missing:
        assert resolved_path is not None
        raise ValidationFailure(f"Configured repo_path does not exist: {resolved_path}")
    if mode == "installed_module" and not python_module_available and not allow_missing:
        assert python_module is not None
        raise ValidationFailure(
            f"Configured python_module is not importable: {python_module}"
        )
    if mode == "subprocess_cli" and not command_available and not allow_missing:
        assert command
        raise ValidationFailure(f"Configured command is not executable: {command[0]}")
    if (
        mode == "auto"
        and not allow_missing
        and not any((repo_path_available, python_module_available, command_available))
    ):
        attempted_hints: list[str] = []
        if resolved_path is not None:
            attempted_hints.append(f"repo_path={resolved_path}")
        if python_module is not None:
            attempted_hints.append(f"python_module={python_module}")
        if command:
            attempted_hints.append(f"command={command[0]}")
        hint_suffix = (
            f" Tried: {', '.join(attempted_hints)}." if attempted_hints else ""
        )
        raise ValidationFailure(
            "auto mode could not resolve any usable first-party adapter hint."
            + hint_suffix
        )

    return ValidAdapterConfig(
        kind=kind,
        mode=mode,
        session_root=_optional_string(raw_adapter, "session_root") or ".autoclanker",
        allow_missing=allow_missing,
        repo_path=repo_path,
        python_module=python_module,
        command=command,
        config_path=config_path,
        metadata=metadata,
        base_dir=None if base_dir is None else str(base_dir.resolve()),
    )
