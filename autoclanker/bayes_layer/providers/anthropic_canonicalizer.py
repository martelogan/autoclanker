from __future__ import annotations

import json
import os
import time

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from autoclanker.bayes_layer.belief_io import ingest_human_beliefs
from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationIdea,
    CanonicalizationModel,
    CanonicalizationRequest,
    CanonicalizationSuggestion,
    SurfaceOverlayGene,
)
from autoclanker.bayes_layer.types import (
    Belief,
    JsonValue,
    SemanticLevel,
    SessionContext,
    SurfaceKind,
    ValidationFailure,
    to_json_value,
)

_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_TIMEOUT_SEC = 60
_DEFAULT_MAX_TOKENS = 1400
_DEFAULT_TEMPERATURE = 0.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF_SEC = 1.0


def _extract_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return cast(dict[str, object], payload)
    start = stripped.find("{")
    if start == -1:
        raise ValidationFailure("Anthropic canonicalizer did not return a JSON object.")
    depth = 0
    for index in range(start, len(stripped)):
        char = stripped[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValidationFailure(
                        "Anthropic canonicalizer returned malformed JSON."
                    ) from exc
                if not isinstance(parsed, dict):
                    raise ValidationFailure(
                        "Anthropic canonicalizer must return a JSON object."
                    )
                return cast(dict[str, object], parsed)
    raise ValidationFailure(
        "Anthropic canonicalizer returned an incomplete JSON object."
    )


def _string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    items: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
        normalized = item.strip()
        if normalized:
            items.append(normalized)
    return tuple(items)


def _float_or_none(mapping: Mapping[str, object], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValidationFailure(f"Expected {key!r} to be numeric when provided.")
    return float(value)


def _float_from_any(mapping: Mapping[str, object], *keys: str) -> float | None:
    for key in keys:
        value = _float_or_none(mapping, key)
        if value is not None:
            return value
    return None


def _bool_or_default(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValidationFailure(f"Expected {key!r} to be a boolean.")
    return value


def _require_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"Expected {key!r} to be a non-empty string.")
    return value.strip()


def _mapping_to_overlay_gene(mapping: Mapping[str, object]) -> SurfaceOverlayGene:
    raw_states = mapping.get("states")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValidationFailure("overlay gene states must be a non-empty list.")
    states: list[str] = []
    for item in cast(list[object], raw_states):
        if not isinstance(item, str) or not item.strip():
            raise ValidationFailure("overlay gene states must contain strings.")
        states.append(item.strip())
    default_state = _require_string(mapping, "default_state")
    if default_state not in states:
        raise ValidationFailure("overlay gene default_state must be one of its states.")

    raw_state_descriptions = mapping.get("state_descriptions")
    state_descriptions: dict[str, str] | None = None
    if isinstance(raw_state_descriptions, Mapping):
        parsed_descriptions: dict[str, str] = {}
        for state_id, text in cast(
            Mapping[object, object], raw_state_descriptions
        ).items():
            if isinstance(text, str) and text.strip():
                parsed_descriptions[str(state_id)] = text.strip()
        state_descriptions = parsed_descriptions or None

    raw_state_aliases = mapping.get("state_aliases")
    state_aliases: dict[str, tuple[str, ...]] | None = None
    if isinstance(raw_state_aliases, Mapping):
        parsed_aliases: dict[str, tuple[str, ...]] = {}
        for state_id, items in cast(Mapping[object, object], raw_state_aliases).items():
            if not isinstance(items, list):
                continue
            aliases: list[str] = []
            for item in cast(list[object], items):
                if isinstance(item, str) and item.strip():
                    aliases.append(item.strip())
            if aliases:
                parsed_aliases[str(state_id)] = tuple(aliases)
        state_aliases = parsed_aliases or None

    raw_metadata = mapping.get("metadata")
    metadata: dict[str, JsonValue] | None = None
    if isinstance(raw_metadata, Mapping):
        normalized_metadata: dict[str, JsonValue] = {}
        for key, value in cast(Mapping[object, object], raw_metadata).items():
            normalized_metadata[str(key)] = to_json_value(value)
        metadata = normalized_metadata or None

    return SurfaceOverlayGene(
        gene_id=_require_string(mapping, "gene_id"),
        states=tuple(states),
        default_state=default_state,
        description=_require_string(mapping, "description"),
        aliases=_string_list(mapping, "aliases"),
        state_descriptions=state_descriptions,
        state_aliases=state_aliases,
        surface_kind=cast(
            SurfaceKind,
            mapping.get("surface_kind", "mutation_family"),
        ),
        semantic_level=cast(
            SemanticLevel,
            mapping.get("semantic_level", "strategy"),
        ),
        materializable=_bool_or_default(mapping, "materializable", False),
        code_scopes=_string_list(mapping, "code_scopes"),
        risk_hints=_string_list(mapping, "risk_hints"),
        metadata=metadata,
    )


def _belief_from_mapping(
    mapping: Mapping[str, object],
    *,
    session_context: SessionContext,
) -> Belief:
    belief_mapping = dict(mapping)
    payload = {
        "session_context": cast(dict[str, JsonValue], to_json_value(session_context)),
        "beliefs": [cast(dict[str, JsonValue], to_json_value(belief_mapping))],
    }
    batch = ingest_human_beliefs(payload)
    if len(batch.beliefs) != 1:
        raise ValidationFailure(
            "Anthropic canonicalizer must return exactly one typed belief per suggestion."
        )
    return batch.beliefs[0]


def _bounded_positive_strength(value: int | float, *, maximum: int = 3) -> int:
    normalized = int(round(float(value)))
    return max(1, min(normalized, maximum))


def _bounded_signed_strength(
    value: int | float,
    *,
    maximum: int = 3,
    allow_zero: bool = False,
) -> int:
    normalized = int(round(float(value)))
    if normalized == 0:
        return 0 if allow_zero else 1
    return max(-maximum, min(normalized, maximum))


def _parse_ref_string(value: str) -> dict[str, str]:
    delimiter = "=" if "=" in value else ":" if ":" in value else None
    if delimiter is None:
        raise ValidationFailure(
            "Provider member refs must use 'gene_id=state_id' or 'gene_id:state_id' when encoded as strings."
        )
    gene_id, state_id = value.split(delimiter, 1)
    if not gene_id.strip() or not state_id.strip():
        raise ValidationFailure("Provider member refs must be non-empty.")
    return {
        "gene_id": gene_id.strip(),
        "state_id": state_id.strip(),
    }


def _member_list(
    mapping: Mapping[str, object],
    *,
    key: str,
) -> list[dict[str, str]] | None:
    raw_members = mapping.get(key)
    if raw_members is None:
        return None
    if not isinstance(raw_members, list):
        raise ValidationFailure(f"Expected {key!r} to be a list.")
    members: list[dict[str, str]] = []
    for item in cast(list[object], raw_members):
        if isinstance(item, str):
            members.append(_parse_ref_string(item))
            continue
        if isinstance(item, Mapping):
            members.append(
                {
                    "gene_id": _require_string(
                        cast(Mapping[str, object], item), "gene_id"
                    ),
                    "state_id": _require_string(
                        cast(Mapping[str, object], item), "state_id"
                    ),
                }
            )
            continue
        raise ValidationFailure(f"Expected {key!r} members to be strings or objects.")
    return members


def _normalized_members_from_value(value: object) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValidationFailure("Provider members must be a list when provided.")
    members: list[dict[str, str]] = []
    for item in cast(list[object], value):
        if isinstance(item, str):
            members.append(_parse_ref_string(item))
            continue
        if isinstance(item, Mapping):
            members.append(
                {
                    "gene_id": _require_string(
                        cast(Mapping[str, object], item), "gene_id"
                    ),
                    "state_id": _require_string(
                        cast(Mapping[str, object], item), "state_id"
                    ),
                }
            )
            continue
        raise ValidationFailure(
            "Provider members must contain strings or objects when provided."
        )
    return members


def _normalize_graph_directive_name(value: str) -> str:
    normalized = value.strip()
    if normalized in {
        "screen_include",
        "screen_exclude",
        "linkage_positive",
        "linkage_negative",
    }:
        return normalized
    aliases = {
        "screen_together": "screen_include",
        "screen_keep_together": "screen_include",
        "screen_pair": "screen_include",
        "screen_apart": "screen_exclude",
        "screen_separate": "screen_exclude",
        "link_together": "linkage_positive",
        "link_positive": "linkage_positive",
        "link_apart": "linkage_negative",
        "link_negative": "linkage_negative",
    }
    return aliases.get(normalized, normalized)


def _normalize_provider_belief_mapping(
    mapping: Mapping[str, object],
    *,
    original_idea: CanonicalizationIdea | None,
) -> dict[str, object]:
    normalized = dict(mapping)
    if "id" not in normalized and original_idea is not None:
        normalized["id"] = original_idea.belief_id
    if "confidence_level" not in normalized and original_idea is not None:
        normalized["confidence_level"] = original_idea.confidence_level
    if "evidence_sources" not in normalized:
        normalized["evidence_sources"] = list(
            original_idea.evidence_sources if original_idea is not None else ("other",)
        )
    if "rationale" not in normalized:
        normalized["rationale"] = (
            _require_string(mapping, "reasoning")
            if isinstance(mapping.get("reasoning"), str)
            else (
                _require_string(mapping, "justification")
                if isinstance(mapping.get("justification"), str)
                else (
                    original_idea.rationale
                    if original_idea is not None
                    else _require_string(mapping, "kind")
                )
            )
        )

    kind = _require_string(normalized, "kind")
    if kind == "idea":
        if "gene" not in normalized:
            surface_gene = mapping.get("surface_gene")
            target_state = mapping.get("target_state")
            raw_target = mapping.get("target")
            raw_state = mapping.get("state")
            if isinstance(surface_gene, str) and isinstance(target_state, str):
                normalized["gene"] = {
                    "gene_id": surface_gene.strip(),
                    "state_id": target_state.strip(),
                }
            elif isinstance(raw_target, str) and isinstance(raw_state, str):
                normalized["gene"] = {
                    "gene_id": raw_target.strip(),
                    "state_id": raw_state.strip(),
                }
        if "effect_strength" in normalized and isinstance(
            normalized["effect_strength"], int | float
        ):
            normalized["effect_strength"] = _bounded_signed_strength(
                normalized["effect_strength"]
            )
        elif "effect_strength" not in normalized and original_idea is not None:
            strength = _bounded_positive_strength(original_idea.confidence_level)
            normalized["effect_strength"] = (
                -strength if original_idea.effect == "hurt" else strength
            )
        normalized.pop("surface_gene", None)
        normalized.pop("target_state", None)
        normalized.pop("target", None)
        normalized.pop("state", None)
        normalized.pop("reasoning", None)
        normalized.pop("justification", None)
        return normalized

    if kind == "relation":
        existing_members = _normalized_members_from_value(normalized.get("members"))
        if existing_members is not None:
            normalized["members"] = existing_members
        if "members" not in normalized:
            members = _member_list(mapping, key="surface_members") or _member_list(
                mapping, key="target_members"
            )
            if members is not None:
                normalized["members"] = members
        if "relation" not in normalized and original_idea is not None:
            normalized["relation"] = original_idea.relation
        if "strength" in normalized and isinstance(normalized["strength"], int | float):
            normalized["strength"] = _bounded_positive_strength(normalized["strength"])
        elif "strength" not in normalized and original_idea is not None:
            normalized["strength"] = _bounded_positive_strength(
                original_idea.confidence_level
            )
        if "joint_effect_strength" in normalized and isinstance(
            normalized["joint_effect_strength"], int | float
        ):
            normalized["joint_effect_strength"] = _bounded_signed_strength(
                normalized["joint_effect_strength"],
                allow_zero=True,
            )
        elif (
            "joint_effect_strength" not in normalized
            and original_idea is not None
            and isinstance(normalized.get("relation"), str)
        ):
            relation = cast(str, normalized["relation"])
            joint_strength = _bounded_positive_strength(original_idea.confidence_level)
            normalized["joint_effect_strength"] = (
                -joint_strength
                if relation in {"conflict", "exclusion"}
                else joint_strength
            )
        normalized.pop("surface_members", None)
        normalized.pop("target_members", None)
        normalized.pop("reasoning", None)
        normalized.pop("justification", None)
        return normalized

    if kind == "expert_prior":
        raw_target = normalized.get("target")
        if isinstance(raw_target, str):
            normalized["target"] = {
                "target_kind": "main_effect",
                "gene": _parse_ref_string(raw_target),
            }
        if "target" not in normalized:
            members = _member_list(mapping, key="surface_members")
            surface_gene = mapping.get("surface_gene")
            target_state = mapping.get("target_state")
            if members is not None and len(members) == 2:
                normalized["target"] = {
                    "target_kind": "pair_effect",
                    "members": members,
                }
            elif isinstance(surface_gene, str) and isinstance(target_state, str):
                normalized["target"] = {
                    "target_kind": "main_effect",
                    "gene": {
                        "gene_id": surface_gene.strip(),
                        "state_id": target_state.strip(),
                    },
                }
        if "prior_family" not in normalized:
            normalized["prior_family"] = "normal"
        if "mean" not in normalized:
            mean = _float_from_any(mapping, "mean", "prior_mean")
            if mean is not None:
                normalized["mean"] = mean
        if "scale" not in normalized:
            scale = _float_from_any(mapping, "scale", "prior_scale")
            if scale is not None:
                normalized["scale"] = scale
        normalized.pop("surface_gene", None)
        normalized.pop("target_state", None)
        normalized.pop("surface_members", None)
        normalized.pop("prior_mean", None)
        normalized.pop("prior_scale", None)
        normalized.pop("reasoning", None)
        normalized.pop("justification", None)
        return normalized

    if kind == "graph_directive":
        existing_members = _normalized_members_from_value(normalized.get("members"))
        if existing_members is not None:
            normalized["members"] = existing_members
        if "members" not in normalized:
            members = _member_list(mapping, key="surface_members") or _member_list(
                mapping, key="target_members"
            )
            if members is not None:
                normalized["members"] = members
        if "strength" in normalized and isinstance(normalized["strength"], int | float):
            normalized["strength"] = _bounded_positive_strength(normalized["strength"])
        elif "strength" not in normalized and original_idea is not None:
            normalized["strength"] = _bounded_positive_strength(
                original_idea.confidence_level
            )
        if isinstance(normalized.get("directive"), str):
            normalized["directive"] = _normalize_graph_directive_name(
                cast(str, normalized["directive"])
            )
        normalized.pop("surface_members", None)
        normalized.pop("target_members", None)
        normalized.pop("reasoning", None)
        normalized.pop("justification", None)
        return normalized

    return normalized


def _registry_prompt_payload(request: CanonicalizationRequest) -> dict[str, JsonValue]:
    surface: dict[str, JsonValue] = {}
    serialized_registry = request.registry.to_dict()
    for gene_id, raw_definition in serialized_registry.items():
        definition = cast(dict[str, JsonValue], raw_definition)
        surface[gene_id] = {
            "description": definition.get("description"),
            "aliases": definition.get("aliases", []),
            "states": definition.get("states", []),
            "state_descriptions": definition.get("state_descriptions", {}),
            "state_aliases": definition.get("state_aliases", {}),
            "surface_kind": definition.get("surface_kind"),
            "semantic_level": definition.get("semantic_level"),
            "materializable": definition.get("materializable"),
            "code_scopes": definition.get("code_scopes", []),
            "risk_hints": definition.get("risk_hints", []),
        }
    return {
        "surface_summary": cast(
            dict[str, JsonValue], to_json_value(request.registry.surface_summary())
        ),
        "surface": surface,
    }


def _ideas_prompt_payload(
    ideas: Sequence[object],
) -> list[JsonValue]:
    return cast(list[JsonValue], to_json_value(list(ideas)))


def _system_prompt() -> str:
    return (
        "You convert rough optimization ideas into typed autoclanker Bayesian beliefs.\n"
        "Bayes must only operate on typed canonical features.\n"
        "Never let free text directly influence the posterior.\n"
        "Prefer existing registry entries whenever possible.\n"
        "Only create session-local overlay genes when no existing surface entry cleanly captures the idea.\n"
        "Overlay genes must be non-materializable strategy/risk style surface entries scoped to existing code areas.\n"
        "Use conservative typed beliefs. Prefer idea or relation over expert_prior or graph_directive unless the idea clearly asks for explicit prior geometry or graph screening.\n"
        "Keep the incoming belief_id unchanged.\n"
        "Return JSON only, with this exact top-level shape:\n"
        "{\n"
        '  "suggestions": [\n'
        "    {\n"
        '      "belief_id": "idea_001",\n'
        '      "summary": "short explanation",\n'
        '      "confidence_score": 0.0,\n'
        '      "matched_evidence": ["phrase"],\n'
        '      "needs_review": false,\n'
        '      "belief": null,\n'
        '      "overlay_genes": []\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Allowed belief kinds inside belief are: idea, relation, expert_prior, graph_directive.\n"
        "If you cannot justify a typed belief, set belief to null and needs_review to true.\n"
        "For expert_prior use fields target, prior_family, mean, and scale. Do not use alias names like prior_mean or prior_scale.\n"
        "For graph_directive use fields members, directive, and strength.\n"
        "For idea beliefs use effect_strength between -3 and 3 and keep nonzero.\n"
        "For relation beliefs use relation in synergy/conflict/dependency/exclusion, relation strength between 1 and 3, and joint_effect_strength between -3 and 3 when present; zero is allowed for a neutral pair effect.\n"
        "For expert_prior use conservative means/scales.\n"
        "For graph_directive use only when explicit screening/linkage intent is present, with strength between 1 and 3.\n"
    )


def _user_prompt(request: CanonicalizationRequest) -> str:
    payload = {
        "session_context": cast(
            dict[str, JsonValue], to_json_value(request.session_context)
        ),
        "registry": _registry_prompt_payload(request),
        "ideas": _ideas_prompt_payload(request.ideas),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class AnthropicCanonicalizationConfig:
    api_key: str
    model: str
    api_url: str
    timeout_sec: int
    max_tokens: int
    temperature: float
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_backoff_sec: float = _DEFAULT_RETRY_BACKOFF_SEC


def _is_retryable_http_error(exc: HTTPError) -> bool:
    return exc.code in {429, 500, 502, 503, 504, 529}


def _is_retryable_url_error(exc: URLError) -> bool:
    reason = exc.reason
    if isinstance(reason, TimeoutError):
        return True
    normalized = str(reason).lower()
    return any(
        token in normalized
        for token in (
            "timed out",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
        )
    )


belief_from_mapping = _belief_from_mapping
bool_or_default = _bool_or_default
extract_json_object = _extract_json_object
float_or_none = _float_or_none
is_retryable_http_error = _is_retryable_http_error
is_retryable_url_error = _is_retryable_url_error
mapping_to_overlay_gene = _mapping_to_overlay_gene
normalize_provider_belief_mapping = _normalize_provider_belief_mapping
require_string = _require_string
string_list = _string_list
system_prompt = _system_prompt
user_prompt = _user_prompt


class AnthropicCanonicalizationModel:
    def __init__(self, config: AnthropicCanonicalizationConfig) -> None:
        self._config = config
        self.name = f"anthropic:{config.model}"

    def _request_body(self, request: CanonicalizationRequest) -> dict[str, JsonValue]:
        return {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "system": _system_prompt(),
            "messages": [
                {
                    "role": "user",
                    "content": _user_prompt(request),
                }
            ],
        }

    def _response_text(self, request: CanonicalizationRequest) -> str:
        body = json.dumps(self._request_body(request)).encode("utf-8")
        http_request = Request(
            self._config.api_url,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self._config.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        rendered: str | None = None
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                with urlopen(
                    http_request, timeout=self._config.timeout_sec
                ) as response:
                    rendered = response.read().decode("utf-8")
                break
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")
                if _is_retryable_http_error(exc) and attempt + 1 < attempts:
                    time.sleep(self._config.retry_backoff_sec * (2**attempt))
                    continue
                raise ValidationFailure(
                    f"Anthropic canonicalizer request failed with HTTP {exc.code}: {body_text}"
                ) from exc
            except URLError as exc:
                if _is_retryable_url_error(exc) and attempt + 1 < attempts:
                    time.sleep(self._config.retry_backoff_sec * (2**attempt))
                    continue
                raise ValidationFailure(
                    f"Anthropic canonicalizer request failed: {exc.reason}"
                ) from exc
        if rendered is None:
            raise ValidationFailure("Anthropic canonicalizer request produced no body.")
        raw_payload = json.loads(rendered)
        if not isinstance(raw_payload, dict):
            raise ValidationFailure(
                "Anthropic canonicalizer returned a non-object API response."
            )
        payload = cast(dict[str, object], raw_payload)
        content = payload.get("content")
        if not isinstance(content, list):
            raise ValidationFailure(
                "Anthropic canonicalizer response did not contain content blocks."
            )
        parts: list[str] = []
        for block in cast(list[object], content):
            if not isinstance(block, dict):
                continue
            block_mapping = cast(dict[str, object], block)
            if block_mapping.get("type") != "text":
                continue
            text = block_mapping.get("text")
            if isinstance(text, str):
                parts.append(text)
        if not parts:
            raise ValidationFailure(
                "Anthropic canonicalizer response did not contain text output."
            )
        return "\n".join(parts)

    def canonicalize(
        self,
        request: CanonicalizationRequest,
    ) -> tuple[CanonicalizationSuggestion, ...]:
        response_text = self._response_text(request)
        response_payload = _extract_json_object(response_text)
        raw_suggestions = response_payload.get("suggestions")
        if not isinstance(raw_suggestions, list):
            raise ValidationFailure(
                "Anthropic canonicalizer must return a JSON object with a suggestions list."
            )
        suggestions: list[CanonicalizationSuggestion] = []
        idea_lookup = {idea.belief_id: idea for idea in request.ideas}
        for raw_item in cast(list[object], raw_suggestions):
            if not isinstance(raw_item, Mapping):
                raise ValidationFailure(
                    "Anthropic canonicalizer suggestions must be objects."
                )
            item = cast(Mapping[str, object], raw_item)
            belief_id = _require_string(item, "belief_id")
            summary = _require_string(item, "summary")
            confidence_score = _float_or_none(item, "confidence_score")
            matched_evidence = _string_list(item, "matched_evidence")
            needs_review = _bool_or_default(item, "needs_review", False)
            raw_overlay_genes = item.get("overlay_genes")
            overlay_genes: tuple[SurfaceOverlayGene, ...] = ()
            if raw_overlay_genes is not None:
                if not isinstance(raw_overlay_genes, list):
                    raise ValidationFailure(
                        "Anthropic canonicalizer overlay_genes must be a list."
                    )
                overlay_genes = tuple(
                    _mapping_to_overlay_gene(cast(Mapping[str, object], entry))
                    for entry in cast(list[object], raw_overlay_genes)
                )
            raw_belief = item.get("belief")
            belief: Belief | None = None
            if raw_belief is not None:
                if not isinstance(raw_belief, Mapping):
                    raise ValidationFailure(
                        "Anthropic canonicalizer belief entries must be objects."
                    )
                original_idea = idea_lookup.get(belief_id)
                belief = _belief_from_mapping(
                    _normalize_provider_belief_mapping(
                        cast(Mapping[str, object], raw_belief),
                        original_idea=original_idea,
                    ),
                    session_context=request.session_context,
                )
            suggestions.append(
                CanonicalizationSuggestion(
                    belief_id=belief_id,
                    belief=belief,
                    source="llm",
                    summary=summary,
                    matched_evidence=matched_evidence,
                    overlay_genes=overlay_genes,
                    confidence_score=confidence_score,
                    needs_review=needs_review,
                )
            )
        return tuple(suggestions)


def build_autoclanker_canonicalization_model() -> CanonicalizationModel:
    api_key = os.environ.get("AUTOCLANKER_ANTHROPIC_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if api_key is None or not api_key.strip():
        raise ValidationFailure(
            "Anthropic canonicalization requires ANTHROPIC_API_KEY or AUTOCLANKER_ANTHROPIC_API_KEY."
        )
    model = os.environ.get("AUTOCLANKER_ANTHROPIC_MODEL", _DEFAULT_MODEL).strip()
    api_url = os.environ.get("AUTOCLANKER_ANTHROPIC_API_URL", _DEFAULT_API_URL).strip()
    timeout_raw = os.environ.get("AUTOCLANKER_ANTHROPIC_TIMEOUT_SEC")
    max_tokens_raw = os.environ.get("AUTOCLANKER_ANTHROPIC_MAX_TOKENS")
    temperature_raw = os.environ.get("AUTOCLANKER_ANTHROPIC_TEMPERATURE")
    max_retries_raw = os.environ.get("AUTOCLANKER_ANTHROPIC_MAX_RETRIES")
    retry_backoff_raw = os.environ.get("AUTOCLANKER_ANTHROPIC_RETRY_BACKOFF_SEC")
    return AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key=api_key.strip(),
            model=model,
            api_url=api_url,
            timeout_sec=(
                int(timeout_raw) if timeout_raw is not None else _DEFAULT_TIMEOUT_SEC
            ),
            max_tokens=(
                int(max_tokens_raw)
                if max_tokens_raw is not None
                else _DEFAULT_MAX_TOKENS
            ),
            temperature=(
                float(temperature_raw)
                if temperature_raw is not None
                else _DEFAULT_TEMPERATURE
            ),
            max_retries=(
                int(max_retries_raw)
                if max_retries_raw is not None
                else _DEFAULT_MAX_RETRIES
            ),
            retry_backoff_sec=(
                float(retry_backoff_raw)
                if retry_backoff_raw is not None
                else _DEFAULT_RETRY_BACKOFF_SEC
            ),
        )
    )
