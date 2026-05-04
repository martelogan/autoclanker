from __future__ import annotations

import json
import os
import time

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationModel,
    CanonicalizationRequest,
    CanonicalizationSuggestion,
    SurfaceOverlayGene,
)
from autoclanker.bayes_layer.providers.anthropic_canonicalizer import (
    belief_from_mapping,
    bool_or_default,
    extract_json_object,
    float_or_none,
    is_retryable_http_error,
    is_retryable_url_error,
    mapping_to_overlay_gene,
    normalize_provider_belief_mapping,
    require_string,
    string_list,
    system_prompt,
    user_prompt,
)
from autoclanker.bayes_layer.types import Belief, JsonValue, ValidationFailure

_DEFAULT_MODEL = "gpt-5.5"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_TIMEOUT_SEC = 60
_DEFAULT_MAX_TOKENS = 1400
_DEFAULT_MAX_TOKENS_FIELD = "max_tokens"
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF_SEC = 1.0


@dataclass(frozen=True, slots=True)
class OpenAICompatibleCanonicalizationConfig:
    api_key: str
    model: str
    api_url: str
    timeout_sec: int
    max_tokens: int
    max_tokens_field: str
    temperature: float | None = None
    max_retries: int = _DEFAULT_MAX_RETRIES
    retry_backoff_sec: float = _DEFAULT_RETRY_BACKOFF_SEC


def _default_api_url() -> str:
    base_url = os.environ.get("AUTOCLANKER_OPENAI_BASE_URL") or os.environ.get(
        "OPENAI_BASE_URL", _DEFAULT_BASE_URL
    )
    return f"{base_url.rstrip('/')}/chat/completions"


def _content_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for item in cast(list[object], content):
        if not isinstance(item, Mapping):
            continue
        item_mapping = cast(Mapping[str, object], item)
        text = item_mapping.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts) if parts else None


def _extract_response_text(payload: Mapping[str, object]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    choices = payload.get("choices")
    if isinstance(choices, list):
        parts: list[str] = []
        for choice in cast(list[object], choices):
            if not isinstance(choice, Mapping):
                continue
            choice_mapping = cast(Mapping[str, object], choice)
            message = choice_mapping.get("message")
            if isinstance(message, Mapping):
                text = _content_text(cast(Mapping[str, object], message).get("content"))
                if text:
                    parts.append(text)
            text = _content_text(choice_mapping.get("text"))
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    output = payload.get("output")
    if isinstance(output, list):
        parts = []
        for item in cast(list[object], output):
            if not isinstance(item, Mapping):
                continue
            item_mapping = cast(Mapping[str, object], item)
            text = _content_text(item_mapping.get("content"))
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    raise ValidationFailure(
        "OpenAI-compatible canonicalizer response did not contain text output."
    )


class OpenAICompatibleCanonicalizationModel:
    def __init__(self, config: OpenAICompatibleCanonicalizationConfig) -> None:
        self._config = config
        self.name = f"openai-compatible:{config.model}"

    def _request_body(self, request: CanonicalizationRequest) -> dict[str, JsonValue]:
        body: dict[str, JsonValue] = {
            "model": self._config.model,
            self._config.max_tokens_field: self._config.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_prompt(request),
                },
            ],
        }
        if self._config.temperature is not None:
            body["temperature"] = self._config.temperature
        return body

    def _response_text(self, request: CanonicalizationRequest) -> str:
        body = json.dumps(self._request_body(request)).encode("utf-8")
        http_request = Request(
            self._config.api_url,
            data=body,
            method="POST",
            headers={
                "authorization": f"Bearer {self._config.api_key}",
                "content-type": "application/json",
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
                if is_retryable_http_error(exc) and attempt + 1 < attempts:
                    time.sleep(self._config.retry_backoff_sec * (2**attempt))
                    continue
                raise ValidationFailure(
                    f"OpenAI-compatible canonicalizer request failed with HTTP {exc.code}: {body_text}"
                ) from exc
            except URLError as exc:
                if is_retryable_url_error(exc) and attempt + 1 < attempts:
                    time.sleep(self._config.retry_backoff_sec * (2**attempt))
                    continue
                raise ValidationFailure(
                    f"OpenAI-compatible canonicalizer request failed: {exc.reason}"
                ) from exc
        if rendered is None:
            raise ValidationFailure(
                "OpenAI-compatible canonicalizer request produced no body."
            )
        raw_payload = json.loads(rendered)
        if not isinstance(raw_payload, Mapping):
            raise ValidationFailure(
                "OpenAI-compatible canonicalizer returned a non-object API response."
            )
        return _extract_response_text(cast(Mapping[str, object], raw_payload))

    def canonicalize(
        self,
        request: CanonicalizationRequest,
    ) -> tuple[CanonicalizationSuggestion, ...]:
        response_text = self._response_text(request)
        response_payload = extract_json_object(response_text)
        raw_suggestions = response_payload.get("suggestions")
        if not isinstance(raw_suggestions, list):
            raise ValidationFailure(
                "OpenAI-compatible canonicalizer must return a JSON object with a suggestions list."
            )
        suggestions: list[CanonicalizationSuggestion] = []
        idea_lookup = {idea.belief_id: idea for idea in request.ideas}
        for raw_item in cast(list[object], raw_suggestions):
            if not isinstance(raw_item, Mapping):
                raise ValidationFailure(
                    "OpenAI-compatible canonicalizer suggestions must be objects."
                )
            item = cast(Mapping[str, object], raw_item)
            belief_id = require_string(item, "belief_id")
            summary = require_string(item, "summary")
            confidence_score = float_or_none(item, "confidence_score")
            matched_evidence = string_list(item, "matched_evidence")
            needs_review = bool_or_default(item, "needs_review", False)
            raw_overlay_genes = item.get("overlay_genes")
            overlay_genes: tuple[SurfaceOverlayGene, ...] = ()
            if raw_overlay_genes is not None:
                if not isinstance(raw_overlay_genes, list):
                    raise ValidationFailure(
                        "OpenAI-compatible canonicalizer overlay_genes must be a list."
                    )
                overlay_genes = tuple(
                    mapping_to_overlay_gene(cast(Mapping[str, object], entry))
                    for entry in cast(list[object], raw_overlay_genes)
                )
            raw_belief = item.get("belief")
            belief: Belief | None = None
            if raw_belief is not None:
                if not isinstance(raw_belief, Mapping):
                    raise ValidationFailure(
                        "OpenAI-compatible canonicalizer belief entries must be objects."
                    )
                original_idea = idea_lookup.get(belief_id)
                belief = belief_from_mapping(
                    normalize_provider_belief_mapping(
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
    api_key = os.environ.get("AUTOCLANKER_OPENAI_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if api_key is None or not api_key.strip():
        raise ValidationFailure(
            "OpenAI-compatible canonicalization requires OPENAI_API_KEY or AUTOCLANKER_OPENAI_API_KEY."
        )
    model = (
        os.environ.get("AUTOCLANKER_OPENAI_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or _DEFAULT_MODEL
    ).strip()
    api_url = os.environ.get("AUTOCLANKER_OPENAI_API_URL", _default_api_url()).strip()
    timeout_raw = os.environ.get("AUTOCLANKER_OPENAI_TIMEOUT_SEC")
    max_tokens_raw = os.environ.get("AUTOCLANKER_OPENAI_MAX_TOKENS")
    max_tokens_field = os.environ.get(
        "AUTOCLANKER_OPENAI_MAX_TOKENS_FIELD", _DEFAULT_MAX_TOKENS_FIELD
    ).strip()
    temperature_raw = os.environ.get("AUTOCLANKER_OPENAI_TEMPERATURE")
    max_retries_raw = os.environ.get("AUTOCLANKER_OPENAI_MAX_RETRIES")
    retry_backoff_raw = os.environ.get("AUTOCLANKER_OPENAI_RETRY_BACKOFF_SEC")
    return OpenAICompatibleCanonicalizationModel(
        OpenAICompatibleCanonicalizationConfig(
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
            max_tokens_field=max_tokens_field,
            temperature=(
                float(temperature_raw) if temperature_raw is not None else None
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
