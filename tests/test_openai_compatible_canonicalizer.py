from __future__ import annotations

import json

from typing import cast
from urllib.request import Request

import pytest

from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationIdea,
    CanonicalizationRequest,
    load_canonicalization_model,
)
from autoclanker.bayes_layer.providers import openai_compatible_canonicalizer
from autoclanker.bayes_layer.providers.openai_compatible_canonicalizer import (
    OpenAICompatibleCanonicalizationConfig,
    OpenAICompatibleCanonicalizationModel,
    build_autoclanker_canonicalization_model,
)
from autoclanker.bayes_layer.registry import build_fixture_registry
from autoclanker.bayes_layer.types import SessionContext
from tests.compliance import covers


def _request() -> CanonicalizationRequest:
    return CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_001",
                rationale="Repeated incident shapes probably deserve a compiled matching path.",
                confidence_level=2,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )


@covers("M1-005")
def test_openai_compatible_provider_reads_chat_completions_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = OpenAICompatibleCanonicalizationModel(
        OpenAICompatibleCanonicalizationConfig(
            api_key="test-key",
            model="frontier-test",
            api_url="https://example.invalid/v1/chat/completions",
            timeout_sec=10,
            max_tokens=600,
            max_tokens_field="max_completion_tokens",
            temperature=None,
        )
    )

    class _FakeResponse:
        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            del exc_type, exc, traceback

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "suggestions": [
                                            {
                                                "belief_id": "idea_001",
                                                "summary": "Mapped the repeated incident idea onto compiled matching.",
                                                "confidence_score": 0.82,
                                                "matched_evidence": [
                                                    "repeated incident shapes",
                                                    "compiled matching path",
                                                ],
                                                "needs_review": False,
                                                "belief": {
                                                    "kind": "idea",
                                                    "surface_gene": "parser.matcher",
                                                    "target_state": "matcher_compiled",
                                                    "reasoning": "Repeated incident shapes probably deserve a compiled matching path.",
                                                },
                                                "overlay_genes": [],
                                            }
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    captured_body: dict[str, object] = {}

    def _fake_urlopen(request_obj: object, timeout: int) -> _FakeResponse:
        del timeout
        data = cast(bytes | None, cast(Request, request_obj).data)
        assert data is not None
        captured_body.update(json.loads(data.decode("utf-8")))
        return _FakeResponse()

    monkeypatch.setattr(openai_compatible_canonicalizer, "urlopen", _fake_urlopen)
    suggestions = model.canonicalize(_request())

    assert suggestions[0].belief is not None
    assert suggestions[0].belief.kind == "idea"
    assert suggestions[0].summary.startswith("Mapped the repeated incident idea")
    assert captured_body["model"] == "frontier-test"
    assert captured_body["max_completion_tokens"] == 600
    assert "temperature" not in captured_body


@covers("M1-005")
def test_openai_compatible_builder_uses_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTOCLANKER_OPENAI_MODEL", "gpt-test-xhigh")
    monkeypatch.setenv(
        "AUTOCLANKER_OPENAI_API_URL",
        "https://proxy.example.invalid/v1/chat/completions",
    )
    monkeypatch.setenv("AUTOCLANKER_OPENAI_MAX_TOKENS_FIELD", "max_completion_tokens")

    model = build_autoclanker_canonicalization_model()

    assert model.name == "openai-compatible:gpt-test-xhigh"


@covers("M1-005")
def test_openai_compatible_alias_loads_from_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AUTOCLANKER_OPENAI_MODEL", "gpt-test")

    model = load_canonicalization_model("openai-compatible")

    assert model is not None
    assert model.name == "openai-compatible:gpt-test"
