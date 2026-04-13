from __future__ import annotations

import email.message
import io
import json

from urllib.error import HTTPError

import pytest

from autoclanker.bayes_layer.canonicalization import (
    CanonicalizationIdea,
    CanonicalizationRequest,
)
from autoclanker.bayes_layer.providers import anthropic_canonicalizer
from autoclanker.bayes_layer.providers.anthropic_canonicalizer import (
    AnthropicCanonicalizationConfig,
    AnthropicCanonicalizationModel,
    build_autoclanker_canonicalization_model,
)
from autoclanker.bayes_layer.registry import build_fixture_registry
from autoclanker.bayes_layer.types import SessionContext
from tests.compliance import covers


@covers("M1-005")
def test_anthropic_provider_parses_json_wrapped_in_code_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
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

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """```json
{
  "suggestions": [
    {
      "belief_id": "idea_001",
      "summary": "Mapped the repeated incident idea onto compiled matching.",
      "confidence_score": 0.82,
      "matched_evidence": ["repeated incident shapes", "compiled matching path"],
      "needs_review": false,
      "belief": {
        "kind": "idea",
        "surface_gene": "parser.matcher",
        "target_state": "matcher_compiled",
        "reasoning": "Repeated incident shapes probably deserve a compiled matching path."
      },
      "overlay_genes": []
    }
  ]
}
```"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.belief is not None
    assert suggestion.belief.kind == "idea"
    assert suggestion.summary.startswith("Mapped the repeated incident idea")
    assert suggestion.confidence_score == 0.82


@covers("M1-005")
def test_anthropic_provider_parses_overlay_gene_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_002",
                rationale="Treat keeping a large neighborhood of log context in memory as a likely risk on long traces.",
                confidence_level=3,
                relation="synergy",
                effect=None,
                risk_names=("oom",),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_002",
      "summary": "Proposed a session-local memory-risk family for wide capture behavior.",
      "confidence_score": 0.69,
      "matched_evidence": ["large neighborhood of log context", "memory risk"],
      "needs_review": true,
      "belief": null,
      "overlay_genes": [
        {
          "gene_id": "risk.capture_memory_family",
          "states": ["risk_default", "risk_high"],
          "default_state": "risk_default",
          "description": "Session-local risk family for log-context memory pressure.",
          "surface_kind": "risk_family",
          "semantic_level": "risk",
          "materializable": false,
          "code_scopes": ["capture.window", "io.chunk"],
          "risk_hints": ["oom"]
        }
      ]
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.belief is None
    assert suggestion.needs_review is True
    assert suggestion.overlay_genes[0].gene_id == "risk.capture_memory_family"
    assert suggestion.overlay_genes[0].surface_kind == "risk_family"


@covers("M1-005")
def test_anthropic_provider_reads_messages_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(),
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
                    "content": [
                        {
                            "type": "text",
                            "text": '{"suggestions": []}',
                        }
                    ]
                }
            ).encode("utf-8")

    def _fake_urlopen(request_obj: object, timeout: int) -> _FakeResponse:
        del request_obj, timeout
        return _FakeResponse()

    monkeypatch.setattr(anthropic_canonicalizer, "urlopen", _fake_urlopen)
    suggestions = model.canonicalize(request)

    assert suggestions == ()


@covers("M1-005")
def test_anthropic_provider_retries_transient_overload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
            max_retries=2,
            retry_backoff_sec=0.01,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(),
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
                    "content": [
                        {
                            "type": "text",
                            "text": '{"suggestions": []}',
                        }
                    ]
                }
            ).encode("utf-8")

    attempts = {"count": 0}

    def _fake_urlopen(request_obj: object, timeout: int) -> _FakeResponse:
        del request_obj, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise HTTPError(
                url="https://example.invalid/v1/messages",
                code=529,
                msg="Overloaded",
                hdrs=email.message.Message(),
                fp=io.BytesIO(
                    b'{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
                ),
            )
        return _FakeResponse()

    def _fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(anthropic_canonicalizer, "urlopen", _fake_urlopen)
    monkeypatch.setattr(anthropic_canonicalizer.time, "sleep", _fake_sleep)
    suggestions = model.canonicalize(request)

    assert suggestions == ()
    assert attempts["count"] == 2


@covers("M1-005")
def test_anthropic_provider_normalizes_relation_shorthand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_003",
                rationale="These two parser changes likely reinforce each other.",
                confidence_level=3,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_003",
      "summary": "Mapped the parser pair into a typed relation.",
      "confidence_score": 0.77,
      "matched_evidence": ["reinforce each other"],
      "needs_review": false,
      "belief": {
        "kind": "relation",
        "surface_members": [
          "parser.matcher=matcher_compiled",
          "parser.plan=plan_context_pair"
        ],
        "reasoning": "Compiled matching and the context-pair plan reinforce each other."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    assert len(suggestions) == 1
    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "relation"
    assert belief.strength == 3
    assert belief.joint_effect_strength == 3
    assert {member.gene_id for member in belief.members} == {
        "parser.matcher",
        "parser.plan",
    }


@covers("M1-005")
def test_anthropic_provider_normalizes_expert_prior_shorthand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_004",
                rationale="I want a stronger explicit prior for compiled matching.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_004",
      "summary": "Upgraded the rough idea into a conservative expert prior.",
      "confidence_score": 0.74,
      "matched_evidence": ["explicit prior"],
      "needs_review": false,
      "belief": {
        "kind": "expert_prior",
        "surface_gene": "parser.matcher",
        "target_state": "matcher_compiled",
        "mean": 0.45,
        "scale": 0.3,
        "reasoning": "Compiled matching should have a positive main-effect prior."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    assert len(suggestions) == 1
    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "expert_prior"
    assert belief.prior_family == "normal"
    assert belief.target.target_kind == "main_effect"
    assert belief.target.gene.gene_id == "parser.matcher"
    assert belief.target.gene.state_id == "matcher_compiled"


@covers("M1-005")
def test_anthropic_provider_normalizes_expert_prior_alias_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_004b",
                rationale="I want a stronger explicit prior for compiled matching.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_004b",
      "summary": "Upgraded the rough idea into a conservative expert prior using provider alias fields.",
      "confidence_score": 0.74,
      "matched_evidence": ["explicit prior"],
      "needs_review": false,
      "belief": {
        "kind": "expert_prior",
        "surface_gene": "parser.matcher",
        "target_state": "matcher_compiled",
        "prior_mean": 0.45,
        "prior_scale": 0.3,
        "justification": "Compiled matching should have a positive main-effect prior."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "expert_prior"
    assert belief.mean == 0.45
    assert belief.scale == 0.3
    assert (
        belief.rationale
        == "Compiled matching should have a positive main-effect prior."
    )


@covers("M1-005")
def test_anthropic_provider_normalizes_string_target_refs_for_expert_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_004c",
                rationale="I want a stronger explicit prior for compiled matching.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_004c",
      "summary": "Upgraded the rough idea into a conservative expert prior using a string target ref.",
      "confidence_score": 0.74,
      "matched_evidence": ["explicit prior"],
      "needs_review": false,
      "belief": {
        "kind": "expert_prior",
        "target": "parser.matcher:matcher_compiled",
        "prior_family": "normal",
        "mean": 0.45,
        "scale": 0.3
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "expert_prior"
    assert belief.target.target_kind == "main_effect"
    assert belief.target.gene.gene_id == "parser.matcher"
    assert belief.target.gene.state_id == "matcher_compiled"


@covers("M1-005")
def test_anthropic_provider_clamps_confidence_four_idea_strength_to_schema_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_005",
                rationale="Compiled matching should help a lot here.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_005",
      "summary": "Mapped the idea to compiled matching.",
      "confidence_score": 0.91,
      "matched_evidence": ["compiled matching"],
      "needs_review": false,
      "belief": {
        "kind": "idea",
        "surface_gene": "parser.matcher",
        "target_state": "matcher_compiled",
        "reasoning": "Compiled matching should help a lot here."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "idea"
    assert belief.effect_strength == 3


@covers("M1-005")
def test_anthropic_provider_normalizes_idea_target_and_state_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_005b",
                rationale="Front-load the clustered incident pass and keep nearby breadcrumbs attached.",
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

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_005b",
      "summary": "Mapped the rough parser idea to a concrete strategy state using target/state aliases.",
      "confidence_score": 0.72,
      "matched_evidence": ["front-load", "nearby breadcrumbs"],
      "needs_review": false,
      "belief": {
        "kind": "idea",
        "target": "search.incident_cluster_pass",
        "state": "cluster_context_path",
        "effect_strength": 2
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "idea"
    assert belief.gene.gene_id == "search.incident_cluster_pass"
    assert belief.gene.state_id == "cluster_context_path"


@covers("M1-005")
def test_anthropic_provider_clamps_confidence_four_relation_strengths_to_schema_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_006",
                rationale="Compiled matching and context pairing strongly reinforce each other.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_006",
      "summary": "Mapped the pair into a typed relation.",
      "confidence_score": 0.88,
      "matched_evidence": ["reinforce each other"],
      "needs_review": false,
      "belief": {
        "kind": "relation",
        "surface_members": [
          "parser.matcher=matcher_compiled",
          "parser.plan=plan_context_pair"
        ],
        "reasoning": "Compiled matching and context pairing strongly reinforce each other."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "relation"
    assert belief.strength == 3
    assert belief.joint_effect_strength == 3


@covers("M1-005")
def test_anthropic_provider_preserves_neutral_relation_joint_effect_strength(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_006b",
                rationale="These two parser changes belong together but the pair effect itself should stay neutral.",
                confidence_level=2,
                relation="dependency",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_006b",
      "summary": "Mapped the pair into a typed dependency relation with a neutral joint effect.",
      "confidence_score": 0.61,
      "matched_evidence": ["belong together", "neutral pair effect"],
      "needs_review": false,
      "belief": {
        "kind": "relation",
        "relation": "dependency",
        "strength": 2,
        "joint_effect_strength": 0,
        "surface_members": [
          "parser.matcher=matcher_compiled",
          "parser.plan=plan_context_pair"
        ],
        "reasoning": "These parser changes depend on each other, but the direct pair effect is neutral."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "relation"
    assert belief.relation == "dependency"
    assert belief.joint_effect_strength == 0


@covers("M1-005")
def test_anthropic_provider_clamps_graph_directive_strength_to_schema_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_007",
                rationale="Keep this pair in the interaction screen.",
                confidence_level=4,
                relation="synergy",
                effect=None,
                risk_names=(),
                scope=None,
                evidence_sources=("intuition",),
                raw_mapping={},
            ),
        ),
    )

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_007",
      "summary": "Promoted the pair to a graph directive.",
      "confidence_score": 0.79,
      "matched_evidence": ["interaction screen"],
      "needs_review": false,
      "belief": {
        "kind": "graph_directive",
        "directive": "screen_include",
        "surface_members": [
          "parser.matcher=matcher_compiled",
          "parser.plan=plan_context_pair"
        ],
        "reasoning": "Keep this pair in the interaction screen."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "graph_directive"
    assert belief.strength == 3


@covers("M1-005")
def test_anthropic_provider_normalizes_graph_directive_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = AnthropicCanonicalizationModel(
        AnthropicCanonicalizationConfig(
            api_key="test-key",
            model="claude-sonnet-4-20250514",
            api_url="https://example.invalid/v1/messages",
            timeout_sec=10,
            max_tokens=600,
            temperature=0.0,
        )
    )
    request = CanonicalizationRequest(
        session_context=SessionContext(era_id="era_log_parser_v1"),
        registry=build_fixture_registry(),
        ideas=(
            CanonicalizationIdea(
                belief_id="idea_007b",
                rationale="Keep this pair together in the interaction screen.",
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

    def _fake_response_text(_request: CanonicalizationRequest) -> str:
        return """{
  "suggestions": [
    {
      "belief_id": "idea_007b",
      "summary": "Promoted the pair to a graph directive using a provider alias.",
      "confidence_score": 0.78,
      "matched_evidence": ["together in the interaction screen"],
      "needs_review": false,
      "belief": {
        "kind": "graph_directive",
        "directive": "screen_together",
        "surface_members": [
          "parser.matcher=matcher_compiled",
          "parser.plan=plan_context_pair"
        ],
        "reasoning": "Keep this pair together in the interaction screen."
      },
      "overlay_genes": []
    }
  ]
}"""

    monkeypatch.setattr(model, "_response_text", _fake_response_text)
    suggestions = model.canonicalize(request)

    belief = suggestions[0].belief
    assert belief is not None
    assert belief.kind == "graph_directive"
    assert belief.directive == "screen_include"


@covers("M1-005")
def test_anthropic_builder_uses_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AUTOCLANKER_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    model = build_autoclanker_canonicalization_model()

    assert model.name == "anthropic:claude-sonnet-4-20250514"
