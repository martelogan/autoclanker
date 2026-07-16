from __future__ import annotations

from pathlib import Path

import pytest

from autoclanker.bayes_layer import (
    EraState,
    build_fixture_registry,
    compile_beliefs,
    ingest_belief_input,
    ingest_human_beliefs,
    load_serialized_payload,
    preview_compiled_beliefs,
    validate_adapter_config,
    validate_eval_result,
)
from autoclanker.bayes_layer.belief_io import (
    load_inline_ideas_payload,
    load_serialized_payload_from_text,
)
from autoclanker.bayes_layer.types import ValidationFailure
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


@covers("M1-001")
@pytest.mark.parametrize(
    ("relative_path", "belief_count"),
    [
        ("examples/human_beliefs/basic_session.json", 4),
        ("examples/human_beliefs/basic_session.yaml", 4),
        ("examples/human_beliefs/expert_session.json", 5),
        ("examples/human_beliefs/expert_session.yaml", 5),
    ],
)
def test_example_belief_files_validate(
    relative_path: str,
    belief_count: int,
) -> None:
    batch = ingest_human_beliefs(load_serialized_payload(ROOT / relative_path))

    assert len(batch.beliefs) == belief_count
    assert batch.session_context.era_id == "era_003"


@covers("M1-003")
def test_basic_beliefs_preview_and_compile() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/human_beliefs/basic_session.yaml"
    )
    batch = ingest_human_beliefs(payload)
    registry = build_fixture_registry()

    preview = preview_compiled_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )
    compiled = compile_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )

    assert preview.belief_previews[0].compile_status == "compiled"
    assert preview.belief_previews[-1].compile_status == "metadata_only"
    assert compiled.main_effect_priors
    assert compiled.pair_priors
    assert compiled.candidate_generation_hints


@covers("M1-001", "M1-003")
def test_codebase_patterns_compile_to_candidate_generation_hint() -> None:
    batch = ingest_belief_input(
        {
            "session_context": {
                "era_id": "era_patterns_v1",
                "session_id": "patterns_session",
            },
            "beliefs": [
                {
                    "id": "patterns_001",
                    "kind": "codebase_patterns",
                    "confidence_level": 3,
                    "evidence_sources": ["code_inspection"],
                    "rationale": "Preserve existing codebase idioms when ranking designs.",
                    "scope": ["src/parser"],
                    "preferred_patterns": [
                        "Prefer the existing parser registry plumb point.",
                    ],
                    "discouraged_patterns": [
                        "Avoid bypassing the registry with one-off caches.",
                    ],
                    "test_conventions": [
                        "Use the existing golden parser fixture shape.",
                    ],
                    "artifact_paths": ["tmp/clankerbench/codebase_patterns.md"],
                    "source_digest": "sha256:patterns",
                }
            ],
        }
    )
    registry = build_fixture_registry()

    preview = preview_compiled_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )
    compiled = compile_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )

    assert preview.belief_previews[0].compile_status == "compiled"
    assert compiled.candidate_generation_hints
    hint = compiled.candidate_generation_hints[0].item
    assert hint.target_ref == "codebase_patterns:patterns_001"
    assert hint.prior_family == "codebase_pattern_hint"


@covers("M1-001", "M1-003")
def test_codebase_patterns_artifact_only_compiles_with_read_warning() -> None:
    batch = ingest_human_beliefs(
        {
            "session_context": {
                "era_id": "era_patterns_v1",
                "session_id": "patterns_session",
            },
            "beliefs": [
                {
                    "id": "patterns_artifact",
                    "kind": "codebase_patterns",
                    "confidence_level": 2,
                    "artifact_paths": ["tmp/clankerbench/codebase_patterns.md"],
                }
            ],
        }
    )
    registry = build_fixture_registry()

    preview = preview_compiled_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )
    compiled = compile_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )

    assert compiled.candidate_generation_hints
    assert preview.belief_previews[0].warnings == (
        "Read the referenced pattern artifact before candidate design; no inline pattern bullets were provided.",
    )


@covers("M1-001")
def test_codebase_patterns_reject_empty_pattern_payloads() -> None:
    empty_patterns: list[object] = []
    payload: dict[str, object] = {
        "session_context": {
            "era_id": "era_patterns_v1",
            "session_id": "patterns_session",
        },
        "beliefs": [
            {
                "id": "patterns_empty",
                "kind": "codebase_patterns",
                "confidence_level": 2,
                "preferred_patterns": empty_patterns,
            }
        ],
    }

    with pytest.raises(ValidationFailure, match="not valid under any"):
        ingest_human_beliefs(payload)


@covers("M1-001")
def test_belief_input_rejects_malformed_inline_payloads() -> None:
    with pytest.raises(ValidationFailure, match="was empty"):
        load_serialized_payload_from_text("")

    with pytest.raises(ValidationFailure, match="Failed to parse"):
        load_serialized_payload_from_text("{")

    with pytest.raises(ValidationFailure, match="was empty"):
        load_inline_ideas_payload("")

    with pytest.raises(ValidationFailure, match="string idea was empty"):
        load_inline_ideas_payload('""')

    with pytest.raises(ValidationFailure, match="must be a JSON string idea"):
        load_inline_ideas_payload("1")

    with pytest.raises(ValidationFailure, match="top-level 'ideas'"):
        load_inline_ideas_payload('{"unexpected": true}')

    with pytest.raises(ValidationFailure, match="ideas must be a list"):
        load_inline_ideas_payload('{"ideas": "nope"}')

    with pytest.raises(ValidationFailure, match="must not be empty"):
        load_inline_ideas_payload('[""]')


@covers("M1-003")
def test_expert_beliefs_compile_advanced_controls() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/human_beliefs/expert_session.json"
    )
    batch = ingest_human_beliefs(payload)
    registry = build_fixture_registry()
    compiled = compile_beliefs(
        batch,
        registry,
        EraState(era_id=batch.session_context.era_id),
    )

    assert compiled.main_effect_priors
    assert compiled.pair_priors
    assert compiled.feasibility_priors
    assert compiled.linkage_hints


@covers("M1-004")
def test_eval_and_adapter_examples_validate() -> None:
    eval_result = validate_eval_result(
        load_serialized_payload(ROOT / "examples/eval_results/valid_eval_result.json")
    )
    fixture_config = validate_adapter_config(
        load_serialized_payload(ROOT / "examples/adapters/fixture.yaml"),
        base_dir=ROOT / "examples/adapters",
    )
    autoresearch_config = validate_adapter_config(
        load_serialized_payload(ROOT / "examples/adapters/autoresearch.local.yaml"),
        base_dir=ROOT / "examples/adapters",
    )
    cevolve_config = validate_adapter_config(
        load_serialized_payload(ROOT / "examples/adapters/cevolve.local.yaml"),
        base_dir=ROOT / "examples/adapters",
    )

    goalloop_config = validate_adapter_config(
        load_serialized_payload(ROOT / "examples/adapters/goalloop.local.yaml"),
        base_dir=ROOT / "examples/adapters",
    )

    assert eval_result.status == "valid"
    assert fixture_config.kind == "fixture"
    assert autoresearch_config.kind == "autoresearch"
    assert cevolve_config.kind == "cevolve"
    assert goalloop_config.kind == "goalloop"


@covers("M1-002")
def test_duplicate_belief_ids_are_rejected() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/human_beliefs/basic_session.json"
    )
    raw_beliefs = list(payload["beliefs"])  # type: ignore[index]
    raw_beliefs[1]["id"] = raw_beliefs[0]["id"]  # type: ignore[index]
    payload["beliefs"] = raw_beliefs

    with pytest.raises(ValidationFailure):
        ingest_human_beliefs(payload)


@covers("M2-003")
def test_validate_eval_result_preserves_genotype_order() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/eval_results/valid_eval_result.json"
    )
    payload["intended_genotype"] = [
        {"gene_id": "emit.summary", "state_id": "summary_streaming"},
        {"gene_id": "io.chunk", "state_id": "chunk_large"},
        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"},
    ]
    payload["realized_genotype"] = [
        {"gene_id": "capture.window", "state_id": "window_wide"},
        {"gene_id": "parser.matcher", "state_id": "matcher_jit"},
    ]

    result = validate_eval_result(payload)

    assert tuple(ref.gene_id for ref in result.intended_genotype) == (
        "emit.summary",
        "io.chunk",
        "parser.matcher",
    )
    assert tuple(ref.gene_id for ref in result.realized_genotype) == (
        "capture.window",
        "parser.matcher",
    )


@covers("M2-003")
def test_validate_eval_result_normalizes_array_raw_metrics_to_counts() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/eval_results/valid_eval_result.json"
    )
    payload["raw_metrics"] = {
        "score": 0.72,
        "empty_samples": [],
        "sample_ids": ["a", "b"],
    }

    result = validate_eval_result(payload)

    assert result.raw_metrics["score"] == 0.72
    assert result.raw_metrics["empty_samples_count"] == 0
    assert result.raw_metrics["sample_ids_count"] == 2
    assert "empty_samples" not in result.raw_metrics
    assert result.failure_metadata is not None
    assert result.failure_metadata["raw_metrics_array_fields_normalized"] == [
        "empty_samples",
        "sample_ids",
    ]


@covers("M2-003")
def test_validate_eval_result_preserves_structured_evidence_metadata() -> None:
    payload = load_serialized_payload(
        ROOT / "examples/eval_results/valid_eval_result.json"
    )
    payload["evidence_metadata"] = {
        "paired_evidence": {
            "baseline_candidate_id": "cand_baseline",
            "p_value": 0.031,
            "confidence": "moderate",
        },
        "callsite_attribution": {
            "top_callsite": "example_parser/token_scan",
            "self_time_pct": 12.4,
        },
    }

    result = validate_eval_result(payload)

    assert result.evidence_metadata is not None
    assert result.evidence_metadata["paired_evidence"] == {
        "baseline_candidate_id": "cand_baseline",
        "p_value": 0.031,
        "confidence": "moderate",
    }
    assert result.evidence_metadata["callsite_attribution"] == {
        "top_callsite": "example_parser/token_scan",
        "self_time_pct": 12.4,
    }
