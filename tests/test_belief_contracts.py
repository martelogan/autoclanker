from __future__ import annotations

from pathlib import Path

import pytest

from autoclanker.bayes_layer import (
    EraState,
    build_fixture_registry,
    compile_beliefs,
    ingest_human_beliefs,
    load_serialized_payload,
    preview_compiled_beliefs,
    validate_adapter_config,
    validate_eval_result,
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

    assert eval_result.status == "valid"
    assert fixture_config.kind == "fixture"
    assert autoresearch_config.kind == "autoresearch"
    assert cevolve_config.kind == "cevolve"


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
