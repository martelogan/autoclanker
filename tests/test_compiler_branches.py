from __future__ import annotations

import pytest

from autoclanker.bayes_layer import (
    EraState,
    build_fixture_registry,
    compile_beliefs,
    ingest_human_beliefs,
    validate_adapter_config,
)
from autoclanker.bayes_layer.types import ValidationFailure


def test_compiler_covers_preference_masks_and_vram_branches() -> None:
    payload = {
        "session_context": {
            "era_id": "era_branch",
            "session_id": "branch_session",
            "user_profile": "expert",
        },
        "beliefs": [
            {
                "id": "idea_vram",
                "kind": "idea",
                "confidence_level": 3,
                "gene": {"gene_id": "capture.window", "state_id": "window_wide"},
                "effect_strength": 1,
                "complexity_delta": 2,
                "risk": {"oom": 2},
            },
            {
                "id": "rel_dep",
                "kind": "relation",
                "confidence_level": 2,
                "members": [
                    {"gene_id": "parser.matcher", "state_id": "matcher_compiled"},
                    {"gene_id": "parser.plan", "state_id": "plan_context_pair"},
                    {"gene_id": "emit.summary", "state_id": "summary_streaming"},
                ],
                "relation": "dependency",
                "strength": 2,
            },
            {
                "id": "rel_excl",
                "kind": "relation",
                "confidence_level": 2,
                "members": [
                    {"gene_id": "capture.window", "state_id": "window_wide"},
                    {"gene_id": "io.chunk", "state_id": "chunk_large"},
                ],
                "relation": "exclusion",
                "strength": 3,
            },
            {
                "id": "pref",
                "kind": "preference",
                "confidence_level": 3,
                "left_pattern": {
                    "members": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_compiled"},
                    ]
                },
                "right_pattern": {
                    "members": [
                        {"gene_id": "parser.matcher", "state_id": "matcher_jit"},
                    ]
                },
                "preference": "left",
                "strength": 4,
            },
            {
                "id": "hard_exclude",
                "kind": "constraint",
                "confidence_level": 4,
                "constraint_type": "hard_exclude",
                "severity": 3,
                "scope": [
                    {"gene_id": "io.chunk", "state_id": "chunk_large"},
                ],
            },
            {
                "id": "budget_cap",
                "kind": "constraint",
                "confidence_level": 2,
                "constraint_type": "budget_cap",
                "severity": 2,
                "scope": [
                    {"gene_id": "capture.window", "state_id": "window_wide"},
                ],
            },
            {
                "id": "vram_prior",
                "kind": "expert_prior",
                "confidence_level": 4,
                "target": {
                    "target_kind": "vram_effect",
                    "gene": {"gene_id": "capture.window", "state_id": "window_wide"},
                },
                "prior_family": "normal",
                "mean": 0.9,
                "scale": 0.2,
                "observation_weight": 2.0,
                "decay_override": {
                    "per_eval_multiplier": 0.9,
                    "cross_era_transfer": 0.1,
                },
            },
            {
                "id": "screen_exclude",
                "kind": "graph_directive",
                "confidence_level": 3,
                "members": [
                    {"gene_id": "parser.matcher", "state_id": "matcher_jit"},
                    {"gene_id": "parser.plan", "state_id": "plan_full_scan"},
                ],
                "directive": "screen_exclude",
                "strength": 2,
            },
            {
                "id": "link_negative",
                "kind": "graph_directive",
                "confidence_level": 3,
                "members": [
                    {"gene_id": "capture.window", "state_id": "window_wide"},
                    {"gene_id": "io.chunk", "state_id": "chunk_large"},
                ],
                "directive": "linkage_negative",
                "strength": 2,
            },
        ],
    }

    batch = ingest_human_beliefs(payload)
    compiled = compile_beliefs(
        batch,
        build_fixture_registry(),
        EraState(era_id="era_branch"),
    )

    assert compiled.vram_priors
    assert compiled.feasibility_priors
    assert compiled.hard_masks
    assert compiled.preference_observations
    assert compiled.candidate_generation_hints
    assert compiled.linkage_hints
    assert any(preview.warnings for preview in compiled.belief_previews)


def test_invalid_adapter_mode_combination_is_rejected() -> None:
    payload = {
        "adapter": {
            "kind": "fixture",
            "mode": "local_repo_path",
            "repo_path": ".local/fixture",
        }
    }

    with pytest.raises(ValidationFailure):
        validate_adapter_config(payload)


def test_first_party_auto_mode_requires_a_resolution_hint() -> None:
    payload = {
        "adapter": {
            "kind": "autoresearch",
            "mode": "auto",
        }
    }

    with pytest.raises(ValidationFailure):
        validate_adapter_config(payload)


def test_auto_mode_is_rejected_for_generic_adapters() -> None:
    payload = {
        "adapter": {
            "kind": "python_module",
            "mode": "auto",
            "python_module": "tests.fixtures.python_module_adapter",
        }
    }

    with pytest.raises(ValidationFailure):
        validate_adapter_config(payload)


def test_first_party_auto_mode_accepts_any_single_usable_hint() -> None:
    payload = {
        "adapter": {
            "kind": "autoresearch",
            "mode": "auto",
            "repo_path": ".local/missing-autoresearch",
            "python_module": "tests.fixtures.python_module_adapter",
        }
    }

    config = validate_adapter_config(payload)

    assert config.mode == "auto"
    assert config.python_module == "tests.fixtures.python_module_adapter"
