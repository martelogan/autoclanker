from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from autoclanker.bayes_layer.types import JsonValue, ValidationFailure


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationFailure(f"{label} must be a mapping.")
    return cast(Mapping[str, object], value)


def _require_bool(mapping: Mapping[str, object], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValidationFailure(f"Expected {key!r} to be a boolean.")
    return value


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise ValidationFailure(f"Expected {key!r} to be numeric.")
    return float(value)


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise ValidationFailure(f"Expected {key!r} to be an integer.")
    return value


def _require_str(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValidationFailure(f"Expected {key!r} to be a non-empty string.")
    return value.strip()


def _mapping_to_int_float(
    mapping: Mapping[str, object],
    *,
    label: str,
) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in mapping.items():
        if not isinstance(value, (int, float)):
            raise ValidationFailure(f"{label} values must be numeric.")
        result[int(key)] = float(value)
    return result


def _string_list(mapping: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise ValidationFailure(f"Expected {key!r} to be a list of strings.")
    items = cast(list[str], raw_items)
    return tuple(item.strip() for item in items)


@dataclass(frozen=True, slots=True)
class PriorDecayConfig:
    per_eval_multiplier: float
    cross_era_main_effect_transfer: float
    cross_era_pair_effect_transfer: float


@dataclass(frozen=True, slots=True)
class BeliefCompilerConfig:
    effect_strength_to_prior_mean: dict[int, float]
    confidence_level_to_variance_scale: dict[int, float]
    relation_strength_to_pair_mean: dict[int, float]
    prior_decay: PriorDecayConfig
    preview_required: bool


@dataclass(frozen=True, slots=True)
class UtilityConfig:
    lambda_sparsity: float
    lambda_vram: float
    soft_vram_limit_mb: float


@dataclass(frozen=True, slots=True)
class ObjectiveSurrogateConfig:
    include_active_gene_count: bool
    include_metadata_features: bool
    max_pair_features: int


@dataclass(frozen=True, slots=True)
class FeasibilitySurrogateConfig:
    enable_failure_modes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QueryPolicyConfig:
    enabled: bool
    max_queries_per_era: int
    min_expected_value_of_information: float
    allowed_query_types: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CommitPolicyConfig:
    posterior_gain_probability_threshold: float
    min_valid_probability: float
    epsilon_commit_floor: float


@dataclass(frozen=True, slots=True)
class SessionArtifactConfig:
    manifest: str
    beliefs: str
    surface_snapshot: str
    surface_overlay: str
    canonicalization_summary: str
    compiled_preview: str
    compiled_priors: str
    observations: str
    posterior_summary: str
    query: str
    commit_decision: str
    influence_summary: str


@dataclass(frozen=True, slots=True)
class SessionStoreConfig:
    default_root: str
    allow_external_session_roots: bool
    artifact_filenames: SessionArtifactConfig


@dataclass(frozen=True, slots=True)
class BayesLayerConfig:
    beliefs: BeliefCompilerConfig
    utility: UtilityConfig
    objective_surrogate: ObjectiveSurrogateConfig
    feasibility_surrogate: FeasibilitySurrogateConfig
    query_policy: QueryPolicyConfig
    commit: CommitPolicyConfig
    session_store: SessionStoreConfig
    integration_emit_schema_validated_eval_results: bool
    integration_preserve_genotypes: bool
    integration_aggregate_by_patch_hash: bool
    builtin_adapters: tuple[str, ...]
    interface_primary_surface: str
    default_belief_file_format: str
    default_user_profile: str
    interface_preview_required: bool
    stdout_format: str
    quiet_machine_readable_commands: bool
    graph_user_visible_by_default: bool
    graph_auto_derive_from_data: bool
    graph_allow_graph_directives: bool
    graph_default_screening_mode: str


DEFAULT_BAYES_LAYER_CONFIG_PATH = _repo_root() / "configs" / "default_bayes_layer.yaml"
DEFAULT_ADAPTER_CONFIG_PATH = _repo_root() / "configs" / "default_adapter.yaml"


def repo_relative_path(*parts: str) -> Path:
    return _repo_root().joinpath(*parts)


def load_yaml_document(path: Path) -> dict[str, JsonValue]:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValidationFailure(f"Expected YAML document at {path} to be a mapping.")
    return cast(dict[str, JsonValue], content)


def load_bayes_layer_config(path: Path | None = None) -> BayesLayerConfig:
    document = load_yaml_document(path or DEFAULT_BAYES_LAYER_CONFIG_PATH)

    beliefs_mapping = _require_mapping(document.get("beliefs"), "beliefs")
    prior_decay_mapping = _require_mapping(
        beliefs_mapping.get("prior_decay"), "prior_decay"
    )
    objective_mapping = _require_mapping(document.get("objective"), "objective")
    utility_mapping = _require_mapping(
        objective_mapping.get("utility"), "objective.utility"
    )
    objective_surrogate_mapping = _require_mapping(
        objective_mapping.get("surrogate"),
        "objective.surrogate",
    )
    feasibility_mapping = _require_mapping(document.get("feasibility"), "feasibility")
    feasibility_surrogate_mapping = _require_mapping(
        feasibility_mapping.get("surrogate"),
        "feasibility.surrogate",
    )
    query_policy_mapping = _require_mapping(
        document.get("query_policy"), "query_policy"
    )
    commit_mapping = _require_mapping(document.get("commit"), "commit")
    integration_mapping = _require_mapping(document.get("integration"), "integration")
    interface_mapping = _require_mapping(document.get("interface"), "interface")
    session_store_mapping = _require_mapping(
        document.get("session_store"), "session_store"
    )
    artifact_mapping = _require_mapping(
        session_store_mapping.get("artifact_filenames"),
        "session_store.artifact_filenames",
    )
    graph_mapping = _require_mapping(document.get("graph"), "graph")

    return BayesLayerConfig(
        beliefs=BeliefCompilerConfig(
            effect_strength_to_prior_mean=_mapping_to_int_float(
                _require_mapping(
                    beliefs_mapping.get("effect_strength_to_prior_mean"),
                    "beliefs.effect_strength_to_prior_mean",
                ),
                label="effect_strength_to_prior_mean",
            ),
            confidence_level_to_variance_scale=_mapping_to_int_float(
                _require_mapping(
                    beliefs_mapping.get("confidence_level_to_variance_scale"),
                    "beliefs.confidence_level_to_variance_scale",
                ),
                label="confidence_level_to_variance_scale",
            ),
            relation_strength_to_pair_mean=_mapping_to_int_float(
                _require_mapping(
                    beliefs_mapping.get("relation_strength_to_pair_mean"),
                    "beliefs.relation_strength_to_pair_mean",
                ),
                label="relation_strength_to_pair_mean",
            ),
            prior_decay=PriorDecayConfig(
                per_eval_multiplier=_require_float(
                    prior_decay_mapping,
                    "per_eval_multiplier",
                ),
                cross_era_main_effect_transfer=_require_float(
                    prior_decay_mapping,
                    "cross_era_main_effect_transfer",
                ),
                cross_era_pair_effect_transfer=_require_float(
                    prior_decay_mapping,
                    "cross_era_pair_effect_transfer",
                ),
            ),
            preview_required=_require_bool(beliefs_mapping, "preview_required"),
        ),
        utility=UtilityConfig(
            lambda_sparsity=_require_float(utility_mapping, "lambda_sparsity"),
            lambda_vram=_require_float(utility_mapping, "lambda_vram"),
            soft_vram_limit_mb=_require_float(utility_mapping, "soft_vram_limit_mb"),
        ),
        objective_surrogate=ObjectiveSurrogateConfig(
            include_active_gene_count=_require_bool(
                objective_surrogate_mapping,
                "include_active_gene_count",
            ),
            include_metadata_features=_require_bool(
                objective_surrogate_mapping,
                "include_metadata_features",
            ),
            max_pair_features=_require_int(
                objective_surrogate_mapping, "max_pair_features"
            ),
        ),
        feasibility_surrogate=FeasibilitySurrogateConfig(
            enable_failure_modes=_string_list(
                feasibility_surrogate_mapping,
                "enable_failure_modes",
            ),
        ),
        query_policy=QueryPolicyConfig(
            enabled=_require_bool(query_policy_mapping, "enabled"),
            max_queries_per_era=_require_int(
                query_policy_mapping, "max_queries_per_era"
            ),
            min_expected_value_of_information=_require_float(
                query_policy_mapping,
                "min_expected_value_of_information",
            ),
            allowed_query_types=_string_list(
                query_policy_mapping, "allowed_query_types"
            ),
        ),
        commit=CommitPolicyConfig(
            posterior_gain_probability_threshold=_require_float(
                commit_mapping,
                "posterior_gain_probability_threshold",
            ),
            min_valid_probability=_require_float(
                commit_mapping, "min_valid_probability"
            ),
            epsilon_commit_floor=_require_float(commit_mapping, "epsilon_commit_floor"),
        ),
        session_store=SessionStoreConfig(
            default_root=_require_str(session_store_mapping, "default_root"),
            allow_external_session_roots=_require_bool(
                session_store_mapping,
                "allow_external_session_roots",
            ),
            artifact_filenames=SessionArtifactConfig(
                manifest=_require_str(artifact_mapping, "manifest"),
                beliefs=_require_str(artifact_mapping, "beliefs"),
                surface_snapshot=_require_str(artifact_mapping, "surface_snapshot"),
                surface_overlay=_require_str(artifact_mapping, "surface_overlay"),
                canonicalization_summary=_require_str(
                    artifact_mapping, "canonicalization_summary"
                ),
                compiled_preview=_require_str(artifact_mapping, "compiled_preview"),
                compiled_priors=_require_str(artifact_mapping, "compiled_priors"),
                observations=_require_str(artifact_mapping, "observations"),
                posterior_summary=_require_str(artifact_mapping, "posterior_summary"),
                query=_require_str(artifact_mapping, "query"),
                commit_decision=_require_str(artifact_mapping, "commit_decision"),
                influence_summary=_require_str(artifact_mapping, "influence_summary"),
            ),
        ),
        integration_emit_schema_validated_eval_results=_require_bool(
            integration_mapping,
            "emit_schema_validated_eval_results",
        ),
        integration_preserve_genotypes=_require_bool(
            integration_mapping,
            "preserve_intended_and_realized_genotypes",
        ),
        integration_aggregate_by_patch_hash=_require_bool(
            integration_mapping,
            "aggregate_by_patch_hash",
        ),
        builtin_adapters=_string_list(integration_mapping, "builtin_adapters"),
        interface_primary_surface=_require_str(interface_mapping, "primary_surface"),
        default_belief_file_format=_require_str(
            interface_mapping,
            "default_belief_file_format",
        ),
        default_user_profile=_require_str(interface_mapping, "default_user_profile"),
        interface_preview_required=_require_bool(interface_mapping, "preview_required"),
        stdout_format=_require_str(interface_mapping, "stdout_format"),
        quiet_machine_readable_commands=_require_bool(
            interface_mapping,
            "quiet_machine_readable_commands",
        ),
        graph_user_visible_by_default=_require_bool(
            graph_mapping,
            "user_visible_by_default",
        ),
        graph_auto_derive_from_data=_require_bool(
            graph_mapping, "auto_derive_from_data"
        ),
        graph_allow_graph_directives=_require_bool(
            graph_mapping,
            "allow_graph_directives",
        ),
        graph_default_screening_mode=_require_str(
            graph_mapping,
            "default_screening_mode",
        ),
    )
