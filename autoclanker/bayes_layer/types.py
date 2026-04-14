from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonMapping: TypeAlias = Mapping[str, JsonValue]

BeliefKind: TypeAlias = Literal[
    "proposal",
    "idea",
    "relation",
    "preference",
    "constraint",
    "expert_prior",
    "graph_directive",
]
CompileStatus: TypeAlias = Literal["compiled", "metadata_only", "rejected"]
UserProfile: TypeAlias = Literal["basic", "expert"]
BeliefsStatus: TypeAlias = Literal["absent", "preview_pending", "applied"]
AdapterKind: TypeAlias = Literal[
    "fixture",
    "autoresearch",
    "cevolve",
    "python_module",
    "subprocess",
]
AdapterMode: TypeAlias = Literal[
    "fixture",
    "auto",
    "local_repo_path",
    "installed_module",
    "subprocess_cli",
]
RelationType: TypeAlias = Literal["synergy", "conflict", "dependency", "exclusion"]
ConstraintType: TypeAlias = Literal[
    "hard_exclude",
    "soft_avoid",
    "require",
    "budget_cap",
]
GraphDirectiveType: TypeAlias = Literal[
    "screen_include",
    "screen_exclude",
    "linkage_positive",
    "linkage_negative",
]
QueryType: TypeAlias = Literal[
    "effect_sign",
    "risk_triage",
    "relation_check",
    "pairwise_preference",
]
SurfaceKind: TypeAlias = Literal[
    "runtime_option",
    "mutation_family",
    "search_angle",
    "risk_family",
    "constraint_family",
]
SemanticLevel: TypeAlias = Literal[
    "concrete",
    "strategy",
    "risk",
    "constraint",
]
CanonicalizationSource: TypeAlias = Literal["deterministic", "llm", "hybrid"]
CanonicalizationMode: TypeAlias = Literal["deterministic", "hybrid", "llm"]
EvalPolicyMode: TypeAlias = Literal["auto", "exclusive", "parallel_ok"]
EvalMeasurementMode: TypeAlias = Literal["exclusive", "parallel_ok"]
EvalStabilizationMode: TypeAlias = Literal["off", "soft"]
PreferenceDirection: TypeAlias = Literal["left", "right", "tie"]
IsolationMode: TypeAlias = Literal["git_worktree", "copy", "fixture"]
FrontierOriginKind: TypeAlias = Literal[
    "legacy_pool",
    "manual",
    "belief",
    "query",
    "merge",
    "seed",
]
EvalDriftStatus: TypeAlias = Literal["locked", "drifted", "unverified"]
FailureMode: TypeAlias = Literal[
    "valid_run",
    "compile_fail",
    "runtime_fail",
    "oom",
    "timeout",
    "metric_instability",
]
EvalStatus: TypeAlias = Literal[
    "valid", "compile_fail", "runtime_fail", "oom", "timeout"
]


class AutoclankerError(Exception):
    """Base exception for repo-native autoclanker failures."""


class ValidationFailure(AutoclankerError):
    """Raised when a payload fails schema or semantic validation."""


class SessionFailure(AutoclankerError):
    """Raised when a session artifact is missing or inconsistent."""


class AdapterFailure(AutoclankerError):
    """Raised when an adapter cannot satisfy a requested action."""


def _stringify_key(value: object) -> str:
    return str(value)


def to_json_value(value: object) -> JsonValue:
    """Convert dataclasses and tuples into JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if is_dataclass(value):
        result: dict[str, JsonValue] = {}
        for field in fields(value):
            item = to_json_value(getattr(value, field.name))
            if item is None:
                continue
            result[field.name] = item
        return result

    if isinstance(value, Mapping):
        result = {}
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            converted = to_json_value(item)
            if converted is None:
                continue
            result[_stringify_key(key)] = converted
        return result

    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        sequence = cast(Sequence[Any], value)
        return [to_json_value(item) for item in sequence]

    raise TypeError(f"Unsupported JSON conversion for value: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class GeneStateRef:
    gene_id: str
    state_id: str

    @property
    def canonical_key(self) -> str:
        return f"{self.gene_id}:{self.state_id}"


@dataclass(frozen=True, slots=True)
class RiskVector:
    compile_fail: int | None = None
    runtime_fail: int | None = None
    oom: int | None = None
    timeout: int | None = None
    metric_instability: int | None = None

    def nonzero_items(self) -> tuple[tuple[str, int], ...]:
        items: list[tuple[str, int]] = []
        for name in (
            "compile_fail",
            "runtime_fail",
            "oom",
            "timeout",
            "metric_instability",
        ):
            value = cast(int | None, getattr(self, name))
            if value:
                items.append((name, value))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class BeliefContext:
    hardware_profile_id: str | None = None
    budget_profile_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class SessionContext:
    era_id: str
    session_id: str | None = None
    hardware_profile_id: str | None = None
    budget_profile_id: str | None = None
    author: str | None = None
    user_profile: UserProfile | None = None


@dataclass(frozen=True, slots=True)
class CandidatePattern:
    members: tuple[GeneStateRef, ...]


@dataclass(frozen=True, slots=True)
class DecayOverride:
    per_eval_multiplier: float | None = None
    cross_era_transfer: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BeliefBase:
    id: str
    confidence_level: int
    evidence_sources: tuple[str, ...] = ()
    context: BeliefContext | None = None
    rationale: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ProposalBelief(BeliefBase):
    kind: Literal["proposal"] = "proposal"
    proposal_text: str
    suggested_scope: str | None = None
    risk_hints: RiskVector | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class IdeaBelief(BeliefBase):
    kind: Literal["idea"] = "idea"
    gene: GeneStateRef
    effect_strength: int
    risk: RiskVector | None = None
    complexity_delta: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class RelationBelief(BeliefBase):
    kind: Literal["relation"] = "relation"
    members: tuple[GeneStateRef, ...]
    relation: RelationType
    strength: int
    joint_effect_strength: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class PreferenceBelief(BeliefBase):
    kind: Literal["preference"] = "preference"
    left_pattern: CandidatePattern
    right_pattern: CandidatePattern
    preference: PreferenceDirection
    strength: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ConstraintBelief(BeliefBase):
    kind: Literal["constraint"] = "constraint"
    constraint_type: ConstraintType
    severity: int
    scope: tuple[GeneStateRef, ...]


@dataclass(frozen=True, slots=True)
class MainEffectTarget:
    target_kind: Literal["main_effect"]
    gene: GeneStateRef


@dataclass(frozen=True, slots=True)
class PairEffectTarget:
    target_kind: Literal["pair_effect"]
    members: tuple[GeneStateRef, GeneStateRef]


@dataclass(frozen=True, slots=True)
class FeasibilityTarget:
    target_kind: Literal["feasibility_logit"]
    gene: GeneStateRef
    failure_mode: FailureMode


@dataclass(frozen=True, slots=True)
class VramTarget:
    target_kind: Literal["vram_effect"]
    gene: GeneStateRef


ExpertTarget: TypeAlias = (
    MainEffectTarget | PairEffectTarget | FeasibilityTarget | VramTarget
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExpertPriorBelief(BeliefBase):
    kind: Literal["expert_prior"] = "expert_prior"
    target: ExpertTarget
    prior_family: Literal["normal", "logit_normal"]
    mean: float
    scale: float
    observation_weight: float | None = None
    decay_override: DecayOverride | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphDirectiveBelief(BeliefBase):
    kind: Literal["graph_directive"] = "graph_directive"
    members: tuple[GeneStateRef, GeneStateRef]
    directive: GraphDirectiveType
    strength: int


Belief: TypeAlias = (
    ProposalBelief
    | IdeaBelief
    | RelationBelief
    | PreferenceBelief
    | ConstraintBelief
    | ExpertPriorBelief
    | GraphDirectiveBelief
)


@dataclass(frozen=True, slots=True)
class ValidatedBeliefBatch:
    session_context: SessionContext
    beliefs: tuple[Belief, ...]
    canonical_payload: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class EraState:
    era_id: str
    observation_count: int = 0


@dataclass(frozen=True, slots=True)
class PriorDecay:
    per_eval_multiplier: float
    cross_era_transfer: float


@dataclass(frozen=True, slots=True)
class CompiledPriorItem:
    target_kind: str
    target_ref: str
    prior_family: str
    mean: float
    scale: float
    observation_weight: float | None = None
    decay: PriorDecay | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BeliefPreview:
    belief_id: str
    compile_status: CompileStatus
    compiled_items: tuple[CompiledPriorItem, ...]
    warnings: tuple[str, ...] = ()
    influence_summary: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PriorSpec:
    source_belief_id: str
    item: CompiledPriorItem


@dataclass(frozen=True, slots=True)
class CompiledPriorPreview:
    era_id: str
    belief_previews: tuple[BeliefPreview, ...]


@dataclass(frozen=True, slots=True)
class CompiledPriorBundle:
    era_id: str
    main_effect_priors: tuple[PriorSpec, ...]
    pair_priors: tuple[PriorSpec, ...]
    feasibility_priors: tuple[PriorSpec, ...]
    vram_priors: tuple[PriorSpec, ...]
    hard_masks: tuple[PriorSpec, ...]
    preference_observations: tuple[PriorSpec, ...]
    candidate_generation_hints: tuple[PriorSpec, ...]
    linkage_hints: tuple[PriorSpec, ...]
    belief_previews: tuple[BeliefPreview, ...]

    @property
    def all_items(self) -> tuple[PriorSpec, ...]:
        return (
            self.main_effect_priors
            + self.pair_priors
            + self.feasibility_priors
            + self.vram_priors
            + self.hard_masks
            + self.preference_observations
            + self.candidate_generation_hints
            + self.linkage_hints
        )


@dataclass(frozen=True, slots=True)
class EvalContractSnapshot:
    contract_digest: str
    benchmark_tree_digest: str
    eval_harness_digest: str
    adapter_config_digest: str
    environment_digest: str
    measurement_mode: EvalMeasurementMode | None = None
    stabilization_mode: EvalStabilizationMode | None = None
    lease_scope: str | None = None
    workspace_snapshot_id: str | None = None
    workspace_snapshot_mode: str | None = None
    captured_paths: dict[str, JsonValue] | None = None
    captured_at: str | None = None


@dataclass(frozen=True, slots=True)
class EvalExecutionMetadata:
    isolation_mode: IsolationMode
    workspace_root: str | None = None
    workspace_snapshot_id: str | None = None
    contract_digest: str | None = None
    measurement_mode: EvalMeasurementMode | None = None
    stabilization_mode: EvalStabilizationMode | None = None
    lease_scope: str | None = None
    lease_acquired: bool | None = None
    lease_wait_sec: float | None = None
    noisy_system: bool | None = None
    loadavg_1m_before: float | None = None
    loadavg_1m_after: float | None = None
    stabilization_delay_sec: float | None = None


@dataclass(frozen=True, slots=True)
class EvalExecutionContext:
    session_id: str
    era_id: str
    contract: EvalContractSnapshot
    isolation_mode: IsolationMode
    workspace_root: str | None = None
    seed: int = 0
    replication_index: int = 0
    measurement_mode: EvalMeasurementMode | None = None
    stabilization_mode: EvalStabilizationMode | None = None
    lease_scope: str | None = None


@dataclass(frozen=True, slots=True)
class ValidEvalResult:
    era_id: str
    candidate_id: str
    intended_genotype: tuple[GeneStateRef, ...]
    realized_genotype: tuple[GeneStateRef, ...]
    patch_hash: str
    status: EvalStatus
    seed: int
    runtime_sec: float
    peak_vram_mb: float
    raw_metrics: dict[str, JsonValue]
    delta_perf: float
    utility: float
    replication_index: int
    stdout_digest: str | None = None
    stderr_digest: str | None = None
    artifact_paths: tuple[str, ...] = ()
    failure_metadata: dict[str, JsonValue] | None = None
    eval_contract: EvalContractSnapshot | None = None
    execution_metadata: EvalExecutionMetadata | None = None


@dataclass(frozen=True, slots=True)
class ValidAdapterConfig:
    kind: AdapterKind
    mode: AdapterMode
    session_root: str
    allow_missing: bool = False
    repo_path: str | None = None
    python_module: str | None = None
    command: tuple[str, ...] = ()
    config_path: str | None = None
    metadata: dict[str, JsonValue] | None = None
    base_dir: str | None = None


@dataclass(frozen=True, slots=True)
class EncodedCandidate:
    candidate_id: str
    genotype: tuple[GeneStateRef, ...]
    main_effects: tuple[str, ...]
    pair_effects: tuple[str, ...]
    global_features: dict[str, float]


@dataclass(frozen=True, slots=True)
class AggregatedObservation:
    patch_hash: str
    candidate_ids: tuple[str, ...]
    realized_genotype: tuple[GeneStateRef, ...]
    utility_mean: float
    utility_variance: float
    delta_perf_mean: float
    valid_rate: float
    runtime_sec_mean: float
    peak_vram_mb_mean: float
    count: int
    status_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class FrontierCandidate:
    candidate_id: str
    genotype: tuple[GeneStateRef, ...]
    family_id: str = "family_default"
    origin_kind: FrontierOriginKind = "legacy_pool"
    parent_candidate_ids: tuple[str, ...] = ()
    parent_belief_ids: tuple[str, ...] = ()
    origin_query_ids: tuple[str, ...] = ()
    notes: str | None = None
    budget_weight: float | None = None


@dataclass(frozen=True, slots=True)
class FrontierDocument:
    candidates: tuple[FrontierCandidate, ...]
    frontier_id: str = "frontier_default"
    default_family_id: str = "family_default"


@dataclass(frozen=True, slots=True)
class PosteriorFeature:
    feature_name: str
    target_kind: str
    posterior_mean: float
    posterior_variance: float
    support: int
    prior_mean: float
    prior_variance: float | None = None


@dataclass(frozen=True, slots=True)
class ObjectivePosterior:
    era_id: str
    baseline_utility: float
    features: tuple[PosteriorFeature, ...]
    observation_count: int
    backend: str = "heuristic_independent_normal"
    sampleable: bool = False
    feature_order: tuple[str, ...] = ()
    posterior_mean_vector: tuple[float, ...] = ()
    posterior_covariance: tuple[tuple[float, ...], ...] = ()
    aggregate_count: int = 0
    effective_observation_count: float = 0.0
    condition_number: float | None = None
    used_jitter: float = 0.0
    observation_noise: float | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FeasibilityPosterior:
    era_id: str
    baseline_valid_probability: float
    features: tuple[PosteriorFeature, ...]
    failure_mode_biases: dict[str, float]
    observation_count: int


@dataclass(frozen=True, slots=True)
class FrontierFamilyRepresentative:
    family_id: str
    representative_candidate_id: str
    representative_acquisition_score: float
    candidate_count: int
    compared_candidate_ids: tuple[str, ...]
    budget_weight: float


@dataclass(frozen=True, slots=True)
class MergeSuggestion:
    merge_id: str
    family_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    rationale: str


@dataclass(frozen=True, slots=True)
class FrontierSummary:
    frontier_id: str
    candidate_count: int
    family_count: int
    family_representatives: tuple[FrontierFamilyRepresentative, ...]
    dropped_family_reasons: dict[str, str]
    pending_queries: tuple[QuerySuggestion, ...]
    pending_merge_suggestions: tuple[MergeSuggestion, ...]
    budget_allocations: dict[str, float]


@dataclass(frozen=True, slots=True)
class PosteriorGraphEdge:
    source: str
    target: str
    weight: float
    relation: str


@dataclass(frozen=True, slots=True)
class PosteriorGraph:
    nodes: tuple[str, ...]
    edges: tuple[PosteriorGraphEdge, ...]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate_id: str
    genotype: tuple[GeneStateRef, ...]
    predicted_utility: float
    uncertainty: float
    valid_probability: float
    acquisition_score: float
    objective_backend: str | None = None
    acquisition_backend: str = "optimistic_upper_confidence"
    acquisition_fallback_reason: str | None = None
    sampled_utility: float | None = None
    sampled_score_mean: float | None = None
    sampled_score_std: float | None = None
    posterior_win_rate: float | None = None
    rationale: tuple[str, ...] = ()
    influence_summary: tuple[str, ...] = ()
    family_id: str | None = None
    origin_kind: FrontierOriginKind | None = None
    parent_candidate_ids: tuple[str, ...] = ()
    parent_belief_ids: tuple[str, ...] = ()
    origin_query_ids: tuple[str, ...] = ()
    notes: str | None = None
    budget_weight: float | None = None


@dataclass(frozen=True, slots=True)
class QuerySuggestion:
    query_id: str
    query_type: QueryType
    prompt: str
    target_refs: tuple[str, ...]
    expected_value: float
    confidence_gap: float
    candidate_ids: tuple[str, ...] = ()
    family_ids: tuple[str, ...] = ()
    comparison_scope: str | None = None


@dataclass(frozen=True, slots=True)
class InfluenceSummary:
    source_belief_id: str
    target_ref: str
    summary: str


@dataclass(frozen=True, slots=True)
class PosteriorSummary:
    era_id: str
    observation_count: int
    aggregate_count: int
    objective_baseline: float
    valid_baseline: float
    top_features: tuple[PosteriorFeature, ...]
    top_candidates: tuple[RankedCandidate, ...]
    graph: PosteriorGraph
    objective_backend: str = "heuristic_independent_normal"
    acquisition_backend: str = "optimistic_upper_confidence"
    acquisition_fallback_reason: str | None = None
    objective_sampleable: bool = False
    objective_effective_observation_count: float = 0.0
    objective_feature_count: int = 0
    objective_main_feature_count: int = 0
    objective_pair_feature_count: int = 0
    objective_condition_number: float | None = None
    objective_used_jitter: float = 0.0
    objective_observation_noise: float | None = None
    objective_fallback_reason: str | None = None
    fit_runtime_ms: float | None = None
    suggest_runtime_ms: float | None = None
    influence_summary: tuple[InfluenceSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class CommitDecision:
    era_id: str
    session_id: str
    recommended: bool
    candidate_id: str | None
    predicted_gain: float
    gain_probability: float
    valid_probability: float
    reason: str
    thresholds: dict[str, float]
    influence_summary: tuple[InfluenceSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionManifest:
    session_id: str
    era_id: str
    adapter_kind: str
    adapter_execution_mode: str
    session_root: str
    created_at: str
    preview_required: bool
    beliefs_status: BeliefsStatus = "absent"
    preview_digest: str | None = None
    compiled_priors_active: bool = False
    user_profile: str | None = None
    canonicalization_mode: CanonicalizationMode | None = None
    surface_overlay_active: bool = False
    eval_contract_digest: str | None = None
    eval_contract_required: bool = False
    workspace_snapshot_mode: str | None = None


@dataclass(frozen=True, slots=True)
class SessionStatus:
    session_id: str
    era_id: str
    session_path: str
    observation_count: int
    artifact_paths: dict[str, str]
    beliefs_status: BeliefsStatus
    preview_digest: str | None
    compiled_priors_active: bool
    adapter_execution_mode: str
    ready_for_fit: bool
    ready_for_commit_recommendation: bool
    canonicalization_mode: CanonicalizationMode | None = None
    surface_overlay_active: bool = False
    eval_contract_digest: str | None = None
    eval_contract_required: bool = False
    current_eval_contract_digest: str | None = None
    eval_contract_matches_current: bool | None = None
    eval_contract_drift_status: EvalDriftStatus | None = None
    last_eval_measurement_mode: EvalMeasurementMode | None = None
    last_eval_stabilization_mode: EvalStabilizationMode | None = None
    last_eval_used_lease: bool | None = None
    last_eval_noisy_system: bool | None = None
    frontier_family_count: int = 0
    frontier_candidate_count: int = 0
    pending_query_count: int = 0
    pending_merge_suggestion_count: int = 0
    last_objective_backend: str | None = None
    last_acquisition_backend: str | None = None
    last_follow_up_query_type: str | None = None
    last_follow_up_comparison: str | None = None
