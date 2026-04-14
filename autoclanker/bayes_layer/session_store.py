from __future__ import annotations

import json

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import yaml

from autoclanker.bayes_layer.belief_io import (
    ingest_human_beliefs,
    load_serialized_payload,
    validate_eval_result,
)
from autoclanker.bayes_layer.config import (
    BayesLayerConfig,
    SessionArtifactConfig,
    load_bayes_layer_config,
)
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    BeliefsStatus,
    CanonicalizationMode,
    CommitDecision,
    EvalContractSnapshot,
    EvalMeasurementMode,
    EvalStabilizationMode,
    FrontierFamilyRepresentative,
    FrontierSummary,
    JsonValue,
    MergeSuggestion,
    PosteriorSummary,
    QuerySuggestion,
    QueryType,
    SessionFailure,
    SessionManifest,
    SessionStatus,
    ValidatedBeliefBatch,
    ValidEvalResult,
    to_json_value,
)


def _require_object_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SessionFailure(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)


def _require_object_list(
    value: object, *, label: str
) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        raise SessionFailure(f"{label} must be a list.")
    result: list[Mapping[str, object]] = []
    for index, item in enumerate(cast(list[object], value)):
        result.append(_require_object_mapping(item, label=f"{label}[{index}]"))
    return tuple(result)


def _require_string_list(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SessionFailure(f"{label} must be a list of strings.")
    raw_items = cast(list[object], value)
    if any(not isinstance(item, str) for item in raw_items):
        raise SessionFailure(f"{label} must be a list of strings.")
    return tuple(cast(list[str], raw_items))


def _require_number(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise SessionFailure(f"{label} must be numeric.")
    return float(value)


def _require_int(value: object, *, label: str) -> int:
    if not isinstance(value, int):
        raise SessionFailure(f"{label} must be an integer.")
    return value


def _require_query_type(value: object, *, label: str) -> QueryType:
    if value not in {
        "effect_sign",
        "risk_triage",
        "relation_check",
        "pairwise_preference",
    }:
        raise SessionFailure(
            f"{label} must be one of effect_sign, risk_triage, relation_check, pairwise_preference."
        )
    return cast(QueryType, value)


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    return trimmed or None


def _comparison_focus_from_query_mapping(
    query_mapping: Mapping[str, object],
) -> str | None:
    raw_candidate_ids = query_mapping.get("candidate_ids")
    if isinstance(raw_candidate_ids, list):
        candidate_ids = [
            item for item in cast(list[object], raw_candidate_ids) if isinstance(item, str)
        ]
        if len(candidate_ids) >= 2:
            return f"{candidate_ids[0]} vs {candidate_ids[1]}"
    raw_family_ids = query_mapping.get("family_ids")
    if isinstance(raw_family_ids, list):
        family_ids = [
            item for item in cast(list[object], raw_family_ids) if isinstance(item, str)
        ]
        if len(family_ids) >= 2:
            return f"{family_ids[0]} vs {family_ids[1]}"
    return None


class SessionStore(Protocol):
    @property
    def artifact_filenames(self) -> SessionArtifactConfig: ...

    def session_path(self, session_id: str) -> Path: ...

    def artifact_path(self, session_id: str, filename: str) -> Path: ...

    def init_session(self, manifest: SessionManifest) -> Path: ...

    def write_manifest(self, manifest: SessionManifest) -> Path: ...

    def load_manifest(self, session_id: str) -> SessionManifest: ...

    def write_beliefs(self, session_id: str, beliefs: ValidatedBeliefBatch) -> Path: ...

    def load_beliefs(self, session_id: str) -> ValidatedBeliefBatch: ...

    def write_surface_snapshot(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path: ...

    def write_eval_contract(
        self, session_id: str, contract: EvalContractSnapshot
    ) -> Path: ...

    def load_eval_contract(self, session_id: str) -> EvalContractSnapshot | None: ...

    def write_surface_overlay(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path: ...

    def load_surface_overlay(self, session_id: str) -> GeneRegistry | None: ...

    def write_canonicalization_summary(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path: ...

    def write_preview(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path: ...

    def write_compiled_priors(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path: ...

    def append_observation(
        self, session_id: str, observation: ValidEvalResult
    ) -> Path: ...

    def read_observations(self, session_id: str) -> tuple[ValidEvalResult, ...]: ...

    def write_posterior_summary(
        self, session_id: str, summary: PosteriorSummary
    ) -> Path: ...

    def write_query_artifact(
        self,
        session_id: str,
        *,
        ranked_candidates: tuple[Mapping[str, JsonValue], ...],
        queries: tuple[QuerySuggestion, ...],
        frontier_summary: FrontierSummary | None = None,
    ) -> Path: ...

    def write_frontier_status(
        self, session_id: str, frontier_summary: FrontierSummary
    ) -> Path: ...

    def load_frontier_status(self, session_id: str) -> FrontierSummary | None: ...

    def write_commit_decision(
        self, session_id: str, decision: CommitDecision
    ) -> Path: ...

    def write_influence_summary(
        self,
        session_id: str,
        *,
        payload: Mapping[str, JsonValue],
    ) -> Path: ...

    def write_eval_run_record(
        self,
        session_id: str,
        *,
        candidate_id: str,
        payload: Mapping[str, JsonValue],
    ) -> Path: ...

    def status(self, session_id: str) -> SessionStatus: ...


def _json_dump(path: Path, payload: Mapping[str, JsonValue]) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _yaml_dump(path: Path, payload: Mapping[str, JsonValue]) -> Path:
    path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False),
        encoding="utf-8",
    )
    return path


class FilesystemSessionStore:
    def __init__(
        self,
        *,
        root: str | Path | None = None,
        config: BayesLayerConfig | None = None,
    ) -> None:
        self._config = config or load_bayes_layer_config()
        configured_root = root or self._config.session_store.default_root
        self._root = Path(configured_root).expanduser()
        self._artifacts = self._config.session_store.artifact_filenames

    @property
    def root(self) -> Path:
        return self._root

    @property
    def artifact_filenames(self) -> SessionArtifactConfig:
        return self._artifacts

    def session_path(self, session_id: str) -> Path:
        return self.root / session_id

    def artifact_path(self, session_id: str, filename: str) -> Path:
        return self._artifact_path(session_id, filename)

    def _artifact_path(self, session_id: str, filename: str) -> Path:
        session_path = self.session_path(session_id)
        session_path.mkdir(parents=True, exist_ok=True)
        return session_path / filename

    def _manifest_path(self, session_id: str) -> Path:
        return self._artifact_path(session_id, self._artifacts.manifest)

    def _eval_runs_path(self, session_id: str, *, create: bool = True) -> Path:
        path = self.session_path(session_id) / self._artifacts.eval_runs_dir
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def init_session(self, manifest: SessionManifest) -> Path:
        session_path = self.session_path(manifest.session_id)
        session_path.mkdir(parents=True, exist_ok=True)
        self.write_manifest(manifest)
        return session_path

    def write_manifest(self, manifest: SessionManifest) -> Path:
        return _yaml_dump(
            self._manifest_path(manifest.session_id),
            cast(Mapping[str, JsonValue], to_json_value(manifest)),
        )

    def load_manifest(self, session_id: str) -> SessionManifest:
        path = self._manifest_path(session_id)
        if not path.exists():
            raise SessionFailure(
                f"Session manifest not found for session {session_id!r}."
            )
        raw = load_serialized_payload(path)
        return SessionManifest(
            session_id=str(raw["session_id"]),
            era_id=str(raw["era_id"]),
            adapter_kind=str(raw["adapter_kind"]),
            adapter_execution_mode=str(
                raw.get("adapter_execution_mode", raw["adapter_kind"])
            ),
            session_root=str(raw["session_root"]),
            created_at=str(raw["created_at"]),
            preview_required=bool(raw["preview_required"]),
            beliefs_status=cast(BeliefsStatus, raw.get("beliefs_status", "absent")),
            preview_digest=cast(str | None, raw.get("preview_digest")),
            compiled_priors_active=bool(raw.get("compiled_priors_active", False)),
            user_profile=cast(str | None, raw.get("user_profile")),
            canonicalization_mode=cast(
                CanonicalizationMode | None, raw.get("canonicalization_mode")
            ),
            surface_overlay_active=bool(raw.get("surface_overlay_active", False)),
            eval_contract_digest=cast(str | None, raw.get("eval_contract_digest")),
            eval_contract_required=bool(raw.get("eval_contract_required", False)),
            workspace_snapshot_mode=cast(
                str | None, raw.get("workspace_snapshot_mode")
            ),
        )

    def write_beliefs(self, session_id: str, beliefs: ValidatedBeliefBatch) -> Path:
        path = self._artifact_path(session_id, self._artifacts.beliefs)
        return _yaml_dump(path, beliefs.canonical_payload)

    def load_beliefs(self, session_id: str) -> ValidatedBeliefBatch:
        path = self._artifact_path(session_id, self._artifacts.beliefs)
        if not path.exists():
            raise SessionFailure(
                f"Belief artifact not found for session {session_id!r}."
            )
        return ingest_human_beliefs(load_serialized_payload(path))

    def write_surface_snapshot(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.surface_snapshot),
            payload,
        )

    def write_eval_contract(
        self, session_id: str, contract: EvalContractSnapshot
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.eval_contract),
            cast(Mapping[str, JsonValue], to_json_value(contract)),
        )

    def load_eval_contract(self, session_id: str) -> EvalContractSnapshot | None:
        path = self._artifact_path(session_id, self._artifacts.eval_contract)
        if not path.exists():
            return None
        raw = load_serialized_payload(path)
        return EvalContractSnapshot(
            contract_digest=str(raw["contract_digest"]),
            benchmark_tree_digest=str(raw["benchmark_tree_digest"]),
            eval_harness_digest=str(raw["eval_harness_digest"]),
            adapter_config_digest=str(raw["adapter_config_digest"]),
            environment_digest=str(raw["environment_digest"]),
            measurement_mode=cast(
                EvalMeasurementMode | None, raw.get("measurement_mode")
            ),
            stabilization_mode=cast(
                EvalStabilizationMode | None, raw.get("stabilization_mode")
            ),
            lease_scope=cast(str | None, raw.get("lease_scope")),
            workspace_snapshot_id=cast(str | None, raw.get("workspace_snapshot_id")),
            workspace_snapshot_mode=cast(
                str | None, raw.get("workspace_snapshot_mode")
            ),
            captured_paths=cast(dict[str, JsonValue] | None, raw.get("captured_paths")),
            captured_at=cast(str | None, raw.get("captured_at")),
        )

    def write_surface_overlay(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.surface_overlay),
            payload,
        )

    def load_surface_overlay(self, session_id: str) -> GeneRegistry | None:
        path = self._artifact_path(session_id, self._artifacts.surface_overlay)
        if not path.exists():
            return None
        raw = load_serialized_payload(path)
        registry_payload = raw.get("registry")
        if not isinstance(registry_payload, Mapping):
            raise SessionFailure(
                f"Surface overlay artifact for session {session_id!r} was invalid."
            )
        return GeneRegistry.from_serialized_dict(
            cast(Mapping[str, object], registry_payload)
        )

    def write_canonicalization_summary(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.canonicalization_summary),
            payload,
        )

    def write_preview(self, session_id: str, payload: Mapping[str, JsonValue]) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.compiled_preview),
            payload,
        )

    def write_compiled_priors(
        self, session_id: str, payload: Mapping[str, JsonValue]
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.compiled_priors),
            payload,
        )

    def append_observation(self, session_id: str, observation: ValidEvalResult) -> Path:
        path = self._artifact_path(session_id, self._artifacts.observations)
        serialized = json.dumps(to_json_value(observation), sort_keys=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
        return path

    def read_observations(self, session_id: str) -> tuple[ValidEvalResult, ...]:
        path = self._artifact_path(session_id, self._artifacts.observations)
        if not path.exists():
            return ()
        observations: list[ValidEvalResult] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise SessionFailure(
                    f"Invalid JSON line in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise SessionFailure(
                    f"Observation line {line_number} in {path} was not an object."
                )
            observations.append(validate_eval_result(cast(dict[str, object], payload)))
        return tuple(observations)

    def write_posterior_summary(
        self, session_id: str, summary: PosteriorSummary
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.posterior_summary),
            cast(Mapping[str, JsonValue], to_json_value(summary)),
        )

    def write_query_artifact(
        self,
        session_id: str,
        *,
        ranked_candidates: tuple[Mapping[str, JsonValue], ...],
        queries: tuple[QuerySuggestion, ...],
        frontier_summary: FrontierSummary | None = None,
    ) -> Path:
        first_ranked = ranked_candidates[0] if ranked_candidates else None
        first_query = queries[0] if queries else None
        payload: dict[str, JsonValue] = {
            "ranked_candidates": cast(JsonValue, list(ranked_candidates)),
            "queries": to_json_value(queries),
        }
        if first_ranked is not None:
            objective_backend = _optional_string(first_ranked.get("objective_backend"))
            acquisition_backend = _optional_string(
                first_ranked.get("acquisition_backend")
            )
            if objective_backend is not None:
                payload["objective_backend"] = objective_backend
            if acquisition_backend is not None:
                payload["acquisition_backend"] = acquisition_backend
        if first_query is not None:
            payload["follow_up_query_type"] = first_query.query_type
            comparison_focus = _comparison_focus_from_query_mapping(
                cast(Mapping[str, object], to_json_value(first_query))
            )
            if comparison_focus is not None:
                payload["follow_up_comparison"] = comparison_focus
        if frontier_summary is not None:
            payload["frontier_summary"] = to_json_value(frontier_summary)
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.query),
            payload,
        )

    def write_frontier_status(
        self, session_id: str, frontier_summary: FrontierSummary
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.frontier_status),
            cast(Mapping[str, JsonValue], to_json_value(frontier_summary)),
        )

    def load_frontier_status(self, session_id: str) -> FrontierSummary | None:
        path = self._artifact_path(session_id, self._artifacts.frontier_status)
        if not path.exists():
            return None
        raw = load_serialized_payload(path)
        family_representatives = tuple(
            FrontierFamilyRepresentative(
                family_id=str(item["family_id"]),
                representative_candidate_id=str(item["representative_candidate_id"]),
                representative_acquisition_score=_require_number(
                    item["representative_acquisition_score"],
                    label="family_representatives[].representative_acquisition_score",
                ),
                candidate_count=_require_int(
                    item["candidate_count"],
                    label="family_representatives[].candidate_count",
                ),
                compared_candidate_ids=_require_string_list(
                    item["compared_candidate_ids"],
                    label="family_representatives[].compared_candidate_ids",
                ),
                budget_weight=_require_number(
                    item["budget_weight"],
                    label="family_representatives[].budget_weight",
                ),
            )
            for item in _require_object_list(
                raw.get("family_representatives", []),
                label="family_representatives",
            )
        )
        pending_queries = tuple(
            QuerySuggestion(
                query_id=str(item["query_id"]),
                query_type=_require_query_type(
                    item["query_type"],
                    label="pending_queries[].query_type",
                ),
                prompt=str(item["prompt"]),
                target_refs=_require_string_list(
                    item["target_refs"],
                    label="pending_queries[].target_refs",
                ),
                expected_value=_require_number(
                    item["expected_value"],
                    label="pending_queries[].expected_value",
                ),
                confidence_gap=_require_number(
                    item["confidence_gap"],
                    label="pending_queries[].confidence_gap",
                ),
                candidate_ids=_require_string_list(
                    item.get("candidate_ids", []),
                    label="pending_queries[].candidate_ids",
                ),
                family_ids=_require_string_list(
                    item.get("family_ids", []),
                    label="pending_queries[].family_ids",
                ),
                comparison_scope=cast(str | None, item.get("comparison_scope")),
            )
            for item in _require_object_list(
                raw.get("pending_queries", []),
                label="pending_queries",
            )
        )
        pending_merge_suggestions = tuple(
            MergeSuggestion(
                merge_id=str(item["merge_id"]),
                family_ids=_require_string_list(
                    item["family_ids"],
                    label="pending_merge_suggestions[].family_ids",
                ),
                candidate_ids=_require_string_list(
                    item["candidate_ids"],
                    label="pending_merge_suggestions[].candidate_ids",
                ),
                rationale=str(item["rationale"]),
            )
            for item in _require_object_list(
                raw.get("pending_merge_suggestions", []),
                label="pending_merge_suggestions",
            )
        )
        dropped_family_reasons_raw = raw.get("dropped_family_reasons")
        if dropped_family_reasons_raw is not None:
            dropped_family_reasons_raw = _require_object_mapping(
                dropped_family_reasons_raw,
                label="dropped_family_reasons",
            )
        budget_allocations_raw = raw.get("budget_allocations")
        if budget_allocations_raw is not None:
            budget_allocations_raw = _require_object_mapping(
                budget_allocations_raw,
                label="budget_allocations",
            )
        return FrontierSummary(
            frontier_id=str(raw.get("frontier_id", "frontier_default")),
            candidate_count=_require_int(
                raw.get("candidate_count", 0),
                label="candidate_count",
            ),
            family_count=_require_int(
                raw.get("family_count", len(family_representatives)),
                label="family_count",
            ),
            family_representatives=family_representatives,
            dropped_family_reasons=(
                {}
                if dropped_family_reasons_raw is None
                else {
                    str(key): str(value)
                    for key, value in dropped_family_reasons_raw.items()
                }
            ),
            pending_queries=pending_queries,
            pending_merge_suggestions=pending_merge_suggestions,
            budget_allocations=(
                {}
                if budget_allocations_raw is None
                else {
                    str(key): _require_number(
                        value,
                        label=f"budget_allocations[{key}]",
                    )
                    for key, value in budget_allocations_raw.items()
                }
            ),
        )

    def _load_query_payload(self, session_id: str) -> Mapping[str, object]:
        path = self._artifact_path(session_id, self._artifacts.query)
        if not path.exists():
            return {}
        return load_serialized_payload(path)

    def write_commit_decision(self, session_id: str, decision: CommitDecision) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.commit_decision),
            cast(Mapping[str, JsonValue], to_json_value(decision)),
        )

    def write_influence_summary(
        self,
        session_id: str,
        *,
        payload: Mapping[str, JsonValue],
    ) -> Path:
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.influence_summary),
            payload,
        )

    def write_eval_run_record(
        self,
        session_id: str,
        *,
        candidate_id: str,
        payload: Mapping[str, JsonValue],
    ) -> Path:
        return _json_dump(
            self._eval_runs_path(session_id) / f"{candidate_id}.json",
            payload,
        )

    def status(self, session_id: str) -> SessionStatus:
        session_path = self.session_path(session_id)
        manifest = self.load_manifest(session_id)
        observations = self.read_observations(session_id)
        frontier_status = self.load_frontier_status(session_id)
        last_execution = (
            None
            if not observations or observations[-1].execution_metadata is None
            else observations[-1].execution_metadata
        )
        artifact_paths = {
            "manifest": str(self._manifest_path(session_id)),
            "beliefs": str(self._artifact_path(session_id, self._artifacts.beliefs)),
            "surface_snapshot": str(
                self._artifact_path(session_id, self._artifacts.surface_snapshot)
            ),
            "eval_contract": str(
                self._artifact_path(session_id, self._artifacts.eval_contract)
            ),
            "surface_overlay": str(
                self._artifact_path(session_id, self._artifacts.surface_overlay)
            ),
            "canonicalization_summary": str(
                self._artifact_path(
                    session_id, self._artifacts.canonicalization_summary
                )
            ),
            "compiled_preview": str(
                self._artifact_path(session_id, self._artifacts.compiled_preview)
            ),
            "compiled_priors": str(
                self._artifact_path(session_id, self._artifacts.compiled_priors)
            ),
            "observations": str(
                self._artifact_path(session_id, self._artifacts.observations)
            ),
            "posterior_summary": str(
                self._artifact_path(session_id, self._artifacts.posterior_summary)
            ),
            "query": str(self._artifact_path(session_id, self._artifacts.query)),
            "frontier_status": str(
                self._artifact_path(session_id, self._artifacts.frontier_status)
            ),
            "commit_decision": str(
                self._artifact_path(session_id, self._artifacts.commit_decision)
            ),
            "influence_summary": str(
                self._artifact_path(session_id, self._artifacts.influence_summary)
            ),
            "results_markdown": str(
                self._artifact_path(session_id, self._artifacts.results_markdown)
            ),
            "convergence_plot": str(
                self._artifact_path(session_id, self._artifacts.convergence_plot)
            ),
            "candidate_rankings_plot": str(
                self._artifact_path(session_id, self._artifacts.candidate_rankings_plot)
            ),
            "prior_graph_plot": str(
                self._artifact_path(session_id, self._artifacts.prior_graph_plot)
            ),
            "posterior_graph_plot": str(
                self._artifact_path(session_id, self._artifacts.posterior_graph_plot)
            ),
            "eval_runs": str(self._eval_runs_path(session_id, create=False)),
        }
        posterior_summary_path = self._artifact_path(
            session_id, self._artifacts.posterior_summary
        )
        posterior_summary = (
            {}
            if not posterior_summary_path.exists()
            else load_serialized_payload(posterior_summary_path)
        )
        query_payload = self._load_query_payload(session_id)
        ranked_candidates_raw = query_payload.get("ranked_candidates", [])
        first_ranked_candidate = (
            None
            if not isinstance(ranked_candidates_raw, list) or not ranked_candidates_raw
            else (
                cast(dict[str, object], ranked_candidates_raw[0])
                if isinstance(ranked_candidates_raw[0], dict)
                else None
            )
        )
        queries_raw = query_payload.get("queries", [])
        first_query = (
            None
            if not isinstance(queries_raw, list) or not queries_raw
            else (
                cast(dict[str, object], queries_raw[0])
                if isinstance(queries_raw[0], dict)
                else None
            )
        )
        return SessionStatus(
            session_id=session_id,
            era_id=manifest.era_id,
            session_path=str(session_path),
            observation_count=len(observations),
            artifact_paths=artifact_paths,
            beliefs_status=manifest.beliefs_status,
            preview_digest=manifest.preview_digest,
            compiled_priors_active=manifest.compiled_priors_active,
            adapter_execution_mode=manifest.adapter_execution_mode,
            ready_for_fit=manifest.beliefs_status != "preview_pending",
            ready_for_commit_recommendation=(
                manifest.beliefs_status != "preview_pending" and len(observations) > 0
            ),
            canonicalization_mode=manifest.canonicalization_mode,
            surface_overlay_active=manifest.surface_overlay_active,
            eval_contract_digest=manifest.eval_contract_digest,
            eval_contract_required=manifest.eval_contract_required,
            last_eval_measurement_mode=(
                None if last_execution is None else last_execution.measurement_mode
            ),
            last_eval_stabilization_mode=(
                None if last_execution is None else last_execution.stabilization_mode
            ),
            last_eval_used_lease=(
                None if last_execution is None else last_execution.lease_acquired
            ),
            last_eval_noisy_system=(
                None if last_execution is None else last_execution.noisy_system
            ),
            frontier_family_count=(
                0 if frontier_status is None else frontier_status.family_count
            ),
            frontier_candidate_count=(
                0 if frontier_status is None else frontier_status.candidate_count
            ),
            pending_query_count=(
                0 if frontier_status is None else len(frontier_status.pending_queries)
            ),
            pending_merge_suggestion_count=(
                0
                if frontier_status is None
                else len(frontier_status.pending_merge_suggestions)
            ),
            last_objective_backend=(
                _optional_string(query_payload.get("objective_backend"))
                or (
                    None
                    if first_ranked_candidate is None
                    else _optional_string(first_ranked_candidate.get("objective_backend"))
                )
                or _optional_string(posterior_summary.get("objective_backend"))
            ),
            last_acquisition_backend=(
                _optional_string(query_payload.get("acquisition_backend"))
                or (
                    None
                    if first_ranked_candidate is None
                    else _optional_string(
                        first_ranked_candidate.get("acquisition_backend")
                    )
                )
                or _optional_string(posterior_summary.get("acquisition_backend"))
            ),
            last_follow_up_query_type=(
                _optional_string(query_payload.get("follow_up_query_type"))
                or (
                    None
                    if first_query is None
                    else _optional_string(first_query.get("query_type"))
                )
            ),
            last_follow_up_comparison=(
                _optional_string(query_payload.get("follow_up_comparison"))
                or (
                    None
                    if first_query is None
                    else _comparison_focus_from_query_mapping(first_query)
                )
            ),
        )
