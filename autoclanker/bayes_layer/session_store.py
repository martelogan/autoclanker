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
    JsonValue,
    PosteriorSummary,
    QuerySuggestion,
    SessionFailure,
    SessionManifest,
    SessionStatus,
    ValidatedBeliefBatch,
    ValidEvalResult,
    to_json_value,
)


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
    ) -> Path: ...

    def write_commit_decision(
        self, session_id: str, decision: CommitDecision
    ) -> Path: ...

    def write_influence_summary(
        self,
        session_id: str,
        *,
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
    ) -> Path:
        payload = {
            "ranked_candidates": cast(list[JsonValue], list(ranked_candidates)),
            "queries": cast(list[JsonValue], to_json_value(queries)),
        }
        return _json_dump(
            self._artifact_path(session_id, self._artifacts.query),
            payload,
        )

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

    def status(self, session_id: str) -> SessionStatus:
        session_path = self.session_path(session_id)
        manifest = self.load_manifest(session_id)
        observations = self.read_observations(session_id)
        artifact_paths = {
            "manifest": str(self._manifest_path(session_id)),
            "beliefs": str(self._artifact_path(session_id, self._artifacts.beliefs)),
            "surface_snapshot": str(
                self._artifact_path(session_id, self._artifacts.surface_snapshot)
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
        }
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
        )
