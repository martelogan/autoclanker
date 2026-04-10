from __future__ import annotations

import importlib.util
import shutil

from collections.abc import Mapping, Sequence

from autoclanker.bayes_layer.adapters.external import (
    JsonSubprocessAdapter,
    load_module_adapter,
    load_repo_adapter,
    resolved_repo_path,
)
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.adapters.protocols import (
    AdapterProbeResult,
    EvalLoopAdapter,
)
from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)

_DEFAULT_MODULE_CANDIDATES = ("autoclanker_adapter", "autoresearch_autoclanker_adapter")


class AutoresearchAdapter:
    kind = "autoresearch"

    def __init__(
        self,
        config: ValidAdapterConfig,
        *,
        bayes_config: BayesLayerConfig | None = None,
    ) -> None:
        self.config = config
        self._bayes_config = bayes_config or load_bayes_layer_config()
        self._delegate_cache: tuple[EvalLoopAdapter, str, str] | None = None
        self._fixture = FixtureAdapter(
            ValidAdapterConfig(
                kind="fixture",
                mode="fixture",
                session_root=config.session_root,
                allow_missing=False,
            ),
            bayes_config=self._bayes_config,
        )

    def _resolve_real_delegate(self) -> tuple[EvalLoopAdapter, str, str]:
        if self.config.mode == "auto":
            resolution_errors: list[str] = []
            if self.config.repo_path is not None:
                path = resolved_repo_path(self.config)
                if path.exists():
                    return (
                        load_repo_adapter(
                            self.config,
                            kind=self.kind,
                            default_module_candidates=_DEFAULT_MODULE_CANDIDATES,
                            detail=f"Resolved local autoresearch repo at {path}",
                        ),
                        "local_repo_path",
                        f"Resolved local autoresearch repo at {path}",
                    )
                resolution_errors.append(f"Autoresearch repo path not found: {path}")
            if self.config.python_module is not None:
                if importlib.util.find_spec(self.config.python_module) is not None:
                    return (
                        load_module_adapter(
                            self.config,
                            kind=self.kind,
                            module_name=self.config.python_module,
                            execution_mode="installed_module",
                            detail=f"Resolved installed module {self.config.python_module}",
                        ),
                        "installed_module",
                        f"Resolved installed module {self.config.python_module}",
                    )
                resolution_errors.append(
                    f"Autoresearch module not importable: {self.config.python_module}"
                )
            if self.config.command:
                command = self.config.command[0]
                if shutil.which(command) is not None:
                    return (
                        JsonSubprocessAdapter(self.config, kind=self.kind),
                        "subprocess_cli",
                        f"Resolved subprocess command {command}",
                    )
                resolution_errors.append(f"Autoresearch command not found: {command}")
            if resolution_errors:
                raise AdapterFailure("; ".join(resolution_errors))
            raise AdapterFailure(
                "Autoresearch auto mode did not receive any usable integration hint. Provide repo_path, python_module, or command."
            )
        if self.config.mode == "local_repo_path":
            path = resolved_repo_path(self.config)
            if not path.exists():
                raise AdapterFailure(f"Autoresearch repo path not found: {path}")
            return (
                load_repo_adapter(
                    self.config,
                    kind=self.kind,
                    default_module_candidates=_DEFAULT_MODULE_CANDIDATES,
                    detail=f"Resolved local autoresearch repo at {path}",
                ),
                self.config.mode,
                f"Resolved local autoresearch repo at {path}",
            )
        if self.config.mode == "installed_module":
            if self.config.python_module is None:
                raise AdapterFailure(
                    "Autoresearch installed_module mode requires python_module."
                )
            if importlib.util.find_spec(self.config.python_module) is None:
                raise AdapterFailure(
                    f"Autoresearch module not importable: {self.config.python_module}"
                )
            return (
                load_module_adapter(
                    self.config,
                    kind=self.kind,
                    module_name=self.config.python_module,
                    execution_mode=self.config.mode,
                    detail=f"Resolved installed module {self.config.python_module}",
                ),
                self.config.mode,
                f"Resolved installed module {self.config.python_module}",
            )
        if self.config.mode == "subprocess_cli":
            command = self.config.command[0]
            if shutil.which(command) is None:
                raise AdapterFailure(f"Autoresearch command not found: {command}")
            return (
                JsonSubprocessAdapter(self.config, kind=self.kind),
                self.config.mode,
                f"Resolved subprocess command {command}",
            )
        raise AdapterFailure(f"Unsupported autoresearch mode: {self.config.mode}")

    def _delegate(self) -> tuple[EvalLoopAdapter, str, str]:
        if self._delegate_cache is not None:
            return self._delegate_cache
        try:
            resolved = self._resolve_real_delegate()
        except AdapterFailure as exc:
            if self.config.allow_missing:
                resolved = (
                    self._fixture,
                    "fixture_fallback",
                    str(exc),
                )
            else:
                raise
        self._delegate_cache = resolved
        return resolved

    def probe(self) -> AdapterProbeResult:
        try:
            delegate, execution_mode, detail = self._delegate()
        except AdapterFailure as exc:
            return AdapterProbeResult(
                kind=self.kind,
                mode=self.config.mode,
                available=False,
                detail=str(exc),
                session_root=self.config.session_root,
                metadata={"execution_mode": "unavailable"},
            )
        delegate_probe = delegate.probe()
        metadata = dict(delegate_probe.metadata or {})
        metadata["execution_mode"] = execution_mode
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=delegate_probe.available,
            detail=detail
            if execution_mode == "fixture_fallback"
            else delegate_probe.detail,
            session_root=self.config.session_root,
            metadata=metadata,
        )

    def build_registry(self) -> GeneRegistry:
        delegate, _execution_mode, _detail = self._delegate()
        return delegate.build_registry()

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        delegate, execution_mode, _detail = self._delegate()
        payload = dict(delegate.materialize_candidate(genotype))
        payload["adapter_kind"] = self.kind
        payload["execution_mode"] = execution_mode
        return payload

    def evaluate_candidate(
        self,
        *,
        era_id: str,
        candidate_id: str,
        genotype: Sequence[GeneStateRef],
        seed: int = 0,
        replication_index: int = 0,
    ) -> ValidEvalResult:
        delegate, _execution_mode, _detail = self._delegate()
        return delegate.evaluate_candidate(
            era_id=era_id,
            candidate_id=candidate_id,
            genotype=genotype,
            seed=seed,
            replication_index=replication_index,
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        delegate, execution_mode, _detail = self._delegate()
        payload = dict(delegate.commit_candidate(candidate_id))
        payload["adapter_kind"] = self.kind
        payload["execution_mode"] = execution_mode
        return payload
