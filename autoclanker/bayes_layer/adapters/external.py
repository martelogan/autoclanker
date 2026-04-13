from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import subprocess
import sys

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from types import ModuleType
from typing import cast

from autoclanker.bayes_layer.adapters.protocols import (
    AdapterProbeResult,
    EvalLoopAdapter,
)
from autoclanker.bayes_layer.belief_io import (
    resolve_relative_path,
    validate_eval_result,
)
from autoclanker.bayes_layer.eval_contract import capture_eval_contract
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    EvalContractSnapshot,
    EvalExecutionContext,
    GeneStateRef,
    JsonValue,
    SemanticLevel,
    SurfaceKind,
    ValidAdapterConfig,
    ValidEvalResult,
    to_json_value,
)


def resolved_repo_path(config: ValidAdapterConfig) -> Path:
    if config.repo_path is None:
        raise AdapterFailure("Adapter config did not include repo_path.")
    base_dir = None if config.base_dir is None else Path(config.base_dir)
    return resolve_relative_path(config.repo_path, base_dir=base_dir).expanduser()


def _module_candidates(
    config: ValidAdapterConfig, default_candidates: Sequence[str]
) -> tuple[str, ...]:
    metadata = config.metadata or {}
    adapter_module = metadata.get("adapter_module")
    if isinstance(adapter_module, str) and adapter_module.strip():
        return (adapter_module.strip(),)
    return tuple(default_candidates)


@contextmanager
def _prepend_sys_path(path: Path) -> Iterator[None]:
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        with suppress(ValueError):  # pragma: no branch - defensive cleanup
            sys.path.remove(str(path))


def _load_module_from_path(module_name: str, repo_path: Path) -> ModuleType:
    module_relpath = Path(*module_name.split("."))
    file_candidate = repo_path / module_relpath.with_suffix(".py")
    package_candidate = repo_path / module_relpath / "__init__.py"
    if file_candidate.exists():
        spec = importlib.util.spec_from_file_location(
            f"autoclanker_repo_adapter_{module_name}_{abs(hash(file_candidate))}",
            file_candidate,
        )
        if spec is None or spec.loader is None:
            raise AdapterFailure(f"Could not load adapter module from {file_candidate}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if package_candidate.exists():
        spec = importlib.util.spec_from_file_location(
            f"autoclanker_repo_adapter_{module_name}_{abs(hash(package_candidate))}",
            package_candidate,
        )
        if spec is None or spec.loader is None:
            raise AdapterFailure(
                f"Could not load adapter package from {package_candidate}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    with _prepend_sys_path(repo_path):
        return importlib.import_module(module_name)


def _load_contract_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _validate_delegate(candidate: object, module_name: str) -> EvalLoopAdapter:
    required_methods = (
        "probe",
        "build_registry",
        "materialize_candidate",
        "evaluate_candidate",
        "commit_candidate",
    )
    missing = tuple(
        name
        for name in required_methods
        if not callable(getattr(candidate, name, None))
    )
    if missing:
        raise AdapterFailure(
            f"Module {module_name!r} exported an incomplete autoclanker adapter: missing {', '.join(missing)}."
        )
    if not hasattr(candidate, "kind") or not hasattr(candidate, "config"):
        raise AdapterFailure(
            f"Module {module_name!r} exported an incomplete autoclanker adapter: missing kind/config."
        )
    return cast(EvalLoopAdapter, candidate)


def _contract_delegate(
    module: ModuleType, config: ValidAdapterConfig
) -> EvalLoopAdapter:
    builder = cast(
        Callable[[ValidAdapterConfig], object] | None,
        getattr(module, "build_autoclanker_adapter", None),
    )
    if callable(builder):
        return _validate_delegate(builder(config), module.__name__)
    adapter_object = getattr(module, "AUTOCLANKER_ADAPTER", None)
    if adapter_object is not None:
        return _validate_delegate(adapter_object, module.__name__)
    raise AdapterFailure(
        f"Module {module.__name__!r} did not export build_autoclanker_adapter or AUTOCLANKER_ADAPTER."
    )


def _require_json_object(payload: object, *, message: str) -> dict[str, JsonValue]:
    if not isinstance(payload, dict):
        raise AdapterFailure(message)
    return cast(dict[str, JsonValue], payload)


def _require_string(value: object, *, message: str) -> str:
    if not isinstance(value, str):
        raise AdapterFailure(message)
    return value


def _supports_keyword(callable_obj: object, keyword: str) -> bool:
    if not callable(callable_obj):
        return False
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(keyword)
    if parameter is None:
        return False
    return parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }


class ModuleContractAdapter:
    def __init__(
        self,
        config: ValidAdapterConfig,
        *,
        kind: str,
        module: ModuleType,
        execution_mode: str,
        detail: str,
    ) -> None:
        self.kind = kind
        self.config = config
        self._module = module
        self._delegate: EvalLoopAdapter = _contract_delegate(module, config)
        self._execution_mode = execution_mode
        self._detail = detail

    def _probe_metadata(
        self, result: AdapterProbeResult | None = None
    ) -> dict[str, JsonValue]:
        metadata: dict[str, JsonValue] = {}
        if result is not None and result.metadata is not None:
            metadata.update(dict(result.metadata))
        metadata["execution_mode"] = self._execution_mode
        metadata["delegate_module"] = self._module.__name__
        return metadata

    def probe(self) -> AdapterProbeResult:
        delegate_probe = self._delegate.probe()
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=delegate_probe.available,
            detail=delegate_probe.detail or self._detail,
            session_root=self.config.session_root,
            metadata=self._probe_metadata(delegate_probe),
        )

    def build_registry(self) -> GeneRegistry:
        return self._delegate.build_registry()

    def capture_eval_contract(self) -> EvalContractSnapshot:
        capture_method = getattr(self._delegate, "capture_eval_contract", None)
        if callable(capture_method):
            return cast(EvalContractSnapshot, capture_method())
        return capture_eval_contract(self.config, kind=self.kind)

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        payload = dict(self._delegate.materialize_candidate(genotype))
        payload["adapter_kind"] = self.kind
        payload["execution_mode"] = self._execution_mode
        return payload

    def evaluate_candidate(
        self,
        *,
        era_id: str,
        candidate_id: str,
        genotype: Sequence[GeneStateRef],
        seed: int = 0,
        replication_index: int = 0,
        execution_context: EvalExecutionContext | None = None,
    ) -> ValidEvalResult:
        if execution_context is not None and _supports_keyword(
            getattr(self._delegate, "evaluate_candidate", None), "execution_context"
        ):
            result = self._delegate.evaluate_candidate(
                era_id=era_id,
                candidate_id=candidate_id,
                genotype=genotype,
                seed=seed,
                replication_index=replication_index,
                execution_context=execution_context,
            )
        else:
            result = self._delegate.evaluate_candidate(
                era_id=era_id,
                candidate_id=candidate_id,
                genotype=genotype,
                seed=seed,
                replication_index=replication_index,
            )
        return ValidEvalResult(
            era_id=result.era_id,
            candidate_id=result.candidate_id,
            intended_genotype=result.intended_genotype,
            realized_genotype=result.realized_genotype,
            patch_hash=result.patch_hash,
            status=result.status,
            seed=result.seed,
            runtime_sec=result.runtime_sec,
            peak_vram_mb=result.peak_vram_mb,
            raw_metrics={
                **result.raw_metrics,
                "adapter_kind": self.kind,
                "execution_mode": self._execution_mode,
                "workspace_root": (
                    None
                    if execution_context is None
                    else execution_context.workspace_root
                ),
            },
            delta_perf=result.delta_perf,
            utility=result.utility,
            replication_index=result.replication_index,
            stdout_digest=result.stdout_digest,
            stderr_digest=result.stderr_digest,
            artifact_paths=result.artifact_paths,
            failure_metadata=result.failure_metadata,
            eval_contract=result.eval_contract,
            execution_metadata=result.execution_metadata,
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        payload = dict(self._delegate.commit_candidate(candidate_id))
        payload["adapter_kind"] = self.kind
        payload["execution_mode"] = self._execution_mode
        return payload


class JsonSubprocessAdapter:
    def __init__(self, config: ValidAdapterConfig, *, kind: str) -> None:
        self.kind = kind
        self.config = config
        if not self.config.command:
            raise AdapterFailure("Subprocess adapters require a non-empty command.")

    def _run(self, request: Mapping[str, JsonValue]) -> JsonValue:
        completed = subprocess.run(
            self.config.command,
            input=json.dumps(request),
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise AdapterFailure(
                f"Subprocess adapter command failed with rc={completed.returncode}: {detail}"
            )
        try:
            return cast(JsonValue, json.loads(completed.stdout))
        except json.JSONDecodeError as exc:
            raise AdapterFailure(
                f"Subprocess adapter emitted invalid JSON: {exc}"
            ) from exc

    def _request(self, operation: str, **extra: object) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "operation": operation,
            "config": to_json_value(self.config),
        }
        for key, value in extra.items():
            payload[key] = to_json_value(value)
        return payload

    def probe(self) -> AdapterProbeResult:
        payload = _require_json_object(
            self._run(self._request("probe")),
            message="Subprocess probe must return an object.",
        )
        metadata_raw = payload.get("metadata")
        metadata = (
            None
            if not isinstance(metadata_raw, Mapping)
            else cast(Mapping[str, JsonValue], metadata_raw)
        )
        return AdapterProbeResult(
            kind=str(payload.get("kind", self.kind)),
            mode=str(payload.get("mode", self.config.mode)),
            available=bool(payload.get("available", False)),
            detail=str(payload.get("detail", "subprocess adapter probe completed")),
            session_root=str(payload.get("session_root", self.config.session_root)),
            metadata=metadata,
        )

    def build_registry(self) -> GeneRegistry:
        payload = _require_json_object(
            self._run(self._request("build_registry")),
            message="Subprocess build_registry must return an object.",
        )
        mapping: dict[str, tuple[str, ...]] = {}
        defaults: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        aliases: dict[str, tuple[str, ...]] = {}
        state_descriptions: dict[str, dict[str, str]] = {}
        state_aliases: dict[str, dict[str, tuple[str, ...]]] = {}
        surface_kinds: dict[str, SurfaceKind] = {}
        semantic_levels: dict[str, SemanticLevel] = {}
        materializable: dict[str, bool] = {}
        code_scopes: dict[str, tuple[str, ...]] = {}
        risk_hints: dict[str, tuple[str, ...]] = {}
        origins: dict[str, str] = {}
        metadata: dict[str, dict[str, JsonValue]] = {}
        for gene_id, raw_definition in payload.items():
            definition = _require_json_object(
                raw_definition,
                message="Registry response entries must be objects.",
            )
            raw_states = definition.get("states")
            if not isinstance(raw_states, list):
                raise AdapterFailure("Registry response requires a states list.")
            mapping[str(gene_id)] = tuple(str(item) for item in raw_states)
            defaults[str(gene_id)] = _require_string(
                definition.get("default_state"),
                message="Registry response requires a string default_state.",
            )
            description = definition.get("description")
            if isinstance(description, str) and description.strip():
                descriptions[str(gene_id)] = description.strip()
            raw_aliases = definition.get("aliases")
            if isinstance(raw_aliases, list) and all(
                isinstance(item, str) for item in raw_aliases
            ):
                aliases[str(gene_id)] = tuple(
                    item.strip()
                    for item in cast(list[str], raw_aliases)
                    if item.strip()
                )
            raw_state_descriptions = definition.get("state_descriptions")
            if isinstance(raw_state_descriptions, Mapping):
                state_descriptions[str(gene_id)] = {
                    str(state_id): str(state_description).strip()
                    for state_id, state_description in raw_state_descriptions.items()
                    if isinstance(state_description, str) and state_description.strip()
                }
            raw_state_aliases = definition.get("state_aliases")
            if isinstance(raw_state_aliases, Mapping):
                parsed_aliases: dict[str, tuple[str, ...]] = {}
                for state_id, raw_items in raw_state_aliases.items():
                    if not isinstance(raw_items, list) or any(
                        not isinstance(item, str) for item in raw_items
                    ):
                        continue
                    parsed_aliases[str(state_id)] = tuple(
                        item.strip()
                        for item in cast(list[str], raw_items)
                        if item.strip()
                    )
                state_aliases[str(gene_id)] = parsed_aliases
            surface_kind = definition.get("surface_kind")
            if isinstance(surface_kind, str) and surface_kind.strip():
                surface_kinds[str(gene_id)] = cast(SurfaceKind, surface_kind.strip())
            semantic_level = definition.get("semantic_level")
            if isinstance(semantic_level, str) and semantic_level.strip():
                semantic_levels[str(gene_id)] = cast(
                    SemanticLevel, semantic_level.strip()
                )
            materializable_value = definition.get("materializable")
            if isinstance(materializable_value, bool):
                materializable[str(gene_id)] = materializable_value
            raw_code_scopes = definition.get("code_scopes")
            if isinstance(raw_code_scopes, list) and all(
                isinstance(item, str) for item in raw_code_scopes
            ):
                code_scopes[str(gene_id)] = tuple(
                    item.strip()
                    for item in cast(list[str], raw_code_scopes)
                    if item.strip()
                )
            raw_risk_hints = definition.get("risk_hints")
            if isinstance(raw_risk_hints, list) and all(
                isinstance(item, str) for item in raw_risk_hints
            ):
                risk_hints[str(gene_id)] = tuple(
                    item.strip()
                    for item in cast(list[str], raw_risk_hints)
                    if item.strip()
                )
            origin = definition.get("origin")
            if isinstance(origin, str) and origin.strip():
                origins[str(gene_id)] = origin.strip()
            raw_metadata = definition.get("metadata")
            if isinstance(raw_metadata, Mapping):
                metadata[str(gene_id)] = {
                    str(key): cast(JsonValue, value)
                    for key, value in raw_metadata.items()
                }
        return GeneRegistry.from_mapping(
            mapping,
            defaults=defaults,
            descriptions=descriptions or None,
            aliases=aliases or None,
            state_descriptions=state_descriptions or None,
            state_aliases=state_aliases or None,
            surface_kinds=surface_kinds or None,
            semantic_levels=semantic_levels or None,
            materializable=materializable or None,
            code_scopes=code_scopes or None,
            risk_hints=risk_hints or None,
            origins=origins or None,
            metadata=metadata or None,
        )

    def capture_eval_contract(self) -> EvalContractSnapshot:
        try:
            payload = self._run(self._request("capture_eval_contract"))
        except AdapterFailure:
            return capture_eval_contract(self.config, kind=self.kind)
        if not isinstance(payload, Mapping):
            return capture_eval_contract(self.config, kind=self.kind)
        try:
            mapping = cast(Mapping[str, object], payload)
            return EvalContractSnapshot(
                contract_digest=_require_string(
                    mapping.get("contract_digest"),
                    message="capture_eval_contract requires contract_digest.",
                ),
                benchmark_tree_digest=_require_string(
                    mapping.get("benchmark_tree_digest"),
                    message="capture_eval_contract requires benchmark_tree_digest.",
                ),
                eval_harness_digest=_require_string(
                    mapping.get("eval_harness_digest"),
                    message="capture_eval_contract requires eval_harness_digest.",
                ),
                adapter_config_digest=_require_string(
                    mapping.get("adapter_config_digest"),
                    message="capture_eval_contract requires adapter_config_digest.",
                ),
                environment_digest=_require_string(
                    mapping.get("environment_digest"),
                    message="capture_eval_contract requires environment_digest.",
                ),
                workspace_snapshot_id=cast(
                    str | None, mapping.get("workspace_snapshot_id")
                ),
                workspace_snapshot_mode=cast(
                    str | None, mapping.get("workspace_snapshot_mode")
                ),
                captured_paths=cast(
                    dict[str, JsonValue] | None, mapping.get("captured_paths")
                ),
                captured_at=cast(str | None, mapping.get("captured_at")),
            )
        except AdapterFailure:
            return capture_eval_contract(self.config, kind=self.kind)

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        payload = _require_json_object(
            self._run(
                self._request(
                    "materialize_candidate",
                    genotype=tuple(genotype),
                )
            ),
            message="Subprocess materialize_candidate must return an object.",
        )
        return payload

    def evaluate_candidate(
        self,
        *,
        era_id: str,
        candidate_id: str,
        genotype: Sequence[GeneStateRef],
        seed: int = 0,
        replication_index: int = 0,
        execution_context: EvalExecutionContext | None = None,
    ) -> ValidEvalResult:
        payload = self._run(
            self._request(
                "evaluate_candidate",
                era_id=era_id,
                candidate_id=candidate_id,
                genotype=tuple(genotype),
                seed=seed,
                replication_index=replication_index,
                execution_context=execution_context,
            )
        )
        if not isinstance(payload, Mapping):
            raise AdapterFailure("Subprocess evaluate_candidate must return an object.")
        return validate_eval_result(cast(Mapping[str, object], payload))

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        payload = _require_json_object(
            self._run(self._request("commit_candidate", candidate_id=candidate_id)),
            message="Subprocess commit_candidate must return an object.",
        )
        return payload


def load_module_adapter(
    config: ValidAdapterConfig,
    *,
    kind: str,
    module_name: str,
    execution_mode: str,
    detail: str,
) -> ModuleContractAdapter:
    return ModuleContractAdapter(
        config,
        kind=kind,
        module=_load_contract_module(module_name),
        execution_mode=execution_mode,
        detail=detail,
    )


def load_repo_adapter(
    config: ValidAdapterConfig,
    *,
    kind: str,
    default_module_candidates: Sequence[str],
    detail: str,
) -> ModuleContractAdapter:
    repo_path = resolved_repo_path(config)
    last_error: Exception | None = None
    for module_name in _module_candidates(config, default_module_candidates):
        try:
            module = _load_module_from_path(module_name, repo_path)
            return ModuleContractAdapter(
                config,
                kind=kind,
                module=module,
                execution_mode=config.mode,
                detail=detail,
            )
        except Exception as exc:  # pragma: no cover - exercised through adapter errors
            last_error = exc
    if last_error is not None:
        raise AdapterFailure(str(last_error)) from last_error
    raise AdapterFailure(f"No adapter module candidates were available for {kind}.")


__all__ = [
    "JsonSubprocessAdapter",
    "load_module_adapter",
    "load_repo_adapter",
    "resolved_repo_path",
]
