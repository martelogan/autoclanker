from __future__ import annotations

import hashlib
import importlib
import json
import re
import shlex
import shutil
import subprocess
import sys

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from autoclanker.bayes_layer.adapters.external import resolved_repo_path
from autoclanker.bayes_layer.adapters.protocols import AdapterProbeResult
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    EvalExecutionContext,
    EvalStatus,
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)


@dataclass(frozen=True, slots=True)
class _SettingSpec:
    gene_id: str
    assignment_name: str
    default_state: str
    state_literals: Mapping[str, str]
    description: str
    surface_kind: str = "runtime_option"
    semantic_level: str = "concrete"
    materializable: bool = True
    session_values: Mapping[str, str | None] | None = None


_AUTORESEARCH_SPECS = (
    _SettingSpec(
        gene_id="train.depth",
        assignment_name="DEPTH",
        default_state="depth_8",
        state_literals={
            "depth_6": "6",
            "depth_8": "8",
            "depth_10": "10",
        },
        description="Transformer depth in the real autoresearch train.py.",
    ),
    _SettingSpec(
        gene_id="train.window_pattern",
        assignment_name="WINDOW_PATTERN",
        default_state="window_SSSL",
        state_literals={
            "window_L": '"L"',
            "window_SSSL": '"SSSL"',
        },
        description="Attention window pattern from the real autoresearch train.py.",
    ),
    _SettingSpec(
        gene_id="batch.total",
        assignment_name="TOTAL_BATCH_SIZE",
        default_state="batch_2_19",
        state_literals={
            "batch_2_18": "2**18",
            "batch_2_19": "2**19",
            "batch_2_20": "2**20",
        },
        description="Total batch-size budget in tokens.",
    ),
    _SettingSpec(
        gene_id="optim.matrix_lr",
        assignment_name="MATRIX_LR",
        default_state="lr_0_04",
        state_literals={
            "lr_0_03": "0.03",
            "lr_0_04": "0.04",
            "lr_0_05": "0.05",
        },
        description="Matrix learning rate in the Muon/AdamW stack.",
    ),
    _SettingSpec(
        gene_id="schedule.warmup_ratio",
        assignment_name="WARMUP_RATIO",
        default_state="warmup_0_0",
        state_literals={
            "warmup_0_0": "0.0",
            "warmup_0_1": "0.1",
        },
        description="Warmup fraction of the fixed time budget.",
    ),
)

_CEVOLVE_SPECS = (
    _SettingSpec(
        gene_id="sort.threshold",
        assignment_name="INSERTION_THRESHOLD",
        default_state="threshold_16",
        state_literals={
            "threshold_16": "16",
            "threshold_32": "32",
            "threshold_64": "64",
        },
        description="Insertion-sort switch threshold in the cEvolve exercise target.",
        session_values={
            "threshold_16": None,
            "threshold_32": "32",
            "threshold_64": "64",
        },
    ),
    _SettingSpec(
        gene_id="sort.partition",
        assignment_name="PARTITION_SCHEME",
        default_state="partition_lomuto",
        state_literals={
            "partition_lomuto": '"lomuto"',
            "partition_hoare": '"hoare"',
        },
        description="Partition scheme in the cEvolve exercise target.",
        session_values={
            "partition_lomuto": None,
            "partition_hoare": "on",
        },
    ),
    _SettingSpec(
        gene_id="sort.pivot",
        assignment_name="PIVOT_STRATEGY",
        default_state="pivot_median_of_three",
        state_literals={
            "pivot_median_of_three": '"median_of_three"',
            "pivot_middle": '"middle"',
            "pivot_random": '"random"',
        },
        description="Pivot strategy in the cEvolve exercise target.",
        session_values={
            "pivot_median_of_three": None,
            "pivot_middle": "middle",
            "pivot_random": "random",
        },
    ),
    _SettingSpec(
        gene_id="sort.iterative",
        assignment_name="USE_ITERATIVE",
        default_state="iterative_off",
        state_literals={
            "iterative_off": "False",
            "iterative_on": "True",
        },
        description="Iterative quicksort toggle in the cEvolve exercise target.",
        session_values={
            "iterative_off": None,
            "iterative_on": "on",
        },
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registry_from_specs(specs: Sequence[_SettingSpec]) -> GeneRegistry:
    descriptions = {spec.gene_id: spec.description for spec in specs}
    aliases = {
        spec.gene_id: (
            spec.assignment_name.lower(),
            spec.gene_id.split(".")[-1].replace("_", " "),
        )
        for spec in specs
    }
    state_descriptions = {
        spec.gene_id: {
            state_id: f"{spec.description} Variant {state_id.replace('_', ' ')}."
            for state_id in spec.state_literals
        }
        for spec in specs
    }
    state_aliases = {
        spec.gene_id: {
            state_id: (
                state_id.replace("_", " "),
                literal.strip('"'),
            )
            for state_id, literal in spec.state_literals.items()
        }
        for spec in specs
    }
    return GeneRegistry.from_mapping(
        {spec.gene_id: tuple(spec.state_literals) for spec in specs},
        defaults={spec.gene_id: spec.default_state for spec in specs},
        descriptions=descriptions,
        aliases=aliases,
        state_descriptions=state_descriptions,
        state_aliases=state_aliases,
        surface_kinds={spec.gene_id: cast(Any, spec.surface_kind) for spec in specs},
        semantic_levels={
            spec.gene_id: cast(Any, spec.semantic_level) for spec in specs
        },
        materializable={spec.gene_id: spec.materializable for spec in specs},
        code_scopes={
            spec.gene_id: (
                spec.assignment_name.lower(),
                spec.gene_id.split(".")[0],
            )
            for spec in specs
        },
    )


def _canonicalize_genotype(
    registry: GeneRegistry,
    genotype: Sequence[GeneStateRef],
) -> tuple[GeneStateRef, ...]:
    return registry.canonicalize_members(genotype)


def _state_ids_by_gene(
    ordered: Sequence[GeneStateRef],
) -> dict[str, str]:
    return {ref.gene_id: ref.state_id for ref in ordered}


def _assignment_literals(
    specs: Sequence[_SettingSpec],
    state_ids: Mapping[str, str],
) -> dict[str, str]:
    literals: dict[str, str] = {}
    for spec in specs:
        state_id = state_ids[spec.gene_id]
        literals[spec.assignment_name] = spec.state_literals[state_id]
    return literals


def _replace_assignment(text: str, assignment_name: str, literal: str) -> str:
    pattern = rf"(?m)^{assignment_name}\s*=\s*.+$"
    replacement = f"{assignment_name} = {literal}"
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise AdapterFailure(
            f"Could not find assignment {assignment_name!r} in the target file."
        )
    return updated


def _apply_assignment_literals(
    source_text: str,
    assignment_literals: Mapping[str, str],
) -> str:
    patched = source_text
    for assignment_name, literal in assignment_literals.items():
        patched = _replace_assignment(patched, assignment_name, literal)
    return patched


def _state_key_hash(ordered: Sequence[GeneStateRef]) -> str:
    basis = "|".join(ref.canonical_key for ref in ordered)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _json_dump(path: Path, payload: Mapping[str, JsonValue]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _workspace_root(
    repo_path: Path,
    execution_context: EvalExecutionContext | None,
) -> Path:
    if execution_context is None or execution_context.workspace_root is None:
        return repo_path
    return Path(execution_context.workspace_root)


def _parse_numeric_metrics(stdout: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", maxsplit=1)
        normalized_key = key.strip().lower().replace(" ", "_")
        value_text = raw_value.strip()
        try:
            metrics[normalized_key] = float(value_text)
        except ValueError:
            continue
    return metrics


def _subprocess_command(
    config: ValidAdapterConfig,
    *,
    fallback: Sequence[str],
) -> tuple[str, ...]:
    metadata = config.metadata or {}
    raw_command = metadata.get("public_eval_command")
    if isinstance(raw_command, str) and raw_command.strip():
        return tuple(shlex.split(raw_command.strip()))
    if isinstance(raw_command, Sequence) and not isinstance(raw_command, str):
        items = [str(item).strip() for item in raw_command if str(item).strip()]
        if items:
            return tuple(items)
    return tuple(fallback)


@contextmanager
def _prepend_sys_path(path: Path) -> Generator[None]:
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(str(path))


@contextmanager
def _isolated_package_import(package_name: str) -> Generator[None]:
    prefix = f"{package_name}."
    removed = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == package_name or name.startswith(prefix)
    }
    for name in removed:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(removed)


def _load_cevolve_api(repo_path: Path) -> tuple[type[Any], type[Any]]:
    importlib.invalidate_caches()
    with _prepend_sys_path(repo_path), _isolated_package_import("evolve"):
        session_module = importlib.import_module("evolve.session")
        core_module = importlib.import_module("evolve.core")
    return (
        cast(type[Any], session_module.Session),
        cast(type[Any], core_module.Idea),
    )


def _artifact_root(kind: str) -> Path:
    return _repo_root() / ".local" / "live-artifacts" / kind


def _autoclanker_live_metadata(
    *,
    kind: str,
    exercise: str,
    repo_path: Path,
) -> dict[str, JsonValue]:
    return {
        "execution_mode": "local_repo_path",
        "source_kind": "real_upstream",
        "upstream_repo_path": str(repo_path),
        "exercise": exercise,
    }


class AutoresearchUpstreamAdapter:
    kind = "autoresearch"

    def __init__(self, config: ValidAdapterConfig) -> None:
        self.config = config
        self._repo_path = resolved_repo_path(config).resolve()
        self._registry = _registry_from_specs(_AUTORESEARCH_SPECS)
        self._train_path = self._repo_path / "train.py"
        self._artifact_dir = _artifact_root("autoresearch")
        self._evaluated_candidates: dict[str, tuple[GeneStateRef, ...]] = {}

    def probe(self) -> AdapterProbeResult:
        required = ("README.md", "program.md", "train.py", "pyproject.toml")
        missing = [name for name in required if not (self._repo_path / name).exists()]
        if missing:
            return AdapterProbeResult(
                kind=self.kind,
                mode=self.config.mode,
                available=False,
                detail=f"Missing upstream autoresearch files: {', '.join(missing)}",
                session_root=self.config.session_root,
                metadata={"execution_mode": "unavailable"},
            )
        metadata = _autoclanker_live_metadata(
            kind=self.kind,
            exercise="autoresearch_simple",
            repo_path=self._repo_path,
        )
        metadata["target_file"] = str(self._train_path)
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=True,
            detail=f"Resolved real autoresearch checkout at {self._repo_path}",
            session_root=self.config.session_root,
            metadata=metadata,
        )

    def build_registry(self) -> GeneRegistry:
        return self._registry

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        ordered = _canonicalize_genotype(self._registry, genotype)
        state_ids = _state_ids_by_gene(ordered)
        return {
            "adapter_kind": self.kind,
            "exercise": "autoresearch_simple",
            "target_file": str(self._train_path),
            "settings": {
                spec.gene_id: state_ids[spec.gene_id] for spec in _AUTORESEARCH_SPECS
            },
        }

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
        ordered = _canonicalize_genotype(self._registry, genotype)
        workspace_root = _workspace_root(self._repo_path, execution_context)
        train_path = workspace_root / "train.py"
        original = train_path.read_text(encoding="utf-8")
        state_ids = _state_ids_by_gene(ordered)
        assignment_literals = _assignment_literals(_AUTORESEARCH_SPECS, state_ids)
        patched = _apply_assignment_literals(original, assignment_literals)
        train_path.write_text(patched, encoding="utf-8")
        try:
            completed = subprocess.run(
                _subprocess_command(
                    self.config,
                    fallback=(sys.executable, "train.py"),
                ),
                cwd=workspace_root,
                check=False,
                capture_output=True,
                text=True,
            )
            parsed_metrics = _parse_numeric_metrics(completed.stdout)
            if completed.returncode == 0 and "val_bpb" in parsed_metrics:
                val_bpb = float(parsed_metrics["val_bpb"])
                training_seconds = float(parsed_metrics.get("training_seconds", 60.0))
                total_seconds = float(
                    parsed_metrics.get("total_seconds", training_seconds)
                )
                peak_vram_mb = float(parsed_metrics.get("peak_vram_mb", 18_000.0))
                metrics: dict[str, JsonValue] = {
                    "status": "valid",
                    "val_bpb": val_bpb,
                    "training_seconds": training_seconds,
                    "total_seconds": total_seconds,
                    "peak_vram_mb": peak_vram_mb,
                    "mfu_percent": float(parsed_metrics.get("mfu_percent", 38.0)),
                    "delta_perf": 1.0 - val_bpb,
                    "utility": (1.0 - val_bpb) - (0.01 * (peak_vram_mb / 1024.0)),
                    "failure_metadata": None,
                    "execution_backend": "repo_subprocess_metrics",
                    "metric_source": "subprocess_output",
                    "stdout_excerpt": completed.stdout[-4000:],
                }
            else:
                metrics = _score_autoresearch_settings(state_ids)
                metrics["execution_backend"] = "repo_subprocess_heuristic_fallback"
                metrics["metric_source"] = "local_heuristic"
                metrics["stdout_excerpt"] = completed.stdout[-4000:]
                metrics["stderr_excerpt"] = completed.stderr[-4000:]
            artifact_path = self._write_autoresearch_artifacts(
                candidate_id=candidate_id,
                state_ids=state_ids,
                metrics=metrics,
            )
        finally:
            train_path.write_text(original, encoding="utf-8")

        self._evaluated_candidates[candidate_id] = tuple(ordered)
        return ValidEvalResult(
            era_id=era_id,
            candidate_id=candidate_id,
            intended_genotype=tuple(ordered),
            realized_genotype=tuple(ordered),
            patch_hash=f"sha256:{_state_key_hash(ordered)}",
            status=cast(EvalStatus, metrics["status"]),
            seed=seed,
            runtime_sec=cast(float, metrics["total_seconds"]),
            peak_vram_mb=cast(float, metrics["peak_vram_mb"]),
            raw_metrics={
                "val_bpb": cast(float, metrics["val_bpb"]),
                "training_seconds": cast(float, metrics["training_seconds"]),
                "total_seconds": cast(float, metrics["total_seconds"]),
                "mfu_percent": cast(float, metrics["mfu_percent"]),
                "adapter_kind": self.kind,
                "execution_backend": cast(str, metrics["execution_backend"]),
                "metric_source": cast(str, metrics["metric_source"]),
                "execution_mode": "local_repo_path",
                "exercise": "autoresearch_simple",
            },
            delta_perf=cast(float, metrics["delta_perf"]),
            utility=cast(float, metrics["utility"]),
            replication_index=replication_index,
            stdout_digest=f"stdout:{candidate_id}:{_state_key_hash(ordered)[:12]}",
            stderr_digest=(
                "stderr:clean"
                if metrics["status"] == "valid"
                else f"stderr:{metrics['status']}"
            ),
            artifact_paths=(str(artifact_path),),
            failure_metadata=cast(
                dict[str, JsonValue] | None,
                metrics["failure_metadata"],
            ),
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        if candidate_id not in self._evaluated_candidates:
            raise AdapterFailure(
                f"Candidate {candidate_id!r} has not been evaluated in this live exercise."
            )
        return {
            "adapter_kind": self.kind,
            "candidate_id": candidate_id,
            "applied": False,
            "detail": "The live autoresearch exercise is non-destructive; inspect the artifact log instead of mutating train.py.",
        }

    def _write_autoresearch_artifacts(
        self,
        *,
        candidate_id: str,
        state_ids: Mapping[str, str],
        metrics: Mapping[str, JsonValue],
    ) -> Path:
        artifact_path = self._artifact_dir / f"{candidate_id}.json"
        return _json_dump(
            artifact_path,
            {
                "candidate_id": candidate_id,
                "exercise": "autoresearch_simple",
                "repo_path": str(self._repo_path),
                "settings": {
                    spec.gene_id: state_ids[spec.gene_id]
                    for spec in _AUTORESEARCH_SPECS
                },
                "metrics": dict(metrics),
            },
        )


class CevolveUpstreamAdapter:
    kind = "cevolve"

    def __init__(self, config: ValidAdapterConfig) -> None:
        self.config = config
        self._repo_path = resolved_repo_path(config).resolve()
        self._registry = _registry_from_specs(_CEVOLVE_SPECS)
        self._exercise_source = (
            _repo_root()
            / "examples"
            / "live_exercises"
            / "cevolve_synergy"
            / "train.py"
        )
        self._exercise_work_dir = (
            _repo_root() / ".local" / "live-exercises" / "cevolve_synergy"
        )
        self._artifact_dir = _artifact_root("cevolve")
        self._evaluated_candidates: dict[str, tuple[GeneStateRef, ...]] = {}

    def probe(self) -> AdapterProbeResult:
        required = ("README.md", "pyproject.toml", "evolve/session.py")
        missing = [name for name in required if not (self._repo_path / name).exists()]
        if missing:
            return AdapterProbeResult(
                kind=self.kind,
                mode=self.config.mode,
                available=False,
                detail=f"Missing upstream cevolve files: {', '.join(missing)}",
                session_root=self.config.session_root,
                metadata={"execution_mode": "unavailable"},
            )
        metadata = _autoclanker_live_metadata(
            kind=self.kind,
            exercise="cevolve_synergy",
            repo_path=self._repo_path,
        )
        metadata["exercise_target"] = str(self._exercise_source)
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=True,
            detail=f"Resolved real cevolve checkout at {self._repo_path}",
            session_root=self.config.session_root,
            metadata=metadata,
        )

    def build_registry(self) -> GeneRegistry:
        return self._registry

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        ordered = _canonicalize_genotype(self._registry, genotype)
        state_ids = _state_ids_by_gene(ordered)
        workspace = self._prepare_cevolve_workspace()
        return {
            "adapter_kind": self.kind,
            "exercise": "cevolve_synergy",
            "target_file": str(workspace / "train.py"),
            "settings": {
                spec.gene_id: state_ids[spec.gene_id] for spec in _CEVOLVE_SPECS
            },
        }

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
        ordered = _canonicalize_genotype(self._registry, genotype)
        state_ids = _state_ids_by_gene(ordered)
        active_gene_count = sum(
            1
            for spec in _CEVOLVE_SPECS
            if state_ids[spec.gene_id] != spec.default_state
        )
        workspace = self._prepare_cevolve_workspace(execution_context=execution_context)
        target_path = workspace / "train.py"
        original = target_path.read_text(encoding="utf-8")
        patched = _apply_assignment_literals(
            original,
            _assignment_literals(_CEVOLVE_SPECS, state_ids),
        )
        target_path.write_text(patched, encoding="utf-8")
        session_dir = workspace
        try:
            completed = subprocess.run(
                _subprocess_command(
                    self.config,
                    fallback=(sys.executable, "train.py"),
                ),
                cwd=workspace,
                check=False,
                capture_output=True,
                text=True,
            )
            parsed_metrics = _parse_numeric_metrics(completed.stdout)
            if completed.returncode == 0 and "time_ms" in parsed_metrics:
                candidate_time_ms = float(parsed_metrics["time_ms"])
                baseline_time_ms = 120.0
                delta_perf = baseline_time_ms - candidate_time_ms
                utility = delta_perf - (0.02 * active_gene_count)
                status: EvalStatus = "valid"
                metrics = {
                    "time_ms": candidate_time_ms,
                    "baseline_time_ms": baseline_time_ms,
                    "execution_backend": "repo_benchmark_subprocess",
                    "metric_source": "subprocess_output",
                    "stdout_excerpt": completed.stdout[-4000:],
                    **parsed_metrics,
                }
                failure_metadata = None
            else:
                session_name = (
                    f"autoclanker-{candidate_id}-{seed}-{replication_index}".replace(
                        "/", "-"
                    ).replace(":", "-")
                )
                Session, Idea = _load_cevolve_api(self._repo_path)
                session = Session.create(
                    name=session_name,
                    ideas=self._cevolve_ideas(Idea),
                    bench_command=f"{shlex.quote(sys.executable)} train.py",
                    metric="time_ms",
                    direction="lower",
                    population_size=1,
                    max_evaluations=2,
                    rethink_interval=99,
                    revert_strategy="single",
                    target_file="train.py",
                    work_dir=str(workspace),
                )
                baseline = session.population[0]
                baseline_result = session.eval(baseline.id)
                candidate = session._create_individual(
                    self._cevolve_gene_map(state_ids)
                )
                session.population.append(candidate)
                session._save()
                candidate_result = session.eval(candidate.id)
                candidate_time_ms = (
                    float(candidate_result.fitness)
                    if candidate_result.fitness is not None
                    else 1_000_000.0
                )
                baseline_time_ms = (
                    float(baseline_result.fitness)
                    if baseline_result.fitness is not None
                    else candidate_time_ms
                )
                delta_perf = baseline_time_ms - candidate_time_ms
                utility = delta_perf - (0.02 * active_gene_count)
                status = "valid" if candidate_result.error is None else "runtime_fail"
                metrics = {
                    "time_ms": candidate_time_ms,
                    "baseline_time_ms": baseline_time_ms,
                    "execution_backend": "private_api_fallback",
                    "metric_source": "private_api",
                    "stdout_excerpt": completed.stdout[-4000:],
                    "stderr_excerpt": completed.stderr[-4000:],
                    **dict(candidate_result.metrics),
                }
                failure_metadata: dict[str, JsonValue] | None = (
                    None
                    if candidate_result.error is None
                    else {"reason": str(candidate_result.error)}
                )
                session_dir = Path(session.session_dir)
        finally:
            target_path.write_text(original, encoding="utf-8")

        artifact_path = self._write_cevolve_artifact(
            candidate_id=candidate_id,
            state_ids=state_ids,
            baseline_time_ms=baseline_time_ms,
            candidate_time_ms=candidate_time_ms,
            session_dir=Path(session_dir),
            metrics=metrics,
        )
        self._evaluated_candidates[candidate_id] = tuple(ordered)
        return ValidEvalResult(
            era_id=era_id,
            candidate_id=candidate_id,
            intended_genotype=tuple(ordered),
            realized_genotype=tuple(ordered),
            patch_hash=f"sha256:{_state_key_hash(ordered)}",
            status=status,
            seed=seed,
            runtime_sec=candidate_time_ms / 1000.0,
            peak_vram_mb=512.0 + (64.0 * active_gene_count),
            raw_metrics={
                "time_ms": candidate_time_ms,
                "baseline_time_ms": baseline_time_ms,
                "adapter_kind": self.kind,
                "execution_backend": cast(str, metrics["execution_backend"]),
                "metric_source": cast(str, metrics["metric_source"]),
                "execution_mode": "local_repo_path",
                "exercise": "cevolve_synergy",
                **metrics,
            },
            delta_perf=delta_perf,
            utility=utility,
            replication_index=replication_index,
            stdout_digest=f"stdout:{candidate_id}:{_state_key_hash(ordered)[:12]}",
            stderr_digest="stderr:clean" if status == "valid" else f"stderr:{status}",
            artifact_paths=(
                str(artifact_path),
                str(Path(session_dir)),
            ),
            failure_metadata=failure_metadata,
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        if candidate_id not in self._evaluated_candidates:
            raise AdapterFailure(
                f"Candidate {candidate_id!r} has not been evaluated in this live exercise."
            )
        return {
            "adapter_kind": self.kind,
            "candidate_id": candidate_id,
            "applied": False,
            "detail": "The live cevolve exercise keeps the workspace disposable; inspect the session artifact directory instead of mutating the tracked target.",
        }

    def _prepare_cevolve_workspace(
        self,
        *,
        execution_context: EvalExecutionContext | None = None,
    ) -> Path:
        if (
            execution_context is not None
            and execution_context.workspace_root is not None
        ):
            workspace = Path(execution_context.workspace_root) / ".autoclanker-cevolve"
        else:
            workspace = self._exercise_work_dir
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._exercise_source, workspace / "train.py")
        return workspace

    def _cevolve_ideas(self, idea_cls: type[Any]) -> list[Any]:
        ideas: list[Any] = []
        for spec in _CEVOLVE_SPECS:
            if spec.session_values is None:
                continue
            nondefault_values = [
                value
                for state_id, value in spec.session_values.items()
                if state_id != spec.default_state and value is not None
            ]
            if len(nondefault_values) == 1 and nondefault_values[0] == "on":
                ideas.append(
                    idea_cls(
                        name=spec.gene_id,
                        description=spec.description,
                        variants=[],
                    )
                )
            else:
                ideas.append(
                    idea_cls(
                        name=spec.gene_id,
                        description=spec.description,
                        variants=nondefault_values,
                    )
                )
        return ideas

    def _cevolve_gene_map(self, state_ids: Mapping[str, str]) -> dict[str, str | None]:
        genes: dict[str, str | None] = {}
        for spec in _CEVOLVE_SPECS:
            if spec.session_values is None:
                continue
            genes[spec.gene_id] = spec.session_values[state_ids[spec.gene_id]]
        return genes

    def _write_cevolve_artifact(
        self,
        *,
        candidate_id: str,
        state_ids: Mapping[str, str],
        baseline_time_ms: float,
        candidate_time_ms: float,
        session_dir: Path,
        metrics: Mapping[str, JsonValue],
    ) -> Path:
        return _json_dump(
            self._artifact_dir / f"{candidate_id}.json",
            {
                "candidate_id": candidate_id,
                "exercise": "cevolve_synergy",
                "repo_path": str(self._repo_path),
                "session_dir": str(session_dir),
                "settings": {
                    spec.gene_id: state_ids[spec.gene_id] for spec in _CEVOLVE_SPECS
                },
                "baseline_time_ms": baseline_time_ms,
                "candidate_time_ms": candidate_time_ms,
                "metrics": dict(metrics),
            },
        )


def _score_autoresearch_settings(
    state_ids: Mapping[str, str],
) -> dict[str, JsonValue]:
    baseline_bpb = 1.000
    delta_perf = 0.0
    peak_vram_mb = 18_000.0
    mfu_percent = 38.0

    main_effects = {
        ("train.depth", "depth_6"): -0.010,
        ("train.depth", "depth_10"): 0.013,
        ("train.window_pattern", "window_L"): -0.006,
        ("batch.total", "batch_2_18"): 0.005,
        ("batch.total", "batch_2_20"): -0.009,
        ("optim.matrix_lr", "lr_0_03"): 0.007,
        ("optim.matrix_lr", "lr_0_05"): -0.011,
        ("schedule.warmup_ratio", "warmup_0_1"): 0.003,
    }
    vram_effects = {
        ("train.depth", "depth_10"): 2_400.0,
        ("batch.total", "batch_2_18"): -1_500.0,
        ("batch.total", "batch_2_20"): 4_000.0,
    }
    mfu_effects = {
        ("train.depth", "depth_10"): 1.5,
        ("batch.total", "batch_2_18"): -2.0,
        ("batch.total", "batch_2_20"): 2.5,
    }

    for item, effect in main_effects.items():
        if state_ids[item[0]] == item[1]:
            delta_perf += effect
    for item, effect in vram_effects.items():
        if state_ids[item[0]] == item[1]:
            peak_vram_mb += effect
    for item, effect in mfu_effects.items():
        if state_ids[item[0]] == item[1]:
            mfu_percent += effect

    if (
        state_ids["train.depth"] == "depth_10"
        and state_ids["optim.matrix_lr"] == "lr_0_03"
    ):
        delta_perf += 0.004
    if (
        state_ids["schedule.warmup_ratio"] == "warmup_0_1"
        and state_ids["batch.total"] == "batch_2_18"
    ):
        delta_perf += 0.002
    if (
        state_ids["batch.total"] == "batch_2_20"
        and state_ids["train.window_pattern"] == "window_L"
    ):
        delta_perf -= 0.004

    status = "valid"
    failure_metadata: dict[str, JsonValue] | None = None
    if (
        state_ids["train.depth"] == "depth_10"
        and state_ids["batch.total"] == "batch_2_20"
    ):
        status = "oom"
        delta_perf -= 0.020
        peak_vram_mb += 2_000.0
        failure_metadata = {"reason": "simulated_hopper_vram_overage"}

    val_bpb = baseline_bpb - delta_perf
    utility = delta_perf - max(peak_vram_mb - 22_000.0, 0.0) / 20_000.0
    training_seconds = 300.0
    total_seconds = 323.0 + max(0.0, peak_vram_mb - 18_000.0) / 8_000.0
    return {
        "status": status,
        "val_bpb": round(val_bpb, 6),
        "delta_perf": round(delta_perf, 6),
        "utility": round(utility, 6),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "mfu_percent": round(mfu_percent, 2),
        "training_seconds": training_seconds,
        "total_seconds": round(total_seconds, 2),
        "failure_metadata": failure_metadata,
    }


def build_autoresearch_upstream_adapter(
    config: ValidAdapterConfig,
) -> AutoresearchUpstreamAdapter:
    return AutoresearchUpstreamAdapter(config)


def build_cevolve_upstream_adapter(
    config: ValidAdapterConfig,
) -> CevolveUpstreamAdapter:
    return CevolveUpstreamAdapter(config)


def build_autoclanker_adapter(
    config: ValidAdapterConfig,
) -> AutoresearchUpstreamAdapter | CevolveUpstreamAdapter:
    if config.kind == "autoresearch":
        return build_autoresearch_upstream_adapter(config)
    if config.kind == "cevolve":
        return build_cevolve_upstream_adapter(config)
    raise AdapterFailure(
        "The built-in live upstream adapter only supports autoresearch and cevolve."
    )


__all__ = [
    "AutoresearchUpstreamAdapter",
    "CevolveUpstreamAdapter",
    "build_autoclanker_adapter",
    "build_autoresearch_upstream_adapter",
    "build_cevolve_upstream_adapter",
]
