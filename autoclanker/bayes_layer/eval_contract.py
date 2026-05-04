from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer.belief_io import resolve_relative_path
from autoclanker.bayes_layer.types import (
    EvalContractSnapshot,
    EvalDriftStatus,
    EvalExecutionMetadata,
    EvalMeasurementMode,
    EvalPolicyMode,
    EvalStabilizationMode,
    IsolationMode,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
    to_json_value,
)

_COPY_IGNORE_NAMES = {
    ".git",
    ".autoclanker",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "dist",
    "build",
    "coverage",
}

_LEASE_PREFIX = "autoclanker-eval-locks"


@dataclass(slots=True)
class EvalMeasurementState:
    measurement_mode: EvalMeasurementMode
    stabilization_mode: EvalStabilizationMode
    lease_scope: str | None
    lease_acquired: bool
    lease_wait_sec: float
    noisy_system: bool
    loadavg_1m_before: float | None
    loadavg_1m_after: float | None = None
    stabilization_delay_sec: float = 0.0


def _json_digest(payload: Mapping[str, JsonValue]) -> str:
    rendered = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tree_digest(path: Path) -> str:
    entries: list[str] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path)
        if any(part in _COPY_IGNORE_NAMES for part in relative.parts):
            continue
        if candidate.is_dir():
            continue
        entries.append(f"{relative.as_posix()}:{_file_digest(candidate)}")
    return "sha256:" + hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()


def _path_digest(path: Path | None, *, label: str) -> str:
    if path is None:
        return _json_digest({"label": label, "state": "unspecified"})
    if not path.exists():
        return _json_digest({"label": label, "state": "missing", "path": str(path)})
    if path.is_dir():
        return _tree_digest(path)
    return _file_digest(path)


def _git_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    return Path(resolved) if resolved else None


def _git_head(path: Path | None) -> str | None:
    if path is None:
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    revision = completed.stdout.strip()
    return revision or None


def _git_is_clean(path: Path | None) -> bool:
    if path is None:
        return False
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def _metadata_path(
    config: ValidAdapterConfig,
    value: object,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    base_dir = None if config.base_dir is None else Path(config.base_dir)
    return resolve_relative_path(value.strip(), base_dir=base_dir).expanduser()


def _metadata_path_list(
    config: ValidAdapterConfig,
    value: object,
) -> tuple[Path, ...]:
    if not isinstance(value, list):
        return ()
    items = cast(list[object], value)
    paths: list[Path] = []
    for item in items:
        resolved = _metadata_path(config, item)
        if resolved is not None:
            paths.append(resolved)
    return tuple(paths)


def _resolved_repo_path(config: ValidAdapterConfig) -> Path | None:
    if config.repo_path is None:
        return None
    base_dir = None if config.base_dir is None else Path(config.base_dir)
    return resolve_relative_path(config.repo_path, base_dir=base_dir).expanduser()


def _select_workspace_root(
    config: ValidAdapterConfig,
    *,
    benchmark_root: Path | None,
) -> Path | None:
    metadata = config.metadata or {}
    explicit = _metadata_path(config, metadata.get("workspace_root"))
    if explicit is not None:
        return explicit
    repo_root = _resolved_repo_path(config)
    if repo_root is not None:
        return repo_root
    if benchmark_root is not None:
        return benchmark_root
    return None


def _select_snapshot_mode(
    requested: object,
    *,
    kind: str,
    workspace_root: Path | None,
) -> str:
    if isinstance(requested, str) and requested.strip() in {"git_worktree", "copy"}:
        mode = requested.strip()
        if mode == "git_worktree" and not _git_is_clean(_git_root(workspace_root)):
            return "copy"
        return mode
    if kind == "fixture":
        return "fixture"
    if _git_is_clean(_git_root(workspace_root)):
        return "git_worktree"
    return "copy"


def _eval_policy(config: ValidAdapterConfig) -> Mapping[str, JsonValue]:
    metadata = config.metadata or {}
    raw = metadata.get("eval_policy")
    if isinstance(raw, Mapping):
        return cast(Mapping[str, JsonValue], raw)
    return {}


def _policy_string(
    policy: Mapping[str, JsonValue],
    key: str,
) -> str | None:
    value = policy.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _policy_bool(
    policy: Mapping[str, JsonValue],
    key: str,
) -> bool | None:
    value = policy.get(key)
    if isinstance(value, bool):
        return value
    return None


def _select_measurement_mode(
    requested: object,
    *,
    kind: str,
    performance_sensitive: bool,
) -> EvalMeasurementMode:
    if requested == "exclusive":
        return "exclusive"
    if requested == "parallel_ok":
        return "parallel_ok"
    if requested not in {None, "auto"}:
        return "exclusive" if performance_sensitive else "parallel_ok"
    if kind == "fixture" and requested is None:
        return "parallel_ok"
    return "exclusive" if performance_sensitive else "parallel_ok"


def _select_stabilization_mode(requested: object) -> EvalStabilizationMode:
    if requested == "off":
        return "off"
    if requested == "soft":
        return "soft"
    return "soft"


def _lease_scope(
    requested: object,
    *,
    kind: str,
    benchmark_tree_digest: str,
    eval_harness_digest: str,
) -> str:
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return f"{kind}:{benchmark_tree_digest}:{eval_harness_digest}"


def capture_eval_contract(
    config: ValidAdapterConfig,
    *,
    kind: str,
) -> EvalContractSnapshot:
    metadata = config.metadata or {}
    benchmark_root = _metadata_path(config, metadata.get("benchmark_root"))
    eval_harness_path = _metadata_path(config, metadata.get("eval_harness_path"))
    environment_paths = _metadata_path_list(config, metadata.get("environment_paths"))
    workspace_root = _select_workspace_root(config, benchmark_root=benchmark_root)
    workspace_snapshot_mode = _select_snapshot_mode(
        metadata.get("workspace_snapshot_mode"),
        kind=kind,
        workspace_root=workspace_root,
    )
    policy = _eval_policy(config)
    git_root = _git_root(workspace_root)
    workspace_snapshot_id: str | None = None
    if workspace_snapshot_mode == "fixture":
        workspace_snapshot_id = "fixture:static"
    elif workspace_snapshot_mode == "git_worktree" and git_root is not None:
        revision = _git_head(git_root)
        if revision is not None:
            workspace_snapshot_id = f"git:{revision}"
    elif workspace_root is not None:
        workspace_snapshot_id = (
            f"tree:{_path_digest(workspace_root, label='workspace')}"
        )

    environment_digest = _json_digest(
        {
            "paths": [
                {
                    "path": str(path),
                    "digest": _path_digest(path, label=f"environment:{path.name}"),
                }
                for path in environment_paths
            ]
        }
    )
    benchmark_tree_digest = _path_digest(benchmark_root, label="benchmark")
    eval_harness_digest = _path_digest(eval_harness_path, label="eval_harness")
    performance_sensitive = _policy_bool(policy, "performance_sensitive")
    if performance_sensitive is None:
        performance_sensitive = kind != "fixture"
    measurement_mode = _select_measurement_mode(
        cast(EvalPolicyMode | str | None, _policy_string(policy, "mode")),
        kind=kind,
        performance_sensitive=performance_sensitive,
    )
    stabilization_mode = _select_stabilization_mode(
        _policy_string(policy, "stabilization")
    )
    lease_scope = _lease_scope(
        _policy_string(policy, "lease_scope"),
        kind=kind,
        benchmark_tree_digest=benchmark_tree_digest,
        eval_harness_digest=eval_harness_digest,
    )
    adapter_config_digest = _json_digest(
        {
            "config": {
                "kind": config.kind,
                "mode": config.mode,
                "repo_path": config.repo_path,
                "python_module": config.python_module,
                "command": list(config.command),
                "config_path": config.config_path,
                "metadata": to_json_value(config.metadata or {}),
            }
        }
    )
    captured_paths: dict[str, JsonValue] = {}
    if benchmark_root is not None:
        captured_paths["benchmark_root"] = str(benchmark_root)
    if eval_harness_path is not None:
        captured_paths["eval_harness_path"] = str(eval_harness_path)
    if environment_paths:
        captured_paths["environment_paths"] = [str(path) for path in environment_paths]
    if workspace_root is not None:
        captured_paths["workspace_root"] = str(workspace_root)

    contract_fields: dict[str, JsonValue] = {
        "benchmark_tree_digest": benchmark_tree_digest,
        "eval_harness_digest": eval_harness_digest,
        "adapter_config_digest": adapter_config_digest,
        "environment_digest": environment_digest,
        "measurement_mode": measurement_mode,
        "stabilization_mode": stabilization_mode,
        "lease_scope": lease_scope,
        "workspace_snapshot_mode": workspace_snapshot_mode,
    }
    if workspace_snapshot_id is not None:
        contract_fields["workspace_snapshot_id"] = workspace_snapshot_id
    if captured_paths:
        contract_fields["captured_paths"] = captured_paths
    contract_digest = _json_digest(contract_fields)
    return EvalContractSnapshot(
        contract_digest=contract_digest,
        benchmark_tree_digest=cast(str, contract_fields["benchmark_tree_digest"]),
        eval_harness_digest=cast(str, contract_fields["eval_harness_digest"]),
        adapter_config_digest=adapter_config_digest,
        environment_digest=environment_digest,
        measurement_mode=measurement_mode,
        stabilization_mode=stabilization_mode,
        lease_scope=lease_scope,
        workspace_snapshot_id=workspace_snapshot_id,
        workspace_snapshot_mode=workspace_snapshot_mode,
        captured_paths=captured_paths or None,
        captured_at=datetime.now(tz=UTC).isoformat(),
    )


def compare_eval_contracts(
    expected: EvalContractSnapshot,
    actual: EvalContractSnapshot,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for field_name in (
        "contract_digest",
        "benchmark_tree_digest",
        "eval_harness_digest",
        "adapter_config_digest",
        "environment_digest",
    ):
        if getattr(expected, field_name) != getattr(actual, field_name):
            mismatches.append(field_name)
    return tuple(mismatches)


def drift_status_for_contracts(
    expected: EvalContractSnapshot | None,
    current: EvalContractSnapshot | None,
) -> EvalDriftStatus:
    if expected is None or current is None:
        return "unverified"
    if not compare_eval_contracts(expected, current):
        return "locked"
    return "drifted"


def hardened_eval_result(
    result: ValidEvalResult,
    *,
    contract: EvalContractSnapshot,
    isolation_mode: IsolationMode,
    workspace_root: str | None,
    measurement: EvalMeasurementState | None = None,
) -> ValidEvalResult:
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
        raw_metrics=result.raw_metrics,
        delta_perf=result.delta_perf,
        utility=result.utility,
        replication_index=result.replication_index,
        stdout_digest=result.stdout_digest,
        stderr_digest=result.stderr_digest,
        artifact_paths=result.artifact_paths,
        failure_metadata=result.failure_metadata,
        eval_contract=contract,
        execution_metadata=EvalExecutionMetadata(
            isolation_mode=isolation_mode,
            workspace_root=workspace_root,
            workspace_snapshot_id=contract.workspace_snapshot_id,
            contract_digest=contract.contract_digest,
            measurement_mode=(
                None if measurement is None else measurement.measurement_mode
            ),
            stabilization_mode=(
                None if measurement is None else measurement.stabilization_mode
            ),
            lease_scope=None if measurement is None else measurement.lease_scope,
            lease_acquired=(
                None if measurement is None else measurement.lease_acquired
            ),
            lease_wait_sec=(
                None if measurement is None else measurement.lease_wait_sec
            ),
            noisy_system=None if measurement is None else measurement.noisy_system,
            loadavg_1m_before=(
                None if measurement is None else measurement.loadavg_1m_before
            ),
            loadavg_1m_after=(
                None if measurement is None else measurement.loadavg_1m_after
            ),
            stabilization_delay_sec=(
                None if measurement is None else measurement.stabilization_delay_sec
            ),
        ),
    )


def execution_workspace_root(contract: EvalContractSnapshot) -> Path | None:
    captured = contract.captured_paths or {}
    raw = captured.get("workspace_root")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def _loadavg_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def _is_noisy(loadavg_1m: float | None) -> bool:
    if loadavg_1m is None:
        return False
    cpu_count = os.cpu_count() or 1
    return loadavg_1m > (cpu_count * 1.25)


def _lease_directory() -> Path:
    return Path(tempfile.gettempdir()) / _LEASE_PREFIX


@contextmanager
def measured_execution_window(
    contract: EvalContractSnapshot,
) -> Generator[EvalMeasurementState]:
    measurement_mode = contract.measurement_mode or "parallel_ok"
    stabilization_mode = contract.stabilization_mode or "off"
    lease_scope = contract.lease_scope
    state = EvalMeasurementState(
        measurement_mode=measurement_mode,
        stabilization_mode=stabilization_mode,
        lease_scope=lease_scope,
        lease_acquired=False,
        lease_wait_sec=0.0,
        noisy_system=False,
        loadavg_1m_before=_loadavg_1m(),
    )
    state.noisy_system = _is_noisy(state.loadavg_1m_before)

    handle = None
    try:
        if measurement_mode == "exclusive" and lease_scope is not None:
            lease_digest = hashlib.sha256(lease_scope.encode("utf-8")).hexdigest()
            lease_path = _lease_directory() / f"{lease_digest}.lock"
            lease_path.parent.mkdir(parents=True, exist_ok=True)
            handle = lease_path.open("a+", encoding="utf-8")
            wait_started = time.monotonic()
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            state.lease_acquired = True
            state.lease_wait_sec = time.monotonic() - wait_started
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {
                        "lease_scope": lease_scope,
                        "contract_digest": contract.contract_digest,
                        "pid": os.getpid(),
                        "acquired_at": datetime.now(tz=UTC).isoformat(),
                    }
                )
            )
            handle.flush()

        if stabilization_mode == "soft":
            delay = 0.15 if measurement_mode == "exclusive" else 0.05
            state.stabilization_delay_sec = delay
            time.sleep(delay)
            stabilized_load = _loadavg_1m()
            if state.loadavg_1m_before is None:
                state.loadavg_1m_before = stabilized_load
            state.noisy_system = state.noisy_system or _is_noisy(stabilized_load)

        yield state
    finally:
        state.loadavg_1m_after = _loadavg_1m()
        state.noisy_system = state.noisy_system or _is_noisy(state.loadavg_1m_after)
        if handle is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


@contextmanager
def isolated_execution_workspace(
    contract: EvalContractSnapshot,
) -> Generator[tuple[IsolationMode, str | None]]:
    source_root = execution_workspace_root(contract)
    if contract.workspace_snapshot_mode == "fixture" or source_root is None:
        yield ("fixture", None)
        return

    if contract.workspace_snapshot_mode == "git_worktree":
        git_root = _git_root(source_root)
        revision = (
            contract.workspace_snapshot_id.removeprefix("git:")
            if (
                contract.workspace_snapshot_id
                and contract.workspace_snapshot_id.startswith("git:")
            )
            else _git_head(git_root)
        )
        if git_root is not None and revision is not None:
            with tempfile.TemporaryDirectory(prefix="autoclanker-worktree-") as tempdir:
                temp_path = Path(tempdir)
                completed = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(git_root),
                        "worktree",
                        "add",
                        "--detach",
                        str(temp_path),
                        revision,
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode == 0:
                    try:
                        yield ("git_worktree", str(temp_path))
                    finally:
                        subprocess.run(
                            [
                                "git",
                                "-C",
                                str(git_root),
                                "worktree",
                                "remove",
                                "--force",
                                str(temp_path),
                            ],
                            check=False,
                            capture_output=True,
                            text=True,
                        )
                    return

    with tempfile.TemporaryDirectory(prefix="autoclanker-copy-") as tempdir:
        temp_path = Path(tempdir) / source_root.name
        shutil.copytree(
            source_root,
            temp_path,
            ignore=shutil.ignore_patterns(*sorted(_COPY_IGNORE_NAMES)),
            dirs_exist_ok=False,
        )
        yield ("copy", str(temp_path))
