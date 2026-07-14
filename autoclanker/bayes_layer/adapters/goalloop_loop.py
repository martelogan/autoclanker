"""Adapter that treats a goal loop's charter gates as the eval surface.

A goal loop (see ``docs/GOALLOOP.md``) already carries autoclanker's core
invariant in file form: a locked, deterministic definition of green. This
adapter closes the circle — a Bayesian session can guide parameterized
iterations of a long-running effort by evaluating candidate genotypes against
the loop's own gates. Genotypes reach the gates as ``GOALLOOP_GENE_*``
environment variables; gates may report measurements by printing a single
``GOALLOOP_METRICS={...}`` JSON line. Evaluation is refused outright when the
loop's charter contract has drifted from its lock, mirroring the session rule
that eval results with a mismatched contract are rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from autoclanker.bayes_layer.adapters.protocols import AdapterProbeResult
from autoclanker.bayes_layer.eval_contract import capture_eval_contract
from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    EvalContractSnapshot,
    EvalExecutionContext,
    EvalExecutionMetadata,
    EvalStatus,
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)
from goalloop.model import (
    Charter,
    LoopPaths,
    contract_digest,
    load_charter,
    load_requirements,
    locked_contract_digest,
)

GOALLOOP_METRICS_PREFIX = "GOALLOOP_METRICS="
_RESERVED_METRIC_KEYS = ("delta_perf", "utility", "peak_vram_mb")


def gene_environment_name(gene_id: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in gene_id.upper())
    return f"GOALLOOP_GENE_{cleaned}"


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _metric_float(metrics: Mapping[str, JsonValue], key: str, default: float) -> float:
    value = metrics.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdapterFailure(
            f"GOALLOOP_METRICS field {key!r} must be a number, got {value!r}."
        )
    return float(value)


def _parse_metrics_line(stdout: str) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith(GOALLOOP_METRICS_PREFIX):
            continue
        raw = stripped[len(GOALLOOP_METRICS_PREFIX) :]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AdapterFailure(
                f"Malformed GOALLOOP_METRICS JSON from a gate: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise AdapterFailure(
                "GOALLOOP_METRICS output must be a JSON object of metrics."
            )
        payload = {
            str(key): cast(JsonValue, value)
            for key, value in cast(dict[str, object], parsed).items()
        }
    return payload


class GoalloopAdapter:
    kind = "goalloop"

    def __init__(self, config: ValidAdapterConfig) -> None:
        self.config = config
        if config.repo_path is None:
            raise AdapterFailure(
                "goalloop adapters require repo_path: the goal loop root."
            )
        base = Path(config.base_dir) if config.base_dir else Path.cwd()
        root = Path(config.repo_path)
        self._loop_root = root if root.is_absolute() else (base / root).resolve()
        self._paths = LoopPaths(root=self._loop_root)
        self._registry: GeneRegistry | None = None

    def _charter(self) -> Charter:
        try:
            return load_charter(self._paths)
        except ValueError as exc:
            raise AdapterFailure(str(exc)) from exc

    def probe(self) -> AdapterProbeResult:
        try:
            charter = load_charter(self._paths)
            rows = load_requirements(self._paths)
        except ValueError as exc:
            return AdapterProbeResult(
                kind=self.kind,
                mode=self.config.mode,
                available=False,
                detail=str(exc),
                session_root=self.config.session_root,
                metadata={"loop_root": str(self._loop_root)},
            )
        digest = contract_digest(charter)
        locked = locked_contract_digest(self._paths)
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=True,
            detail=(
                f"goal loop {charter.name!r} at {self._loop_root} "
                f"with {len(charter.gates)} gate(s)"
            ),
            session_root=self.config.session_root,
            metadata={
                "loop_root": str(self._loop_root),
                "loop_name": charter.name,
                "gates": len(charter.gates),
                "rows_total": len(rows),
                "rows_finished": sum(1 for row in rows if row.finished),
                "contract_digest": digest,
                "contract_locked": locked is not None,
                "contract_drifted": locked is not None and locked != digest,
            },
        )

    def build_registry(self) -> GeneRegistry:
        if self._registry is not None:
            return self._registry
        metadata = self.config.metadata or {}
        genes = metadata.get("genes")
        if not isinstance(genes, dict) or not genes:
            raise AdapterFailure(
                "goalloop adapters require metadata.genes: a serialized gene "
                "registry mapping gene ids to {states, default_state} objects "
                "describing the knobs the loop's gates read from "
                "GOALLOOP_GENE_* environment variables."
            )
        self._registry = GeneRegistry.from_serialized_dict(
            cast(Mapping[str, object], genes)
        )
        return self._registry

    def capture_eval_contract(self) -> EvalContractSnapshot:
        return capture_eval_contract(self.config, kind=self.kind)

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        ordered = self.build_registry().canonicalize_members(genotype)
        environment = {
            gene_environment_name(ref.gene_id): ref.state_id for ref in ordered
        }
        return {
            "adapter_kind": self.kind,
            "loop_root": str(self._loop_root),
            "genotype": [
                {"gene_id": ref.gene_id, "state_id": ref.state_id} for ref in ordered
            ],
            "environment": cast(JsonValue, environment),
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
        registry = self.build_registry()
        ordered = registry.canonicalize_members(genotype)
        charter = self._charter()
        digest = contract_digest(charter)
        locked = locked_contract_digest(self._paths)
        if locked is not None and locked != digest:
            raise AdapterFailure(
                "The goal loop's charter contract drifted from its lock; "
                "refusing to evaluate against a moved definition of green. "
                "Re-lock intentionally with `goalloop lock`."
            )
        contract = self.capture_eval_contract()

        environment = dict(os.environ)
        for ref in ordered:
            environment[gene_environment_name(ref.gene_id)] = ref.state_id
        environment["GOALLOOP_CANDIDATE_ID"] = candidate_id
        environment["GOALLOOP_ERA_ID"] = era_id
        environment["GOALLOOP_SEED"] = str(seed)

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        gates_passed = 0
        failed_gate: str | None = None
        failed_exit_code = 0
        started = time.monotonic()
        for gate in charter.gates:
            completed = subprocess.run(
                gate,
                shell=True,
                cwd=self._loop_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            stdout_parts.append(completed.stdout)
            stderr_parts.append(completed.stderr)
            if completed.returncode != 0:
                failed_gate = gate
                failed_exit_code = completed.returncode
                break
            gates_passed += 1
        runtime_sec = time.monotonic() - started

        stdout_text = "".join(stdout_parts)
        stderr_text = "".join(stderr_parts)
        metrics = _parse_metrics_line(stdout_text)
        delta_perf = _metric_float(metrics, "delta_perf", 0.0)
        utility = _metric_float(metrics, "utility", delta_perf)
        peak_vram_mb = _metric_float(metrics, "peak_vram_mb", 0.0)

        rows = load_requirements(self._paths)
        status: EvalStatus = "valid" if failed_gate is None else "runtime_fail"
        failure_metadata: dict[str, JsonValue] | None = None
        if failed_gate is not None:
            failure_metadata = {
                "reason": "gate_failed",
                "gate": failed_gate,
                "exit_code": failed_exit_code,
                "output_tail": (stdout_text + stderr_text)[-2000:],
            }

        raw_metrics: dict[str, JsonValue] = {
            key: value
            for key, value in metrics.items()
            if key not in _RESERVED_METRIC_KEYS
        }
        raw_metrics.update(
            {
                "gates_total": len(charter.gates),
                "gates_passed": gates_passed,
                "rows_total": len(rows),
                "rows_finished": sum(1 for row in rows if row.finished),
                "goalloop_contract_digest": digest,
                "goalloop_contract_locked": locked is not None,
            }
        )

        patch_basis = "|".join(sorted(ref.canonical_key for ref in ordered))
        return ValidEvalResult(
            era_id=era_id,
            candidate_id=candidate_id,
            intended_genotype=tuple(ordered),
            realized_genotype=tuple(ordered),
            patch_hash=_sha256(patch_basis),
            status=status,
            seed=seed,
            runtime_sec=runtime_sec,
            peak_vram_mb=peak_vram_mb,
            raw_metrics=raw_metrics,
            delta_perf=delta_perf,
            utility=utility,
            replication_index=replication_index,
            stdout_digest=_sha256(stdout_text),
            stderr_digest=_sha256(stderr_text),
            failure_metadata=failure_metadata,
            eval_contract=contract,
            execution_metadata=EvalExecutionMetadata(
                isolation_mode=(
                    "copy"
                    if execution_context is None
                    else execution_context.isolation_mode
                ),
                workspace_root=(
                    str(self._loop_root)
                    if execution_context is None
                    else execution_context.workspace_root
                ),
                workspace_snapshot_id=contract.workspace_snapshot_id,
                contract_digest=contract.contract_digest,
            ),
            evidence_metadata={
                "goalloop_loop_name": charter.name,
                "goalloop_contract_digest": digest,
            },
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        return {
            "adapter_kind": self.kind,
            "candidate_id": candidate_id,
            "applied": False,
            "detail": "goalloop candidates are committed through the loop "
            "itself: apply the winning configuration, flip the tracker rows, "
            "and commit them together.",
        }
