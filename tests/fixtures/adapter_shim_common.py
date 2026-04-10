from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.adapters.protocols import AdapterProbeResult
from autoclanker.bayes_layer.config import load_bayes_layer_config
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)


class ContractShimAdapter:
    def __init__(
        self,
        config: ValidAdapterConfig,
        *,
        kind: str,
        execution_mode: str,
        detail: str,
    ) -> None:
        self.kind = kind
        self.config = config
        self._execution_mode = execution_mode
        self._detail = detail
        self._fixture = FixtureAdapter(
            ValidAdapterConfig(
                kind="fixture",
                mode="fixture",
                session_root=config.session_root,
                allow_missing=False,
            ),
            bayes_config=load_bayes_layer_config(),
        )

    def probe(self) -> AdapterProbeResult:
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=True,
            detail=self._detail,
            session_root=self.config.session_root,
            metadata={
                "execution_mode": self._execution_mode,
                "shim_kind": self.kind,
            },
        )

    def build_registry(self):  # pragma: no cover - structural helper
        return self._fixture.build_registry()

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        payload = dict(self._fixture.materialize_candidate(genotype))
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
    ) -> ValidEvalResult:
        base = self._fixture.evaluate_candidate(
            era_id=era_id,
            candidate_id=candidate_id,
            genotype=genotype,
            seed=seed,
            replication_index=replication_index,
        )
        return ValidEvalResult(
            era_id=base.era_id,
            candidate_id=base.candidate_id,
            intended_genotype=base.intended_genotype,
            realized_genotype=base.realized_genotype,
            patch_hash=f"{base.patch_hash}:{self.kind}",
            status=base.status,
            seed=base.seed,
            runtime_sec=base.runtime_sec,
            peak_vram_mb=base.peak_vram_mb,
            raw_metrics={
                **base.raw_metrics,
                "adapter_kind": self.kind,
                "execution_mode": self._execution_mode,
            },
            delta_perf=base.delta_perf,
            utility=base.utility,
            replication_index=base.replication_index,
            stdout_digest=base.stdout_digest,
            stderr_digest=base.stderr_digest,
            artifact_paths=base.artifact_paths,
            failure_metadata=base.failure_metadata,
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        payload = dict(self._fixture.commit_candidate(candidate_id))
        payload["adapter_kind"] = self.kind
        payload["execution_mode"] = self._execution_mode
        return payload


def repo_shim_detail(kind: str, repo_path: str | None) -> str:
    resolved = Path(repo_path or ".").resolve()
    return f"Resolved {kind} repo contract shim at {resolved}"
