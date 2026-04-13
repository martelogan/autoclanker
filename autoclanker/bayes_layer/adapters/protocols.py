from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from autoclanker.bayes_layer.registry import GeneRegistry
from autoclanker.bayes_layer.types import (
    EvalContractSnapshot,
    EvalExecutionContext,
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)


@dataclass(frozen=True, slots=True)
class AdapterProbeResult:
    kind: str
    mode: str
    available: bool
    detail: str
    session_root: str
    metadata: Mapping[str, JsonValue] | None = None


class EvalLoopAdapter(Protocol):
    kind: str
    config: ValidAdapterConfig

    def probe(self) -> AdapterProbeResult: ...

    def build_registry(self) -> GeneRegistry: ...

    def capture_eval_contract(self) -> EvalContractSnapshot: ...

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]: ...

    def evaluate_candidate(
        self,
        *,
        era_id: str,
        candidate_id: str,
        genotype: Sequence[GeneStateRef],
        seed: int = 0,
        replication_index: int = 0,
        execution_context: EvalExecutionContext | None = None,
    ) -> ValidEvalResult: ...

    def commit_candidate(
        self,
        candidate_id: str,
    ) -> Mapping[str, JsonValue]: ...
