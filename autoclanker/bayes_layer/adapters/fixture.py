from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence

from autoclanker.bayes_layer.adapters.protocols import AdapterProbeResult
from autoclanker.bayes_layer.config import BayesLayerConfig, load_bayes_layer_config
from autoclanker.bayes_layer.registry import GeneRegistry, build_fixture_registry
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidEvalResult,
)


class FixtureAdapter:
    kind = "fixture"

    def __init__(
        self,
        config: ValidAdapterConfig,
        *,
        bayes_config: BayesLayerConfig | None = None,
    ) -> None:
        self.config = config
        self._bayes_config = bayes_config or load_bayes_layer_config()
        self._registry = build_fixture_registry()

    def probe(self) -> AdapterProbeResult:
        return AdapterProbeResult(
            kind=self.kind,
            mode=self.config.mode,
            available=True,
            detail="fixture adapter is available",
            session_root=self.config.session_root,
            metadata={"registry_genes": len(self._registry.genes)},
        )

    def build_registry(self) -> GeneRegistry:
        return self._registry

    def materialize_candidate(
        self,
        genotype: Sequence[GeneStateRef],
    ) -> Mapping[str, JsonValue]:
        ordered = self._registry.canonicalize_members(genotype)
        return {
            "adapter_kind": self.kind,
            "genotype": [
                {"gene_id": ref.gene_id, "state_id": ref.state_id} for ref in ordered
            ],
        }

    def evaluate_candidate(
        self,
        *,
        era_id: str,
        candidate_id: str,
        genotype: Sequence[GeneStateRef],
        seed: int = 0,
        replication_index: int = 0,
    ) -> ValidEvalResult:
        ordered = self._registry.canonicalize_members(genotype)
        state_keys = {ref.canonical_key for ref in ordered}

        delta_perf = 0.0
        peak_vram_mb = 18_000.0
        nondefault_gene_count = 0

        for ref in ordered:
            default_state = self._registry.genes[ref.gene_id].default_state
            if ref.state_id != default_state:
                nondefault_gene_count += 1

        contribution_table = {
            "parser.matcher:matcher_compiled": 0.35,
            "parser.matcher:matcher_jit": 0.18,
            "parser.plan:plan_context_pair": 0.18,
            "parser.plan:plan_full_scan": -0.08,
            "capture.window:window_wide": 0.12,
            "io.chunk:chunk_large": 0.10,
            "emit.summary:summary_streaming": 0.08,
        }
        vram_table = {
            "capture.window:window_wide": 3_500.0,
            "io.chunk:chunk_large": 3_000.0,
            "parser.plan:plan_full_scan": 1_000.0,
        }
        for state_key in state_keys:
            delta_perf += contribution_table.get(state_key, 0.0)
            peak_vram_mb += vram_table.get(state_key, 0.0)

        if {
            "parser.matcher:matcher_compiled",
            "parser.plan:plan_context_pair",
        }.issubset(state_keys):
            delta_perf += 0.24
        if {
            "capture.window:window_wide",
            "io.chunk:chunk_large",
        }.issubset(state_keys):
            delta_perf -= 0.40
            peak_vram_mb += 2_500.0
        if {
            "parser.matcher:matcher_jit",
            "parser.plan:plan_full_scan",
        }.issubset(state_keys):
            delta_perf -= 0.18
            peak_vram_mb += 500.0

        deterministic_jitter = ((seed % 7) - 3) * 0.002
        delta_perf += deterministic_jitter

        soft_limit = self._bayes_config.utility.soft_vram_limit_mb
        vram_overage_units = max(peak_vram_mb - soft_limit, 0.0) / 1_000.0
        utility = (
            delta_perf
            - (self._bayes_config.utility.lambda_sparsity * nondefault_gene_count)
            - (self._bayes_config.utility.lambda_vram * vram_overage_units)
        )

        status: str = "valid"
        failure_metadata: dict[str, JsonValue] = {}
        if {
            "parser.matcher:matcher_jit",
            "parser.plan:plan_full_scan",
        }.issubset(state_keys):
            status = "runtime_fail"
            utility -= 0.25
            failure_metadata = {"reason": "unstable_jit_full_scan_combo"}
        elif peak_vram_mb > (soft_limit + 800.0):
            status = "oom"
            utility -= 0.3
            failure_metadata = {"reason": "simulated_capture_buffer_overage"}

        patch_basis = "|".join(sorted(state_keys))
        patch_hash = f"sha256:{hashlib.sha256(patch_basis.encode('utf-8')).hexdigest()}"

        return ValidEvalResult(
            era_id=era_id,
            candidate_id=candidate_id,
            intended_genotype=tuple(ordered),
            realized_genotype=tuple(ordered),
            patch_hash=patch_hash,
            status="valid" if status == "valid" else status,
            seed=seed,
            runtime_sec=180.0 + (15.0 * nondefault_gene_count),
            peak_vram_mb=peak_vram_mb,
            raw_metrics={
                "score": round(0.55 + delta_perf, 6),
                "nondefault_gene_count": nondefault_gene_count,
                "seed": seed,
            },
            delta_perf=delta_perf,
            utility=utility,
            replication_index=replication_index,
            stdout_digest=f"stdout:{candidate_id}:{seed}",
            stderr_digest="stderr:clean" if status == "valid" else f"stderr:{status}",
            artifact_paths=(f"artifacts/{era_id}/{candidate_id}/metrics.json",),
            failure_metadata=failure_metadata,
        )

    def commit_candidate(self, candidate_id: str) -> Mapping[str, JsonValue]:
        return {
            "adapter_kind": self.kind,
            "candidate_id": candidate_id,
            "applied": False,
            "detail": "fixture adapter does not perform real commits",
        }
