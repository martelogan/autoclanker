from __future__ import annotations

import json
import os

from pathlib import Path
from typing import Protocol, cast

import pytest

from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.types import (
    GeneStateRef,
    ValidAdapterConfig,
    ValidEvalResult,
)
from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


class _RegistryProtocol(Protocol):
    def default_genotype(self) -> tuple[GeneStateRef, ...]: ...


class _RegistryBuilder(Protocol):
    def build_registry(self) -> _RegistryProtocol: ...


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise AssertionError(
            f"Missing required live adapter environment variable {name}."
        )
    return value


def _expected_payload(exercise: str) -> dict[str, object]:
    path = ROOT / "examples" / "live_exercises" / exercise / "expected_outcome.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_float(result: ValidEvalResult, key: str) -> float:
    return float(cast(float | int | str, result.raw_metrics[key]))


def _genotype_from_mapping(
    adapter: _RegistryBuilder,
    mapping: dict[str, object],
) -> tuple[GeneStateRef, ...]:
    registry = adapter.build_registry()
    genotype = {ref.gene_id: ref for ref in registry.default_genotype()}
    for gene_id, state_id in mapping.items():
        genotype[str(gene_id)] = GeneStateRef(
            gene_id=str(gene_id),
            state_id=str(state_id),
        )
    return tuple(genotype.values())


@covers("M5-LIVE-001")
@pytest.mark.upstream_live
def test_live_autoresearch_adapter_smoke() -> None:
    repo_path = _required_env("AUTOCLANKER_LIVE_AUTORESEARCH_PATH")
    expected = _expected_payload("autoresearch_simple")
    adapter = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path=repo_path,
            allow_missing=False,
            metadata={
                "adapter_module": os.environ.get(
                    "AUTOCLANKER_LIVE_AUTORESEARCH_ADAPTER_MODULE",
                    "autoclanker.bayes_layer.live_upstreams",
                )
            },
        )
    )

    probe = adapter.probe()
    registry = adapter.build_registry()
    payload = adapter.materialize_candidate(registry.default_genotype())
    baseline = adapter.evaluate_candidate(
        era_id="era_live_001",
        candidate_id="cand_live_autoresearch_baseline",
        genotype=_genotype_from_mapping(
            adapter,
            cast(dict[str, object], expected["baseline"]),
        ),
        seed=7,
    )
    evaluation = adapter.evaluate_candidate(
        era_id="era_live_001",
        candidate_id="cand_live_autoresearch",
        genotype=_genotype_from_mapping(
            adapter,
            cast(dict[str, object], expected["improved"]),
        ),
        seed=11,
    )
    failure = adapter.evaluate_candidate(
        era_id="era_live_001",
        candidate_id="cand_live_autoresearch_failure",
        genotype=_genotype_from_mapping(
            adapter,
            cast(dict[str, object], expected["failure_candidate"]),
        ),
        seed=13,
    )
    commit = adapter.commit_candidate("cand_live_autoresearch")

    assert Path(repo_path).exists()
    assert (Path(repo_path) / "program.md").exists()
    assert (Path(repo_path) / "train.py").exists()
    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] != "fixture_fallback"
    assert payload["adapter_kind"] == "autoresearch"
    assert _metric_float(baseline, "val_bpb") > _metric_float(evaluation, "val_bpb")
    assert evaluation.candidate_id == "cand_live_autoresearch"
    assert evaluation.raw_metrics["adapter_kind"] == "autoresearch"
    assert evaluation.raw_metrics["execution_mode"] != "fixture_fallback"
    assert evaluation.delta_perf >= cast(
        float,
        cast(dict[str, object], expected["expectation"])["delta_perf_at_least"],
    )
    assert failure.status == cast(
        str,
        cast(dict[str, object], expected["expectation"])["failure_status"],
    )
    assert commit["adapter_kind"] == "autoresearch"
    assert commit["applied"] is False


@covers("M5-LIVE-002")
@pytest.mark.upstream_live
def test_live_cevolve_adapter_smoke() -> None:
    repo_path = _required_env("AUTOCLANKER_LIVE_CEVOLVE_PATH")
    expected = _expected_payload("cevolve_synergy")
    adapter = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path=repo_path,
            allow_missing=False,
            metadata={
                "adapter_module": os.environ.get(
                    "AUTOCLANKER_LIVE_CEVOLVE_ADAPTER_MODULE",
                    "autoclanker.bayes_layer.live_upstreams",
                )
            },
        )
    )

    probe = adapter.probe()
    registry = adapter.build_registry()
    payload = adapter.materialize_candidate(registry.default_genotype())
    baseline = adapter.evaluate_candidate(
        era_id="era_live_001",
        candidate_id="cand_live_cevolve_baseline",
        genotype=_genotype_from_mapping(
            adapter,
            cast(dict[str, object], expected["baseline"]),
        ),
        seed=5,
    )
    evaluation = adapter.evaluate_candidate(
        era_id="era_live_001",
        candidate_id="cand_live_cevolve",
        genotype=_genotype_from_mapping(
            adapter,
            cast(dict[str, object], expected["improved"]),
        ),
        seed=13,
    )
    single_changes = cast(list[object], expected["single_changes"])
    single_results = [
        adapter.evaluate_candidate(
            era_id="era_live_001",
            candidate_id=f"cand_live_cevolve_single_{index}",
            genotype=_genotype_from_mapping(
                adapter,
                cast(dict[str, object], item),
            ),
            seed=21 + index,
        )
        for index, item in enumerate(single_changes, start=1)
    ]
    commit = adapter.commit_candidate("cand_live_cevolve")

    assert Path(repo_path).exists()
    assert (Path(repo_path) / "evolve" / "session.py").exists()
    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] != "fixture_fallback"
    assert payload["adapter_kind"] == "cevolve"
    assert _metric_float(baseline, "time_ms") > _metric_float(evaluation, "time_ms")
    assert evaluation.candidate_id == "cand_live_cevolve"
    assert evaluation.raw_metrics["adapter_kind"] == "cevolve"
    assert evaluation.raw_metrics["execution_mode"] != "fixture_fallback"
    assert cast(float, baseline.raw_metrics["time_ms"]) - cast(
        float, evaluation.raw_metrics["time_ms"]
    ) >= cast(
        float,
        cast(dict[str, object], expected["expectation"])[
            "time_ms_improvement_at_least"
        ],
    )
    assert all(
        cast(float, result.raw_metrics["time_ms"])
        - cast(float, evaluation.raw_metrics["time_ms"])
        >= cast(
            float,
            cast(dict[str, object], expected["expectation"])["synergy_margin_at_least"],
        )
        for result in single_results
    )
    assert commit["adapter_kind"] == "cevolve"
    assert commit["applied"] is False
