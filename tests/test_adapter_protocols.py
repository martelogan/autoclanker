from __future__ import annotations

import sys

from pathlib import Path

import pytest

from autoclanker.bayes_layer.adapters import load_adapter
from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.types import AdapterFailure, ValidAdapterConfig
from tests.compliance import covers


def _fixture_config() -> ValidAdapterConfig:
    return ValidAdapterConfig(
        kind="fixture",
        mode="fixture",
        session_root=".autoclanker",
    )


@covers("M4-001")
def test_fixture_adapter_probe_and_evaluate() -> None:
    adapter = FixtureAdapter(_fixture_config())
    probe = adapter.probe()
    registry = adapter.build_registry()
    genotype = registry.default_genotype()
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_fixture",
        genotype=genotype,
        seed=7,
    )
    commit_result = adapter.commit_candidate("cand_fixture")

    assert probe.available is True
    assert probe.kind == "fixture"
    assert "registry_genes" in (probe.metadata or {})
    assert evaluation.patch_hash.startswith("sha256:")
    assert evaluation.realized_genotype == genotype
    assert commit_result["applied"] is False


@covers("M5-001")
def test_autoresearch_adapter_falls_back_when_missing_path_allowed() -> None:
    adapter = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker",
            repo_path=str(Path(".local") / "missing-autoresearch"),
            allow_missing=True,
        )
    )
    registry = adapter.build_registry()
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_autoresearch",
        genotype=registry.default_genotype(),
    )
    probe = adapter.probe()
    metadata = probe.metadata

    assert probe.available is True
    assert metadata is not None
    assert metadata["execution_mode"] == "fixture_fallback"
    assert evaluation.candidate_id == "cand_autoresearch"


@covers("M5-002")
def test_cevolve_adapter_reports_missing_path_when_required() -> None:
    adapter = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker",
            repo_path=str(Path(".local") / "missing-cevolve"),
            allow_missing=False,
        )
    )

    probe = adapter.probe()

    assert probe.available is False
    with pytest.raises(AdapterFailure):
        adapter.build_registry()


@covers("M4-002")
def test_generic_python_module_adapter_probe_and_execute() -> None:
    adapter = load_adapter(
        ValidAdapterConfig(
            kind="python_module",
            mode="installed_module",
            session_root=".autoclanker",
            python_module="tests.fixtures.python_module_adapter",
        )
    )

    probe = adapter.probe()
    registry = adapter.build_registry()
    materialized = adapter.materialize_candidate(registry.default_genotype())
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_python_module",
        genotype=registry.default_genotype(),
    )

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "installed_module"
    assert materialized["execution_mode"] == "installed_module"
    assert evaluation.raw_metrics["execution_mode"] == "installed_module"


@covers("M5-001")
def test_autoresearch_local_repo_probe_and_execute_contract_shim() -> None:
    adapter = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker",
            repo_path=str(Path("tests/fixtures/autoresearch_repo")),
        )
    )
    probe = adapter.probe()
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    commit_result = adapter.commit_candidate("cand_autoresearch")

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "local_repo_path"
    assert materialized["adapter_kind"] == "autoresearch"
    assert materialized["execution_mode"] == "local_repo_path"
    assert commit_result["adapter_kind"] == "autoresearch"


@covers("M5-002")
def test_cevolve_local_repo_probe_and_execute_contract_shim() -> None:
    adapter = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker",
            repo_path=str(Path("tests/fixtures/cevolve_repo")),
        )
    )
    probe = adapter.probe()
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    commit_result = adapter.commit_candidate("cand_cevolve")

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "local_repo_path"
    assert materialized["adapter_kind"] == "cevolve"
    assert materialized["execution_mode"] == "local_repo_path"
    assert commit_result["adapter_kind"] == "cevolve"


@covers("M5-001")
def test_autoresearch_auto_module_probe_and_execute_contract_shim() -> None:
    adapter = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker",
            python_module="tests.fixtures.python_module_adapter",
        )
    )

    probe = adapter.probe()
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_autoresearch_auto_module",
        genotype=adapter.build_registry().default_genotype(),
    )

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "installed_module"
    assert materialized["execution_mode"] == "installed_module"
    assert evaluation.raw_metrics["execution_mode"] == "installed_module"


@covers("M5-002")
def test_cevolve_auto_subprocess_probe_and_execute_contract_shim() -> None:
    adapter = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker",
            command=(
                sys.executable,
                "-m",
                "tests.fixtures.subprocess_adapter",
            ),
        )
    )

    probe = adapter.probe()
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_cevolve_auto_subprocess",
        genotype=adapter.build_registry().default_genotype(),
    )

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "subprocess_cli"
    assert materialized["execution_mode"] == "subprocess_cli"
    assert evaluation.raw_metrics["execution_mode"] == "subprocess_cli"


@covers("M4-003")
def test_generic_subprocess_adapter_probe_and_execute() -> None:
    adapter = load_adapter(
        ValidAdapterConfig(
            kind="subprocess",
            mode="subprocess_cli",
            session_root=".autoclanker",
            command=(
                sys.executable,
                "-m",
                "tests.fixtures.subprocess_adapter",
            ),
        )
    )
    probe = adapter.probe()
    registry = adapter.build_registry()
    materialized = adapter.materialize_candidate(registry.default_genotype())
    evaluation = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_subprocess",
        genotype=registry.default_genotype(),
    )

    assert probe.available is True
    assert probe.metadata is not None
    assert probe.metadata["execution_mode"] == "subprocess_cli"
    assert materialized["execution_mode"] == "subprocess_cli"
    assert evaluation.raw_metrics["execution_mode"] == "subprocess_cli"
