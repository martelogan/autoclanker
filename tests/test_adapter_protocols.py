from __future__ import annotations

import sys

from pathlib import Path
from typing import cast

import pytest

from autoclanker.bayes_layer.adapters import available_adapter_kinds, load_adapter
from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.adapters.fixture import FixtureAdapter
from autoclanker.bayes_layer.belief_io import validate_adapter_config
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    EvalExecutionContext,
    GeneStateRef,
    JsonValue,
    ValidAdapterConfig,
    ValidationFailure,
)
from goalloop.cli import main as goalloop_main
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


@covers("M4-007")
def test_first_party_adapters_capture_eval_contracts_and_execution_context() -> None:
    for adapter in (
        AutoresearchAdapter(
            ValidAdapterConfig(
                kind="autoresearch",
                mode="auto",
                session_root=".autoclanker",
                repo_path=str(Path("tests/fixtures/autoresearch_repo")),
            )
        ),
        CevolveAdapter(
            ValidAdapterConfig(
                kind="cevolve",
                mode="auto",
                session_root=".autoclanker",
                repo_path=str(Path("tests/fixtures/cevolve_repo")),
            )
        ),
    ):
        contract = adapter.capture_eval_contract()
        execution_context = EvalExecutionContext(
            session_id="session_contract_demo",
            era_id="era_contract_v1",
            contract=contract,
            isolation_mode="copy",
            workspace_root="/tmp/autoclanker-isolated-workspace",
            seed=11,
            replication_index=2,
        )
        evaluation = adapter.evaluate_candidate(
            era_id="era_contract_v1",
            candidate_id=f"cand_{adapter.kind}_context",
            genotype=adapter.build_registry().default_genotype(),
            seed=11,
            replication_index=2,
            execution_context=execution_context,
        )

        assert contract.contract_digest.startswith("sha256:")
        assert evaluation.eval_contract is not None
        assert evaluation.execution_metadata is not None
        assert evaluation.eval_contract.contract_digest == contract.contract_digest
        assert evaluation.execution_metadata.contract_digest == contract.contract_digest
        assert (
            evaluation.execution_metadata.workspace_root
            == execution_context.workspace_root
        )
        assert evaluation.execution_metadata.isolation_mode == "copy"


def _goalloop_genes() -> dict[str, object]:
    return {
        "tuning.chunk": {
            "states": ["chunk_small", "chunk_large"],
            "default_state": "chunk_small",
        }
    }


def _goalloop_loop_config(root: Path) -> ValidAdapterConfig:
    return ValidAdapterConfig(
        kind="goalloop",
        mode="local_repo_path",
        session_root=".autoclanker",
        repo_path=str(root),
        metadata={"genes": cast(JsonValue, _goalloop_genes())},
    )


def _scaffold_goalloop_loop(root: Path, *, gates: list[str] | None = None) -> None:
    argv = ["init", "--name", "adapter-demo", "--root", str(root)]
    for gate in gates or [
        "echo CHUNK=$GOALLOOP_GENE_TUNING_CHUNK",
        'printf \'GOALLOOP_METRICS={"delta_perf": 0.25, "custom_count": 3}\n\'',
        'test "$GOALLOOP_GENE_TUNING_CHUNK" = "chunk_large"',
    ]:
        argv.extend(["--gate", gate])
    assert goalloop_main(argv) == 0


@covers("M4-008")
def test_goalloop_adapter_runs_gates_with_genotype_environment(
    tmp_path: Path,
) -> None:
    _scaffold_goalloop_loop(tmp_path)
    adapter = load_adapter(_goalloop_loop_config(tmp_path))
    probe = adapter.probe()
    assert probe.available is True
    probe_metadata = dict(probe.metadata or {})
    assert probe_metadata["contract_locked"] is True
    assert probe_metadata["contract_drifted"] is False
    assert probe_metadata["gates"] == 3

    registry = adapter.build_registry()
    assert registry.known_gene_ids() == ("tuning.chunk",)

    genotype = (GeneStateRef(gene_id="tuning.chunk", state_id="chunk_large"),)
    materialized = adapter.materialize_candidate(genotype)
    assert materialized["environment"] == {"GOALLOOP_GENE_TUNING_CHUNK": "chunk_large"}

    result = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_goalloop",
        genotype=genotype,
        seed=3,
    )
    assert result.status == "valid"
    assert result.delta_perf == 0.25
    assert result.utility == 0.25
    assert result.raw_metrics["custom_count"] == 3
    assert result.raw_metrics["gates_passed"] == 3
    assert result.raw_metrics["goalloop_contract_locked"] is True
    assert result.patch_hash.startswith("sha256:")
    assert result.eval_contract is not None
    assert result.execution_metadata is not None
    assert (
        result.execution_metadata.contract_digest
        == result.eval_contract.contract_digest
    )
    assert (result.evidence_metadata or {})["goalloop_loop_name"] == "adapter-demo"

    commit_result = adapter.commit_candidate("cand_goalloop")
    assert commit_result["applied"] is False


@covers("M4-008")
def test_goalloop_adapter_maps_gate_failures_and_refuses_contract_drift(
    tmp_path: Path,
) -> None:
    _scaffold_goalloop_loop(tmp_path)
    adapter = load_adapter(_goalloop_loop_config(tmp_path))
    registry = adapter.build_registry()

    failed = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_default",
        genotype=registry.default_genotype(),
    )
    assert failed.status == "runtime_fail"
    assert failed.failure_metadata is not None
    assert failed.failure_metadata["reason"] == "gate_failed"
    assert failed.raw_metrics["gates_passed"] == 2
    assert failed.delta_perf == 0.25

    charter_path = tmp_path / "goalloop.charter.md"
    charter_path.write_text(
        charter_path.read_text(encoding="utf-8").replace(
            "echo CHUNK", "echo DRIFTED_CHUNK"
        ),
        encoding="utf-8",
    )
    drifted_probe = load_adapter(_goalloop_loop_config(tmp_path)).probe()
    assert dict(drifted_probe.metadata or {})["contract_drifted"] is True
    with pytest.raises(AdapterFailure, match="drifted"):
        adapter.evaluate_candidate(
            era_id="era_001",
            candidate_id="cand_drift",
            genotype=registry.default_genotype(),
        )


@covers("M4-008")
def test_goalloop_adapter_enforces_config_and_metrics_contracts(
    tmp_path: Path,
) -> None:
    assert "goalloop" in available_adapter_kinds()

    with pytest.raises(ValidationFailure, match="local_repo_path"):
        validate_adapter_config(
            {
                "adapter": {
                    "kind": "goalloop",
                    "mode": "subprocess_cli",
                    "command": ["true"],
                }
            }
        )

    with pytest.raises(AdapterFailure, match="repo_path"):
        load_adapter(
            ValidAdapterConfig(
                kind="goalloop",
                mode="local_repo_path",
                session_root=".autoclanker",
            )
        )

    _scaffold_goalloop_loop(tmp_path)
    bare = ValidAdapterConfig(
        kind="goalloop",
        mode="local_repo_path",
        session_root=".autoclanker",
        repo_path=str(tmp_path),
    )
    with pytest.raises(AdapterFailure, match="metadata.genes"):
        load_adapter(bare).build_registry()

    missing_probe = load_adapter(_goalloop_loop_config(tmp_path / "nowhere")).probe()
    assert missing_probe.available is False

    bad_json_root = tmp_path / "bad-json"
    _scaffold_goalloop_loop(bad_json_root, gates=["echo GOALLOOP_METRICS=notjson"])
    bad_json_adapter = load_adapter(_goalloop_loop_config(bad_json_root))
    with pytest.raises(AdapterFailure, match="Malformed GOALLOOP_METRICS"):
        bad_json_adapter.evaluate_candidate(
            era_id="era_001",
            candidate_id="cand_bad_json",
            genotype=bad_json_adapter.build_registry().default_genotype(),
        )

    bad_type_root = tmp_path / "bad-type"
    _scaffold_goalloop_loop(
        bad_type_root,
        gates=['printf \'GOALLOOP_METRICS={"delta_perf": "high"}\n\''],
    )
    bad_type_adapter = load_adapter(_goalloop_loop_config(bad_type_root))
    with pytest.raises(AdapterFailure, match="must be a number"):
        bad_type_adapter.evaluate_candidate(
            era_id="era_001",
            candidate_id="cand_bad_type",
            genotype=bad_type_adapter.build_registry().default_genotype(),
        )


@covers("M4-008")
def test_goalloop_adapter_output_joining_env_collisions_and_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    # A gate that omits its trailing newline must not merge its metrics line
    # into the next gate's output.
    no_newline_root = tmp_path / "no-newline"
    _scaffold_goalloop_loop(
        no_newline_root,
        gates=[
            "printf 'GOALLOOP_METRICS={\"delta_perf\": 0.5}'",
            "echo trailing-gate",
        ],
    )
    adapter = load_adapter(_goalloop_loop_config(no_newline_root))
    result = adapter.evaluate_candidate(
        era_id="era_001",
        candidate_id="cand_join",
        genotype=adapter.build_registry().default_genotype(),
    )
    assert result.status == "valid"
    assert result.delta_perf == 0.5

    # Distinct gene ids that mangle to the same env var are rejected.
    collision = ValidAdapterConfig(
        kind="goalloop",
        mode="local_repo_path",
        session_root=".autoclanker",
        repo_path=str(no_newline_root),
        metadata={
            "genes": cast(
                JsonValue,
                {
                    "tuning.chunk": {
                        "states": ["a", "b"],
                        "default_state": "a",
                    },
                    "tuning_chunk": {
                        "states": ["a", "b"],
                        "default_state": "a",
                    },
                },
            )
        },
    )
    with pytest.raises(AdapterFailure, match="distinct name"):
        load_adapter(collision).build_registry()

    # Non-finite metric values are a hard error, never a valid result.
    nan_root = tmp_path / "nan"
    _scaffold_goalloop_loop(
        nan_root,
        gates=["printf 'GOALLOOP_METRICS={\"delta_perf\": NaN}\n'"],
    )
    nan_adapter = load_adapter(_goalloop_loop_config(nan_root))
    with pytest.raises(AdapterFailure, match="must be finite"):
        nan_adapter.evaluate_candidate(
            era_id="era_001",
            candidate_id="cand_nan",
            genotype=nan_adapter.build_registry().default_genotype(),
        )
