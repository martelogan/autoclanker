from __future__ import annotations

import textwrap

from pathlib import Path
from typing import Literal, cast

import pytest

from autoclanker.bayes_layer.adapters.autoresearch import AutoresearchAdapter
from autoclanker.bayes_layer.adapters.cevolve import CevolveAdapter
from autoclanker.bayes_layer.live_upstreams import (
    AutoresearchUpstreamAdapter,
    CevolveUpstreamAdapter,
    build_autoclanker_adapter,
)
from autoclanker.bayes_layer.types import (
    AdapterFailure,
    GeneStateRef,
    ValidAdapterConfig,
    ValidEvalResult,
)
from tests.compliance import covers


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def _autoclanker_config(
    kind: Literal["autoresearch", "cevolve"],
    repo_path: Path,
) -> ValidAdapterConfig:
    return ValidAdapterConfig(
        kind=kind,
        mode="auto",
        session_root=".autoclanker-live",
        repo_path=str(repo_path),
        allow_missing=False,
    )


def _genotype_from_mapping(
    adapter: AutoresearchUpstreamAdapter | CevolveUpstreamAdapter,
    mapping: dict[str, str],
) -> tuple[GeneStateRef, ...]:
    registry = adapter.build_registry()
    genotype = {ref.gene_id: ref for ref in registry.default_genotype()}
    for gene_id, state_id in mapping.items():
        genotype[gene_id] = GeneStateRef(gene_id=gene_id, state_id=state_id)
    return tuple(genotype.values())


def _metric_float(adapter_result: ValidEvalResult, key: str) -> float:
    return float(cast(float | int | str, adapter_result.raw_metrics[key]))


def _make_autoresearch_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "autoresearch"
    _write(repo / "README.md", "# Autoresearch\n")
    _write(repo / "program.md", "# Program\n")
    _write(repo / "pyproject.toml", "[project]\nname='autoresearch'\nversion='0.0.0'\n")
    _write(
        repo / "train.py",
        """
        DEPTH = 8
        WINDOW_PATTERN = "SSSL"
        TOTAL_BATCH_SIZE = 2**19
        MATRIX_LR = 0.04
        WARMUP_RATIO = 0.0
        """,
    )
    return repo


def _make_cevolve_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "cevolve"
    _write(repo / "README.md", "# cEvolve\n")
    _write(repo / "pyproject.toml", "[project]\nname='cevolve'\nversion='0.0.0'\n")
    _write(repo / "evolve" / "__init__.py", "")
    _write(
        repo / "evolve" / "core.py",
        """
        from __future__ import annotations

        from dataclasses import dataclass


        @dataclass
        class Idea:
            name: str
            description: str
            variants: list[str]


        @dataclass
        class Individual:
            id: str
            genes: dict[str, str | None]
        """,
    )
    _write(
        repo / "evolve" / "session.py",
        """
        from __future__ import annotations

        import subprocess

        from dataclasses import dataclass
        from pathlib import Path

        from .core import Individual


        @dataclass
        class EvalResult:
            fitness: float | None
            metrics: dict[str, float]
            error: str | None = None


        class Session:
            def __init__(
                self,
                *,
                name: str,
                ideas: list[object],
                bench_command: str,
                metric: str,
                work_dir: str,
                target_file: str,
            ) -> None:
                self.ideas = ideas
                self.population = [
                    Individual(
                        id="baseline",
                        genes={idea.name: None for idea in ideas},
                    )
                ]
                self._bench_command = bench_command
                self._metric = metric
                self._work_dir = Path(work_dir)
                self._target_path = self._work_dir / target_file
                self._baseline_text = self._target_path.read_text(encoding="utf-8")
                self.session_dir = self._work_dir / ".cevolve" / name
                self.session_dir.mkdir(parents=True, exist_ok=True)

            @classmethod
            def create(
                cls,
                *,
                name: str,
                ideas: list[object],
                bench_command: str,
                metric: str = "time_ms",
                work_dir: str = ".",
                target_file: str = "train.py",
                **_: object,
            ) -> "Session":
                return cls(
                    name=name,
                    ideas=ideas,
                    bench_command=bench_command,
                    metric=metric,
                    work_dir=work_dir,
                    target_file=target_file,
                )

            def _create_individual(
                self,
                genes: dict[str, str | None],
            ) -> Individual:
                return Individual(
                    id=f"ind-{len(self.population)}",
                    genes={idea.name: genes.get(idea.name) for idea in self.ideas},
                )

            def _save(self) -> None:
                return None

            def eval(self, individual_id: str) -> EvalResult:
                try:
                    completed = subprocess.run(
                        self._bench_command,
                        shell=True,
                        cwd=self._work_dir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    metrics: dict[str, float] = {}
                    for line in completed.stdout.splitlines():
                        if ": " not in line:
                            continue
                        key, value = line.split(": ", 1)
                        try:
                            metrics[key] = float(value)
                        except ValueError:
                            continue
                    error = (
                        None
                        if completed.returncode == 0
                        else completed.stderr.strip() or "benchmark failed"
                    )
                    fitness = None if error is not None else metrics.get(self._metric)
                    return EvalResult(
                        fitness=fitness,
                        metrics=metrics,
                        error=error,
                    )
                finally:
                    self._target_path.write_text(self._baseline_text, encoding="utf-8")
        """,
    )
    return repo


@covers("M5-001")
def test_autoresearch_live_upstream_adapter_unit_exercise(tmp_path: Path) -> None:
    repo = _make_autoresearch_repo(tmp_path)
    adapter = AutoresearchUpstreamAdapter(_autoclanker_config("autoresearch", repo))

    probe = adapter.probe()
    baseline = adapter.evaluate_candidate(
        era_id="era_unit_001",
        candidate_id="cand_ar_baseline",
        genotype=adapter.build_registry().default_genotype(),
    )
    improved = adapter.evaluate_candidate(
        era_id="era_unit_001",
        candidate_id="cand_ar_improved",
        genotype=_genotype_from_mapping(
            adapter,
            {
                "train.depth": "depth_10",
                "train.window_pattern": "window_SSSL",
                "batch.total": "batch_2_18",
                "optim.matrix_lr": "lr_0_03",
                "schedule.warmup_ratio": "warmup_0_1",
            },
        ),
    )
    failure = adapter.evaluate_candidate(
        era_id="era_unit_001",
        candidate_id="cand_ar_failure",
        genotype=_genotype_from_mapping(
            adapter,
            {
                "train.depth": "depth_10",
                "train.window_pattern": "window_L",
                "batch.total": "batch_2_20",
                "optim.matrix_lr": "lr_0_05",
                "schedule.warmup_ratio": "warmup_0_0",
            },
        ),
    )
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    commit = adapter.commit_candidate("cand_ar_improved")
    train_text = (repo / "train.py").read_text(encoding="utf-8")

    assert probe.available is True
    assert probe.metadata is not None
    assert materialized["settings"] == {
        "train.depth": "depth_8",
        "train.window_pattern": "window_SSSL",
        "batch.total": "batch_2_19",
        "optim.matrix_lr": "lr_0_04",
        "schedule.warmup_ratio": "warmup_0_0",
    }
    assert _metric_float(baseline, "val_bpb") > _metric_float(improved, "val_bpb")
    assert improved.delta_perf >= 0.015
    assert (
        improved.raw_metrics["execution_backend"]
        == "repo_subprocess_heuristic_fallback"
    )
    assert improved.raw_metrics["metric_source"] == "local_heuristic"
    assert failure.status == "oom"
    assert failure.failure_metadata == {"reason": "simulated_hopper_vram_overage"}
    assert commit["applied"] is False
    assert "DEPTH = 8" in train_text
    assert Path(improved.artifact_paths[0]).exists()


@covers("M5-002")
def test_cevolve_live_upstream_adapter_unit_exercise(tmp_path: Path) -> None:
    repo = _make_cevolve_repo(tmp_path)
    adapter = CevolveUpstreamAdapter(_autoclanker_config("cevolve", repo))

    probe = adapter.probe()
    baseline = adapter.evaluate_candidate(
        era_id="era_unit_001",
        candidate_id="cand_ce_baseline",
        genotype=adapter.build_registry().default_genotype(),
        seed=3,
    )
    improved = adapter.evaluate_candidate(
        era_id="era_unit_001",
        candidate_id="cand_ce_improved",
        genotype=_genotype_from_mapping(
            adapter,
            {
                "sort.threshold": "threshold_32",
                "sort.partition": "partition_hoare",
                "sort.pivot": "pivot_median_of_three",
                "sort.iterative": "iterative_on",
            },
        ),
        seed=7,
    )
    materialized = adapter.materialize_candidate(
        adapter.build_registry().default_genotype()
    )
    commit = adapter.commit_candidate("cand_ce_improved")
    workspace_text = (
        Path(materialized["target_file"]).read_text(encoding="utf-8")
        if isinstance(materialized["target_file"], str)
        else ""
    )

    assert probe.available is True
    assert probe.metadata is not None
    assert materialized["settings"] == {
        "sort.threshold": "threshold_16",
        "sort.partition": "partition_lomuto",
        "sort.pivot": "pivot_median_of_three",
        "sort.iterative": "iterative_off",
    }
    assert _metric_float(baseline, "time_ms") > _metric_float(improved, "time_ms")
    assert improved.delta_perf >= 20.0
    assert improved.raw_metrics["execution_backend"] == "repo_benchmark_subprocess"
    assert improved.raw_metrics["metric_source"] == "subprocess_output"
    assert commit["applied"] is False
    assert "INSERTION_THRESHOLD = 16" in workspace_text
    assert Path(improved.artifact_paths[0]).exists()
    assert Path(improved.artifact_paths[1]).exists()


def test_live_upstream_probes_report_missing_files(tmp_path: Path) -> None:
    autoresearch_probe = AutoresearchUpstreamAdapter(
        _autoclanker_config("autoresearch", tmp_path / "missing-autoresearch")
    ).probe()
    cevolve_probe = CevolveUpstreamAdapter(
        _autoclanker_config("cevolve", tmp_path / "missing-cevolve")
    ).probe()

    assert autoresearch_probe.available is False
    assert cevolve_probe.available is False


def test_build_autoclanker_adapter_dispatch_and_errors(tmp_path: Path) -> None:
    autoresearch_repo = _make_autoresearch_repo(tmp_path / "dispatch")
    cevolve_repo = _make_cevolve_repo(tmp_path / "dispatch")

    assert isinstance(
        build_autoclanker_adapter(
            _autoclanker_config("autoresearch", autoresearch_repo)
        ),
        AutoresearchUpstreamAdapter,
    )
    assert isinstance(
        build_autoclanker_adapter(_autoclanker_config("cevolve", cevolve_repo)),
        CevolveUpstreamAdapter,
    )
    with pytest.raises(AdapterFailure):
        build_autoclanker_adapter(
            ValidAdapterConfig(
                kind="python_module",
                mode="auto",
                session_root=".autoclanker-live",
                repo_path=str(tmp_path / "unsupported"),
                allow_missing=False,
            )
        )


@covers("M5-001", "M5-002")
def test_first_party_live_wrappers_resolve_relative_repo_paths_from_config_base_dir(
    tmp_path: Path,
) -> None:
    autoresearch_repo = _make_autoresearch_repo(tmp_path / "workspace")
    cevolve_repo = _make_cevolve_repo(tmp_path / "workspace")
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    autoresearch_probe = AutoresearchAdapter(
        ValidAdapterConfig(
            kind="autoresearch",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path="../workspace/autoresearch",
            allow_missing=False,
            base_dir=str(config_dir.resolve()),
            metadata={"adapter_module": "autoclanker.bayes_layer.live_upstreams"},
        )
    ).probe()
    cevolve_probe = CevolveAdapter(
        ValidAdapterConfig(
            kind="cevolve",
            mode="auto",
            session_root=".autoclanker-live",
            repo_path="../workspace/cevolve",
            allow_missing=False,
            base_dir=str(config_dir.resolve()),
            metadata={"adapter_module": "autoclanker.bayes_layer.live_upstreams"},
        )
    ).probe()

    assert autoresearch_repo.exists()
    assert cevolve_repo.exists()
    assert autoresearch_probe.available is True
    assert cevolve_probe.available is True
    assert "Resolved real autoresearch checkout" in autoresearch_probe.detail
    assert "Resolved real cevolve checkout" in cevolve_probe.detail
