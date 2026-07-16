from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import pytest

from autoclanker.cli import main as autoclanker_main
from goalloop.cli import main as goalloop_main
from goalloop.model import (
    AUDIT_FILENAME,
    HISTORY_FILENAME,
    TRACKER_FILENAME,
    LoopPaths,
    Requirement,
    audit_converged,
    load_audit_rounds,
    load_charter,
    load_requirements,
    render_tracker_row,
    waves_summary,
)
from tests.compliance import covers

TRACKER_HEADER = (
    "# demo — requirements tracker\n\n"
    "## Wave A — test\n\n"
    "| ID | Requirement | Verify | Status | Notes |\n"
    "| --- | --- | --- | --- | --- |\n"
)


def _init_loop(
    root: Path,
    *,
    gates: tuple[str, ...] = ("true",),
    auditor: str | None = None,
    max_rounds: int = 3,
) -> None:
    argv = ["init", "--name", "demo", "--root", str(root)]
    for gate in gates:
        argv.extend(["--gate", gate])
    if auditor is not None:
        argv.extend(["--auditor", auditor, "--max-audit-rounds", str(max_rounds)])
    assert goalloop_main(argv) == 0


def _write_tracker(root: Path, rows: list[str]) -> None:
    (root / TRACKER_FILENAME).write_text(
        TRACKER_HEADER + "\n".join(rows) + "\n", encoding="utf-8"
    )


def _last_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


@covers("M10-001")
def test_init_scaffolds_charter_tracker_and_history(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, gates=("true", "echo second"))
    payload = _last_json(capsys)
    assert payload["ok"] is True
    paths = LoopPaths(root=tmp_path)
    assert paths.exists()
    charter = load_charter(paths)
    assert charter.name == "demo"
    assert charter.gates == ("true", "echo second")
    assert charter.audit_enabled is False
    assert "## Outcome" in charter.body
    assert "## Stop conditions" in charter.body
    rows = load_requirements(paths)
    assert [row.requirement_id for row in rows] == ["A-01"]
    events = [
        json.loads(line)
        for line in (tmp_path / HISTORY_FILENAME).read_text().splitlines()
    ]
    assert events[0]["event"] == "init"


@covers("M10-001")
def test_init_refuses_to_overwrite_an_existing_loop(tmp_path: Path) -> None:
    _init_loop(tmp_path)
    assert goalloop_main(["init", "--name", "again", "--root", str(tmp_path)]) == 2


@covers("M10-001")
def test_init_defaults_to_a_trivially_green_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert goalloop_main(["init", "--name", "demo", "--root", str(tmp_path)]) == 0
    capsys.readouterr()
    assert load_charter(LoopPaths(root=tmp_path)).gates == ("true",)


@covers("M10-001")
def test_status_reports_waves_gates_and_audit_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    _write_tracker(
        tmp_path,
        [
            "| A-01 | first | pytest -k one | done | |",
            "| A-02 | second | pytest -k two | todo | |",
            "| B-01 | third | pytest -k three | dropped | superseded |",
        ],
    )
    capsys.readouterr()
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    payload = _last_json(capsys)
    assert payload["done"] == 2
    assert payload["total"] == 3
    waves = {wave["wave"]: wave for wave in payload["waves"]}
    assert waves["A"]["pending"] == [
        {"id": "A-02", "status": "todo", "requirement": "second"}
    ]
    assert waves["B"]["pending"] == []
    audit = payload["audit"]
    assert audit == {
        "enabled": True,
        "rounds": 0,
        "max_rounds": 3,
        "converged": False,
    }


@covers("M10-001")
def test_assert_fails_on_unfinished_ids_and_waves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    _write_tracker(
        tmp_path,
        [
            "| A-01 | first | check | done | |",
            "| A-02 | second | check | doing | |",
            "| B-01 | third | check | done | |",
        ],
    )
    capsys.readouterr()
    assert goalloop_main(["assert", "A-01", "B", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys) == {"ok": True, "not_done": [], "unknown": []}
    assert goalloop_main(["assert", "A", "--root", str(tmp_path)]) == 1
    assert _last_json(capsys)["not_done"] == [{"id": "A-02", "status": "doing"}]


@covers("M10-001")
def test_gate_propagates_the_real_exit_code_and_stops_at_first_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, gates=("echo pre", "exit 7", "echo never"))
    capsys.readouterr()
    assert goalloop_main(["gate", "--root", str(tmp_path)]) == 7
    payload = _last_json(capsys)
    assert payload["ok"] is False
    results = payload["results"]
    assert [result["gate"] for result in results] == ["echo pre", "exit 7"]
    assert results[0]["exit_code"] == 0
    assert "pre" in results[0]["output_tail"]
    assert results[1]["exit_code"] == 7


@covers("M10-001")
def test_goal_blocks_while_requirements_are_pending(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    capsys.readouterr()
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 1
    payload = _last_json(capsys)
    assert payload["reason"] == "requirements pending"
    assert payload["pending"] == ["A-01"]


@covers("M10-001")
def test_goal_passes_only_when_rows_finish_and_gates_succeed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, gates=("exit 3",))
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    capsys.readouterr()
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 3
    assert _last_json(capsys)["reason"] == "gate failed"

    charter_path = tmp_path / "goalloop.charter.md"
    charter_path.write_text(
        charter_path.read_text(encoding="utf-8").replace("exit 3", "true"),
        encoding="utf-8",
    )
    # Changing the gates changes the contract: goal refuses until re-locked.
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 1
    assert _last_json(capsys)["reason"] == "contract drifted"
    assert goalloop_main(["lock", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["changed"] is True

    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["reason"] == "goal met"
    events = [
        json.loads(line)
        for line in (tmp_path / HISTORY_FILENAME).read_text().splitlines()
    ]
    assert events[-1]["event"] == "goal"
    assert events[-1]["ok"] is True
    assert events[-1]["contract_digest"] == events[-2]["contract_digest"]


@covers("M10-003")
def test_handoff_emits_the_protocol_pending_rows_and_charter_body(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, gates=("./bin/dev check",))
    _write_tracker(
        tmp_path,
        [
            "| A-01 | first | check | done | |",
            "| A-02 | wire the adapter | check | todo | |",
        ],
    )
    capsys.readouterr()
    assert goalloop_main(["handoff", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "You are continuing a goal loop" in out
    assert "never pipe a gate into tail/head" in out
    assert "A-02 [todo] wire the adapter" in out
    assert "./bin/dev check" in out
    assert "## Stop conditions" in out

    assert goalloop_main(["handoff", "--json", "--root", str(tmp_path)]) == 0
    payload = _last_json(capsys)
    assert "protocol" in payload
    assert payload["status"]["done"] == 1


@covers("M10-002")
def test_audit_prompt_requires_an_auditor_and_embeds_loop_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    capsys.readouterr()
    assert goalloop_main(["audit", "prompt", "--root", str(tmp_path)]) == 2

    audit_root = tmp_path / "audited"
    _init_loop(audit_root, auditor="codex exec")
    _write_tracker(audit_root, ["| A-01 | first | check | done | |"])
    capsys.readouterr()
    assert goalloop_main(["audit", "prompt", "--root", str(audit_root)]) == 0
    out = capsys.readouterr().out
    assert "MUST attempt reproduction" in out
    assert "STRICT JSON" in out
    assert "| A-01 | first | check | done | |" in out
    assert "(none yet)" in out


@covers("M10-002")
def test_audit_ingest_appends_confirmed_waves_and_refutation_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    findings = tmp_path / "findings.json"
    findings.write_text(
        json.dumps(
            [
                {
                    "title": "boundary math off by one",
                    "verdict": "confirmed",
                    "evidence": "pytest -k boundary fails",
                },
                {
                    "title": "alleged race",
                    "verdict": "refuted",
                    "evidence": "single-threaded by construction",
                },
            ]
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 0
    )
    payload = _last_json(capsys)
    assert payload == {
        "ok": True,
        "round": 1,
        "confirmed": 1,
        "refuted": 1,
        "converged": False,
        "rounds_remaining": 2,
    }
    paths = LoopPaths(root=tmp_path)
    rows = {row.requirement_id: row for row in load_requirements(paths)}
    assert rows["R1-01"].requirement == "boundary math off by one"
    assert rows["R1-01"].status == "todo"
    audit_log = (tmp_path / AUDIT_FILENAME).read_text(encoding="utf-8")
    assert "## Round 1" in audit_log
    assert "- CONFIRMED [R1-01] boundary math off by one" in audit_log
    assert "- REFUTED alleged race: single-threaded by construction" in audit_log
    rounds = load_audit_rounds(paths)
    assert rounds[0].confirmed_ids == ["R1-01"]
    assert rounds[0].refuted_titles == ["alleged race"]

    # Goal stays blocked: the confirmed finding is now a pending requirement.
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 1
    assert _last_json(capsys)["pending"] == ["R1-01"]


@covers("M10-002")
def test_audit_converges_on_an_empty_round_and_unblocks_goal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    capsys.readouterr()

    # Audit enabled but no rounds yet: goal must block on convergence.
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 1
    assert _last_json(capsys)["reason"] == "audit not converged"

    findings = tmp_path / "findings.json"
    findings.write_text("[]", encoding="utf-8")
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 0
    )
    assert _last_json(capsys)["converged"] is True
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["reason"] == "goal met"

    assert goalloop_main(["audit", "status", "--root", str(tmp_path)]) == 0
    payload = _last_json(capsys)
    assert payload["converged"] is True
    assert payload["rounds"] == [{"round": 1, "confirmed": 0, "refuted": 0}]


@covers("M10-002")
def test_audit_ingest_enforces_the_round_budget_and_finding_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec", max_rounds=1)
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    findings = tmp_path / "findings.json"

    findings.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )

    findings.write_text(
        json.dumps([{"title": "no verdict", "evidence": "x"}]), encoding="utf-8"
    )
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )

    findings.write_text("[]", encoding="utf-8")
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 0
    )
    capsys.readouterr()
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )
    error = capsys.readouterr().err
    assert "escalate to a human" in error


@covers("M10-001")
def test_tracker_validation_rejects_malformed_rows(tmp_path: Path) -> None:
    _init_loop(tmp_path)
    paths = LoopPaths(root=tmp_path)

    _write_tracker(tmp_path, ["| A-01 | first | check | shipped | |"])
    with pytest.raises(ValueError, match="Invalid status"):
        load_requirements(paths)

    _write_tracker(tmp_path, ["| A-01 | first | check | dropped | |"])
    with pytest.raises(ValueError, match="dropped without a reason"):
        load_requirements(paths)

    _write_tracker(
        tmp_path,
        ["| A-01 | first | check | todo | |", "| A-01 | dup | check | todo | |"],
    )
    with pytest.raises(ValueError, match="Duplicate requirement IDs"):
        load_requirements(paths)

    _write_tracker(tmp_path, ["| A-01 | missing cells | todo |"])
    with pytest.raises(ValueError, match="5 cells"):
        load_requirements(paths)


@covers("M10-001")
def test_charter_validation_rejects_malformed_frontmatter(tmp_path: Path) -> None:
    paths = LoopPaths(root=tmp_path)
    with pytest.raises(ValueError, match="run goalloop init first"):
        load_charter(paths)
    with pytest.raises(ValueError, match="run goalloop init first"):
        load_requirements(paths)

    charter = tmp_path / "goalloop.charter.md"
    charter.write_text("# no frontmatter\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML frontmatter"):
        load_charter(paths)

    charter.write_text("---\n- just\n- a list\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_charter(paths)

    charter.write_text("---\ngates: [true]\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires a name"):
        load_charter(paths)

    charter.write_text("---\nname: demo\ngates: []\n---\nbody\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty gates"):
        load_charter(paths)

    charter.write_text(
        "---\nname: demo\ngates: [true]\naudit: nope\n---\nbody\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="audit block must be a mapping"):
        load_charter(paths)

    charter.write_text(
        "---\nname: demo\ngates: [true]\naudit:\n  auditor: codex\n"
        "  max_rounds: soon\n---\nbody\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_rounds must be an integer"):
        load_charter(paths)


@covers("M10-001")
def test_model_helpers_round_trip_rows_and_convergence(tmp_path: Path) -> None:
    row = Requirement(
        requirement_id="A2-03",
        requirement="do the thing",
        verify="pytest -k thing",
        status="todo",
        notes="",
    )
    assert row.wave == "A2"
    assert render_tracker_row(row) == (
        "| A2-03 | do the thing | pytest -k thing | todo |  |"
    )
    assert waves_summary([row])[0]["wave"] == "A2"

    _init_loop(tmp_path)
    charter = load_charter(LoopPaths(root=tmp_path))
    assert audit_converged(charter, []) is True  # audit disabled -> converged


@covers("M10-004")
def test_contract_lock_tracks_semantic_charter_changes_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    capsys.readouterr()
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    contract = _last_json(capsys)["contract"]
    assert contract["locked"] is True
    assert contract["drifted"] is False
    assert contract["digest"] == contract["locked_digest"]

    # Editing the prose body does not move the contract.
    charter_path = tmp_path / "goalloop.charter.md"
    charter_path.write_text(
        charter_path.read_text(encoding="utf-8").replace(
            "<what will exist when this loop is done>", "A shipped adapter."
        ),
        encoding="utf-8",
    )
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["contract"]["drifted"] is False

    # Editing the audit policy does.
    charter_path.write_text(
        charter_path.read_text(encoding="utf-8").replace(
            "max_rounds: 3", "max_rounds: 9"
        ),
        encoding="utf-8",
    )
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["contract"]["drifted"] is True
    assert goalloop_main(["gate", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["contract"]["drifted"] is True

    assert goalloop_main(["lock", "--root", str(tmp_path)]) == 0
    payload = _last_json(capsys)
    assert payload["changed"] is True
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["contract"]["drifted"] is False


@covers("M10-004")
def test_loops_without_a_recorded_lock_are_never_drifted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    (tmp_path / HISTORY_FILENAME).unlink()
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    capsys.readouterr()
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 0
    contract = _last_json(capsys)["contract"]
    assert contract["locked"] is False
    assert contract["drifted"] is False
    assert goalloop_main(["goal", "--root", str(tmp_path)]) == 0


@covers("M10-003")
def test_umbrella_alias_dispatches_goalloop_with_the_real_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    capsys.readouterr()
    with pytest.raises(SystemExit) as excinfo:
        autoclanker_main(["goalloop", "status", "--root", str(tmp_path)])
    assert excinfo.value.code == 0
    assert json.loads(capsys.readouterr().out)["name"] == "demo"

    with pytest.raises(SystemExit) as excinfo:
        autoclanker_main(["goalloop", "goal", "--root", str(tmp_path)])
    assert excinfo.value.code == 1


@covers("M10-003")
def test_goal_loop_skill_documents_the_portable_workflow() -> None:
    root = Path(__file__).resolve().parents[1]
    skill_dir = root / "skills" / "goal-loop"
    skill = skill_dir / "SKILL.md"
    agent = skill_dir / "agents" / "openai.yaml"

    assert skill.exists()
    assert agent.exists()
    rendered = skill.read_text(encoding="utf-8")
    for phrase in [
        "goalloop init",
        "goalloop handoff",
        "goalloop goal",
        "goalloop lock",
        "goalloop audit prompt",
        "goalloop audit ingest",
        "docs/GOALLOOP.md",
        "goalloop.tracker.md",
        "Claude Code",
        "Codex",
        "pi",
    ]:
        assert phrase in rendered


@covers("M10-001")
def test_tracker_rows_with_malformed_ids_error_instead_of_vanishing(
    tmp_path: Path,
) -> None:
    _init_loop(tmp_path)
    paths = LoopPaths(root=tmp_path)

    _write_tracker(
        tmp_path,
        [
            "| A-01 | first | check | done | |",
            "| A-CORE-01 | unfinished work | check | todo | |",
        ],
    )
    with pytest.raises(ValueError, match="Invalid requirement ID 'A-CORE-01'"):
        load_requirements(paths)

    _write_tracker(tmp_path, ["| a-01 | lowercase | check | todo | |"])
    with pytest.raises(ValueError, match="Invalid requirement ID"):
        load_requirements(paths)

    # Header and separator rows are still recognized, not treated as data.
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    assert [row.requirement_id for row in load_requirements(paths)] == ["A-01"]


@covers("M10-001")
def test_assert_fails_on_unknown_selectors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    capsys.readouterr()
    assert goalloop_main(["assert", "NOPE-99", "--root", str(tmp_path)]) == 1
    payload = _last_json(capsys)
    assert payload["ok"] is False
    assert payload["unknown"] == ["NOPE-99"]
    assert goalloop_main(["assert", "A-01", "--root", str(tmp_path)]) == 0
    assert _last_json(capsys)["unknown"] == []


@covers("M10-001")
def test_malformed_charter_yaml_and_missing_findings_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    charter_path = tmp_path / "goalloop.charter.md"
    original = charter_path.read_text(encoding="utf-8")

    charter_path.write_text(
        "---\nname: demo\ngates: [true\n---\nbody\n", encoding="utf-8"
    )
    capsys.readouterr()
    assert goalloop_main(["status", "--root", str(tmp_path)]) == 2
    assert "not valid YAML" in capsys.readouterr().err

    charter_path.write_text(original, encoding="utf-8")
    assert (
        goalloop_main(
            ["audit", "ingest", str(tmp_path / "nope.json"), "--root", str(tmp_path)]
        )
        == 2
    )
    assert "does not exist" in capsys.readouterr().err


@covers("M10-002")
def test_audit_ingest_rejects_tracker_corruption_and_wave_collisions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path, auditor="codex exec")
    _write_tracker(tmp_path, ["| A-01 | first | check | done | |"])
    findings = tmp_path / "findings.json"
    tracker_before = (tmp_path / TRACKER_FILENAME).read_text(encoding="utf-8")

    findings.write_text(
        json.dumps(
            [{"title": "pipe | injection", "verdict": "confirmed", "evidence": "x"}]
        ),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )
    assert "corrupt the tracker" in capsys.readouterr().err
    assert (tmp_path / TRACKER_FILENAME).read_text(encoding="utf-8") == tracker_before

    findings.write_text(
        json.dumps(
            [{"title": "multi\nline", "verdict": "refuted", "evidence": "proof"}]
        ),
        encoding="utf-8",
    )
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )
    capsys.readouterr()

    # A pre-existing R1 wave blocks round-1 confirmed ingestion outright.
    _write_tracker(
        tmp_path,
        [
            "| A-01 | first | check | done | |",
            "| R1-01 | manually created | check | done | |",
        ],
    )
    findings.write_text(
        json.dumps(
            [{"title": "real bug", "verdict": "confirmed", "evidence": "repro"}]
        ),
        encoding="utf-8",
    )
    assert (
        goalloop_main(["audit", "ingest", str(findings), "--root", str(tmp_path)]) == 2
    )
    assert "reserved for audit ingestion" in capsys.readouterr().err


@covers("M10-003")
def test_umbrella_rejects_global_output_flag_for_goalloop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _init_loop(tmp_path)
    capsys.readouterr()
    assert (
        autoclanker_main(
            [
                "--output",
                str(tmp_path / "copy.json"),
                "goalloop",
                "status",
                "--root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "shell redirection" in capsys.readouterr().err
    assert not (tmp_path / "copy.json").exists()
