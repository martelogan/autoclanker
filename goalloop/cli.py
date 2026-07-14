"""Host-neutral goal-loop CLI: deterministic convergence for agent loops.

Every command is non-interactive and prints JSON to stdout (except `handoff`
and `audit prompt`, whose text output IS the artifact). Exit codes: 0 success,
1 goal/assert not met or gate failure (propagated), 2 validation error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

from goalloop.model import (
    AUDIT_PROMPT_TEMPLATE,
    HANDOFF_PROTOCOL,
    Charter,
    LoopPaths,
    Requirement,
    append_history,
    audit_converged,
    charter_template,
    load_audit_rounds,
    load_charter,
    load_requirements,
    tracker_template,
    waves_summary,
)

EXIT_VALIDATION_ERROR = 2


def _paths(args: argparse.Namespace) -> LoopPaths:
    return LoopPaths(root=Path(str(args.root)).resolve())


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def run_init(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if paths.charter.exists() or paths.tracker.exists():
        raise ValueError(
            f"A goal loop already exists at {paths.root}; refusing to overwrite."
        )
    gates = list(args.gate) if args.gate else ["true"]
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.charter.write_text(
        charter_template(args.name, gates, args.auditor, args.max_audit_rounds),
        encoding="utf-8",
    )
    paths.tracker.write_text(tracker_template(args.name), encoding="utf-8")
    append_history(paths, {"event": "init", "name": args.name, "gates": gates})
    _emit(
        {
            "ok": True,
            "charter": str(paths.charter),
            "tracker": str(paths.tracker),
            "next": "Edit the charter and tracker, then drive the loop with "
            "goalloop handoff / status / goal.",
        }
    )
    return 0


def _status_payload(
    charter: Charter,
    rows: list[Requirement],
    paths: LoopPaths,
) -> dict[str, Any]:
    rounds = load_audit_rounds(paths)
    return {
        "name": charter.name,
        "waves": waves_summary(rows),
        "done": sum(1 for row in rows if row.finished),
        "total": len(rows),
        "gates": list(charter.gates),
        "audit": {
            "enabled": charter.audit_enabled,
            "rounds": len(rounds),
            "max_rounds": charter.max_audit_rounds,
            "converged": audit_converged(charter, rounds),
        },
    }


def run_status(args: argparse.Namespace) -> int:
    paths = _paths(args)
    _emit(_status_payload(load_charter(paths), load_requirements(paths), paths))
    return 0


def run_assert(args: argparse.Namespace) -> int:
    paths = _paths(args)
    rows = load_requirements(paths)
    selectors = set(cast(list[str], args.selectors))
    failing = [
        row
        for row in rows
        if (row.requirement_id in selectors or row.wave in selectors)
        and not row.finished
    ]
    _emit(
        {
            "ok": not failing,
            "not_done": [
                {"id": row.requirement_id, "status": row.status} for row in failing
            ],
        }
    )
    return 1 if failing else 0


def _run_gates(charter: Charter, paths: LoopPaths) -> tuple[int, list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    for gate in charter.gates:
        completed = subprocess.run(
            gate,
            shell=True,
            cwd=paths.root,
            check=False,
            capture_output=True,
            text=True,
        )
        results.append(
            {
                "gate": gate,
                "exit_code": completed.returncode,
                "output_tail": (completed.stdout + completed.stderr)[-2000:],
            }
        )
        if completed.returncode != 0:
            return completed.returncode, results
    return 0, results


def run_gate(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    exit_code, results = _run_gates(charter, paths)
    _emit({"ok": exit_code == 0, "results": results})
    return exit_code


def run_goal(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    rows = load_requirements(paths)
    pending = [row.requirement_id for row in rows if not row.finished]
    rounds = load_audit_rounds(paths)
    converged = audit_converged(charter, rounds)
    if pending:
        _emit({"ok": False, "reason": "requirements pending", "pending": pending})
        return 1
    if charter.audit_enabled and not converged:
        _emit(
            {
                "ok": False,
                "reason": "audit not converged",
                "audit_rounds": len(rounds),
            }
        )
        return 1
    exit_code, results = _run_gates(charter, paths)
    payload = {
        "ok": exit_code == 0,
        "reason": "goal met" if exit_code == 0 else "gate failed",
        "results": results,
    }
    _emit(payload)
    append_history(
        paths,
        {"event": "goal", "ok": exit_code == 0, "pending": 0, "rounds": len(rounds)},
    )
    return exit_code


def run_handoff(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    rows = load_requirements(paths)
    status = _status_payload(charter, rows, paths)
    if args.json:
        _emit({"protocol": HANDOFF_PROTOCOL, "status": status})
        return 0
    pending_lines = [
        f"  {item['id']} [{item['status']}] {item['requirement']}"
        for wave in cast(list[dict[str, Any]], status["waves"])
        for item in cast(list[dict[str, Any]], wave["pending"])
    ]
    pending_text = "\n".join(pending_lines) if pending_lines else "  (none)"
    print(
        f"{HANDOFF_PROTOCOL}\n"
        f"Loop: {charter.name} — {status['done']}/{status['total']} done. "
        f"Tracker: {paths.tracker}\n"
        f"Charter gates: {', '.join(charter.gates)}\n"
        f"Pending rows:\n{pending_text}\n\n"
        f"Charter body follows.\n{charter.body}"
    )
    return 0


def run_audit_prompt(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    if not charter.audit_enabled:
        raise ValueError("No auditor configured in the charter's audit block.")
    refutations = (
        paths.audit.read_text(encoding="utf-8")
        if paths.audit.exists()
        else "(none yet)"
    )
    print(
        AUDIT_PROMPT_TEMPLATE.format(
            name=charter.name,
            charter=paths.charter.read_text(encoding="utf-8"),
            tracker=paths.tracker.read_text(encoding="utf-8"),
            refutations=refutations,
        )
    )
    return 0


def run_audit_ingest(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    if not charter.audit_enabled:
        raise ValueError("No auditor configured in the charter's audit block.")
    rounds = load_audit_rounds(paths)
    round_number = len(rounds) + 1
    if round_number > charter.max_audit_rounds:
        raise ValueError(
            f"Audit round {round_number} exceeds max_rounds "
            f"{charter.max_audit_rounds}; escalate to a human."
        )
    raw = json.loads(Path(str(args.findings)).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Triaged findings must be a JSON array.")
    confirmed: list[dict[str, str]] = []
    refuted: list[dict[str, str]] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise ValueError("Each triaged finding must be an object.")
        finding = cast(dict[str, Any], item)
        verdict = str(finding.get("verdict", ""))
        title = str(finding.get("title", "")).strip()
        evidence = str(finding.get("evidence", "")).strip()
        if not title or verdict not in {"confirmed", "refuted"} or not evidence:
            raise ValueError(
                "Each finding needs title, verdict (confirmed|refuted), and "
                "evidence (the reproduction, or the refutation proof)."
            )
        (confirmed if verdict == "confirmed" else refuted).append(
            {"title": title, "evidence": evidence}
        )

    wave = f"R{round_number}"
    new_rows: list[str] = []
    audit_lines: list[str] = [f"\n## Round {round_number}\n"]
    for index, finding in enumerate(confirmed, start=1):
        requirement_id = f"{wave}-{index:02d}"
        new_rows.append(
            f"| {requirement_id} | {finding['title']} "
            f"| {finding['evidence']} | todo | audit round {round_number} |"
        )
        audit_lines.append(
            f"- CONFIRMED [{requirement_id}] {finding['title']}: {finding['evidence']}"
        )
    for finding in refuted:
        audit_lines.append(f"- REFUTED {finding['title']}: {finding['evidence']}")

    if new_rows:
        with paths.tracker.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Wave {wave} — audit round {round_number} findings\n\n"
                "| ID | Requirement | Verify | Status | Notes |\n"
                "| --- | --- | --- | --- | --- |\n" + "\n".join(new_rows) + "\n"
            )
    with paths.audit.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(audit_lines) + "\n")
    append_history(
        paths,
        {
            "event": "audit_ingest",
            "round": round_number,
            "confirmed": len(confirmed),
            "refuted": len(refuted),
        },
    )
    _emit(
        {
            "ok": True,
            "round": round_number,
            "confirmed": len(confirmed),
            "refuted": len(refuted),
            "converged": not confirmed,
            "rounds_remaining": charter.max_audit_rounds - round_number,
        }
    )
    return 0


def run_audit_status(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    rounds = load_audit_rounds(paths)
    _emit(
        {
            "enabled": charter.audit_enabled,
            "auditor": charter.auditor,
            "rounds": [
                {
                    "round": item.number,
                    "confirmed": len(item.confirmed_ids),
                    "refuted": len(item.refuted_titles),
                }
                for item in rounds
            ],
            "max_rounds": charter.max_audit_rounds,
            "converged": audit_converged(charter, rounds),
        }
    )
    return 0


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Loop root directory.")


def register_goalloop_commands(
    subparsers: Any,
    *,
    wrap: Any = None,
) -> None:
    """Register the goalloop family; hosts that expect dict-returning handlers
    (like the umbrella CLI) pass ``wrap`` to adapt the int-returning handlers.
    """

    def _set(parser: argparse.ArgumentParser, handler: Any) -> None:
        parser.set_defaults(handler=wrap(handler) if wrap else handler)

    init = subparsers.add_parser("init", help="Scaffold a goal loop.")
    init.add_argument("--name", required=True)
    init.add_argument(
        "--gate",
        action="append",
        help="Gate command (repeatable); the deterministic definition of green.",
    )
    init.add_argument(
        "--auditor",
        help="Command that runs an independent read-only audit (e.g. codex exec).",
    )
    init.add_argument("--max-audit-rounds", type=int, default=3)
    _add_root(init)
    _set(init, run_init)

    status = subparsers.add_parser("status", help="JSON progress summary.")
    _add_root(status)
    _set(status, run_status)

    assert_parser = subparsers.add_parser(
        "assert", help="Exit 1 unless the named rows or waves are finished."
    )
    assert_parser.add_argument("selectors", nargs="+", metavar="ID_OR_WAVE")
    _add_root(assert_parser)
    _set(assert_parser, run_assert)

    gate = subparsers.add_parser("gate", help="Run the charter gates.")
    _add_root(gate)
    _set(gate, run_gate)

    goal = subparsers.add_parser(
        "goal",
        help="Exit 0 only when all rows are finished, gates pass, and any "
        "configured audit has converged.",
    )
    _add_root(goal)
    _set(goal, run_goal)

    handoff = subparsers.add_parser(
        "handoff", help="Emit the next-iteration prompt for any agent harness."
    )
    handoff.add_argument("--json", action="store_true")
    _add_root(handoff)
    _set(handoff, run_handoff)

    audit = subparsers.add_parser("audit", help="Adversarial audit phase.")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    prompt = audit_sub.add_parser(
        "prompt", help="Emit the auditor charter prompt (feed to the auditor)."
    )
    _add_root(prompt)
    _set(prompt, run_audit_prompt)
    ingest = audit_sub.add_parser(
        "ingest",
        help="Ingest triaged findings JSON: confirmed -> new tracker rows; "
        "refuted -> the refutation log.",
    )
    ingest.add_argument("findings", help="Path to triaged findings JSON array.")
    _add_root(ingest)
    _set(ingest, run_audit_ingest)
    audit_status = audit_sub.add_parser("status", help="Audit convergence state.")
    _add_root(audit_status)
    _set(audit_status, run_audit_status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goalloop",
        description=(
            "Deterministic goal loops for agent harnesses: charter + tracker "
            "+ gates + handoff prompts + adversarial audit convergence."
        ),
    )
    register_goalloop_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        handler = cast("Callable[[argparse.Namespace], int]", args.handler)
        return handler(args)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return EXIT_VALIDATION_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
