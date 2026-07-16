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
    DEFAULT_FINDING_SEVERITY,
    FINDING_SEVERITIES,
    HANDOFF_PROTOCOL,
    Charter,
    LoopPaths,
    Requirement,
    append_history,
    audit_converged,
    charter_template,
    contract_digest,
    load_audit_rounds,
    load_charter,
    load_requirements,
    locked_contract_digest,
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
        charter_template(
            args.name,
            gates,
            args.auditor,
            args.max_audit_rounds,
            audit_convergence=args.audit_convergence,
        ),
        encoding="utf-8",
    )
    paths.tracker.write_text(tracker_template(args.name), encoding="utf-8")
    digest = contract_digest(load_charter(paths))
    append_history(
        paths,
        {
            "event": "init",
            "name": args.name,
            "gates": gates,
            "contract_digest": digest,
        },
    )
    _emit(
        {
            "ok": True,
            "charter": str(paths.charter),
            "tracker": str(paths.tracker),
            "contract_digest": digest,
            "next": "Edit the charter and tracker, re-lock the contract with "
            "goalloop lock, then drive the loop with goalloop handoff / "
            "status / goal.",
        }
    )
    return 0


def _contract_state(charter: Charter, paths: LoopPaths) -> dict[str, Any]:
    digest = contract_digest(charter)
    locked = locked_contract_digest(paths)
    return {
        "digest": digest,
        "locked_digest": locked,
        "locked": locked is not None,
        "drifted": locked is not None and locked != digest,
    }


def run_lock(args: argparse.Namespace) -> int:
    paths = _paths(args)
    digest = contract_digest(load_charter(paths))
    previous = locked_contract_digest(paths)
    append_history(paths, {"event": "lock", "contract_digest": digest})
    _emit(
        {
            "ok": True,
            "contract_digest": digest,
            "previous": previous,
            "changed": previous != digest,
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
        "contract": _contract_state(charter, paths),
        "audit": {
            "enabled": charter.audit_enabled,
            "rounds": len(rounds),
            "max_rounds": charter.max_audit_rounds,
            "convergence": charter.audit_convergence,
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
    known = {row.requirement_id for row in rows} | {row.wave for row in rows}
    unknown = sorted(selectors - known)
    failing = [
        row
        for row in rows
        if (row.requirement_id in selectors or row.wave in selectors)
        and not row.finished
    ]
    _emit(
        {
            "ok": not failing and not unknown,
            "not_done": [
                {"id": row.requirement_id, "status": row.status} for row in failing
            ],
            "unknown": unknown,
        }
    )
    return 1 if failing or unknown else 0


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
    _emit(
        {
            "ok": exit_code == 0,
            "results": results,
            "contract": _contract_state(charter, paths),
        }
    )
    return exit_code


def run_goal(args: argparse.Namespace) -> int:
    paths = _paths(args)
    charter = load_charter(paths)
    contract = _contract_state(charter, paths)
    if contract["drifted"]:
        _emit(
            {
                "ok": False,
                "reason": "contract drifted",
                "contract": contract,
                "next": "The charter's name/gates/audit policy changed after the "
                "last lock. Re-lock intentionally with goalloop lock.",
            }
        )
        return 1
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
        {
            "event": "goal",
            "ok": exit_code == 0,
            "pending": 0,
            "rounds": len(rounds),
            "contract_digest": contract["digest"],
        },
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
    prompt = AUDIT_PROMPT_TEMPLATE.format(
        name=charter.name,
        charter=paths.charter.read_text(encoding="utf-8"),
        tracker=paths.tracker.read_text(encoding="utf-8"),
        refutations=refutations,
    )
    if charter.audit_notes is not None:
        # Charter-carried operator guidance (scope carve-outs, tooling
        # constraints of the auditor's platform); part of the locked policy.
        prompt += f"\n--- OPERATOR NOTES ---\n{charter.audit_notes}\n"
    print(prompt)
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
    findings_path = Path(str(args.findings))
    if not findings_path.exists():
        raise ValueError(f"Triaged findings file does not exist: {findings_path}")
    raw = json.loads(findings_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Triaged findings must be a JSON array.")
    confirmed: list[dict[str, str | None]] = []
    refuted: list[dict[str, str | None]] = []
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
        raw_severity = finding.get("severity")
        severity: str | None = None
        if raw_severity is not None:
            if raw_severity not in FINDING_SEVERITIES:
                raise ValueError(
                    f"Finding severity must be one of {sorted(FINDING_SEVERITIES)}; "
                    f"got {raw_severity!r}. Omit it to count as "
                    f"{DEFAULT_FINDING_SEVERITY} for convergence."
                )
            severity = str(raw_severity)
        for label, value in (("title", title), ("evidence", evidence)):
            if "|" in value or "\n" in value or "\r" in value:
                raise ValueError(
                    f"Finding {label} must not contain '|' or newlines (they "
                    f"would corrupt the tracker table): {value!r}"
                )
        (confirmed if verdict == "confirmed" else refuted).append(
            {"title": title, "evidence": evidence, "severity": severity}
        )

    wave = f"R{round_number}"
    existing_waves = {row.wave for row in load_requirements(paths)}
    if confirmed and wave in existing_waves:
        raise ValueError(
            f"Tracker already contains wave {wave}; R<N> waves are reserved "
            "for audit ingestion and each round may only be ingested once."
        )
    new_rows: list[str] = []
    audit_lines: list[str] = [f"\n## Round {round_number}\n"]
    for index, finding in enumerate(confirmed, start=1):
        requirement_id = f"{wave}-{index:02d}"
        severity = finding["severity"]
        severity_note = f" ({severity})" if severity else ""
        new_rows.append(
            f"| {requirement_id} | {finding['title']} "
            f"| {finding['evidence']} | todo "
            f"| audit round {round_number}{severity_note} |"
        )
        audit_lines.append(
            f"- CONFIRMED [{requirement_id}]{severity_note} "
            f"{finding['title']}: {finding['evidence']}"
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
    # Recompute from the files so the reported convergence is exactly what
    # goal will see (policy-aware, and proof the writer/parser agree).
    converged = audit_converged(charter, load_audit_rounds(paths))
    _emit(
        {
            "ok": True,
            "round": round_number,
            "confirmed": len(confirmed),
            "refuted": len(refuted),
            "converged": converged,
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
            "convergence": charter.audit_convergence,
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
    init.add_argument("--max-audit-rounds", type=int, default=10)
    init.add_argument(
        "--audit-convergence",
        choices=("zero", "no_major"),
        default="zero",
        help="Convergence policy: zero = a round confirming nothing new; "
        "no_major = a round confirming nothing critical/major (large "
        "surfaces where esoteric minor findings never dry up).",
    )
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

    lock = subparsers.add_parser(
        "lock",
        help="Record the current charter contract (name/gates/audit policy) "
        "digest as the locked definition of done.",
    )
    _add_root(lock)
    _set(lock, run_lock)

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
