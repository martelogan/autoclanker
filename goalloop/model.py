"""File-backed state for goal loops: charter, tracker, audit log, history.

The loop contract is deliberately host-neutral: every artifact is a plain
file at the loop root, so any agent harness (Claude Code, Codex, pi) or a
human can read and continue a loop. The tracker is the single source of
execution state; the charter defines the goal, gates, and audit policy; the
audit log records adversarial rounds with confirmed findings and refutations;
the history is an append-only JSONL event stream.
"""

from __future__ import annotations

import hashlib
import json
import re

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, cast

import yaml

CHARTER_FILENAME: Final = "goalloop.charter.md"
TRACKER_FILENAME: Final = "goalloop.tracker.md"
AUDIT_FILENAME: Final = "goalloop.audit.md"
HISTORY_FILENAME: Final = "goalloop.history.jsonl"

VALID_STATUSES: Final = frozenset({"todo", "doing", "done", "blocked", "dropped"})
FINISHED_STATUSES: Final = frozenset({"done", "dropped"})
CONVERGENCE_POLICIES: Final = frozenset({"zero", "no_major"})
FINDING_SEVERITIES: Final = frozenset({"critical", "major", "minor"})
# Findings ingested without a severity count as major for convergence:
# an unlabeled finding must never be what lets the audit converge.
DEFAULT_FINDING_SEVERITY: Final = "major"
BLOCKING_SEVERITIES: Final = frozenset({"critical", "major"})
_AUDIT_KEYS: Final = frozenset({"auditor", "max_rounds", "convergence", "notes"})
REQUIREMENT_ID_RE: Final = re.compile(r"^[A-Z]+\d*-\d+$")
_TRACKER_HEADER_CELLS: Final = ("ID", "Requirement", "Verify", "Status", "Notes")
_WAVE_RE: Final = re.compile(r"^##\s+Wave\s+(\S+)")
_FRONTMATTER_RE: Final = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_AUDIT_ROUND_RE: Final = re.compile(r"^##\s+Round\s+(\d+)\b")


def _string_list() -> list[str]:
    return []


@dataclass(frozen=True, slots=True)
class Charter:
    name: str
    gates: tuple[str, ...]
    auditor: str | None
    max_audit_rounds: int
    audit_convergence: str
    audit_notes: str | None
    body: str

    @property
    def audit_enabled(self) -> bool:
        return self.auditor is not None


@dataclass(slots=True)
class Requirement:
    requirement_id: str
    requirement: str
    verify: str
    status: str
    notes: str

    @property
    def wave(self) -> str:
        return self.requirement_id.rsplit("-", 1)[0]

    @property
    def finished(self) -> bool:
        return self.status in FINISHED_STATUSES


@dataclass(slots=True)
class AuditRound:
    number: int
    confirmed_ids: list[str] = field(default_factory=_string_list)
    # Parallel to confirmed_ids; unlabeled findings read as the fail-closed
    # DEFAULT_FINDING_SEVERITY so legacy logs keep their exact semantics.
    confirmed_severities: list[str] = field(default_factory=_string_list)
    refuted_titles: list[str] = field(default_factory=_string_list)


@dataclass(frozen=True, slots=True)
class LoopPaths:
    root: Path

    @property
    def charter(self) -> Path:
        return self.root / CHARTER_FILENAME

    @property
    def tracker(self) -> Path:
        return self.root / TRACKER_FILENAME

    @property
    def audit(self) -> Path:
        return self.root / AUDIT_FILENAME

    @property
    def history(self) -> Path:
        return self.root / HISTORY_FILENAME

    def exists(self) -> bool:
        return self.charter.exists() and self.tracker.exists()


def load_charter(paths: LoopPaths) -> Charter:
    if not paths.charter.exists():
        raise ValueError(
            f"No goal loop found at {paths.root} (missing {CHARTER_FILENAME}); "
            "run goalloop init first."
        )
    text = paths.charter.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError(f"{CHARTER_FILENAME} must start with YAML frontmatter.")
    try:
        raw = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise ValueError(
            f"{CHARTER_FILENAME} frontmatter is not valid YAML: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{CHARTER_FILENAME} frontmatter must be a mapping.")
    payload = cast(dict[str, object], raw)
    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"{CHARTER_FILENAME} frontmatter requires a name.")
    raw_gates = payload.get("gates", [])
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError(
            f"{CHARTER_FILENAME} frontmatter requires a non-empty gates list."
        )
    gates = tuple(str(gate) for gate in cast(list[object], raw_gates))
    raw_audit = payload.get("audit")
    auditor: str | None = None
    max_rounds = 10
    convergence = "zero"
    notes: str | None = None
    if raw_audit is not None:
        if not isinstance(raw_audit, dict):
            raise ValueError(f"{CHARTER_FILENAME} audit block must be a mapping.")
        audit_payload = cast(dict[str, object], raw_audit)
        unknown = sorted(set(audit_payload) - _AUDIT_KEYS)
        if unknown:
            # A typo here would silently weaken the audit policy; fail closed.
            raise ValueError(
                f"{CHARTER_FILENAME} audit block has unknown key: {unknown[0]}. "
                f"Known keys: {', '.join(sorted(_AUDIT_KEYS))}."
            )
        raw_auditor = audit_payload.get("auditor")
        auditor = str(raw_auditor) if raw_auditor else None
        raw_rounds = audit_payload.get("max_rounds", 10)
        if isinstance(raw_rounds, bool) or not isinstance(raw_rounds, int):
            raise ValueError(f"{CHARTER_FILENAME} audit.max_rounds must be an integer.")
        max_rounds = raw_rounds
        raw_convergence = audit_payload.get("convergence", "zero")
        if raw_convergence not in CONVERGENCE_POLICIES:
            raise ValueError(
                f"{CHARTER_FILENAME} audit.convergence must be one of "
                f"{sorted(CONVERGENCE_POLICIES)}."
            )
        convergence = str(raw_convergence)
        raw_notes = audit_payload.get("notes")
        if raw_notes is not None:
            if not isinstance(raw_notes, str) or not raw_notes.strip():
                raise ValueError(
                    f"{CHARTER_FILENAME} audit.notes must be a non-empty string."
                )
            notes = raw_notes
    return Charter(
        name=name,
        gates=gates,
        auditor=auditor,
        max_audit_rounds=max_rounds,
        audit_convergence=convergence,
        audit_notes=notes,
        body=text[match.end() :],
    )


def load_requirements(paths: LoopPaths) -> list[Requirement]:
    if not paths.tracker.exists():
        raise ValueError(
            f"No goal loop tracker at {paths.tracker}; run goalloop init first."
        )
    rows: list[Requirement] = []
    for line in paths.tracker.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if tuple(cells) == _TRACKER_HEADER_CELLS:
            continue
        if all(cell and not set(cell) - {"-", ":"} for cell in cells):
            continue  # markdown alignment separator row
        if len(cells) != 5:
            raise ValueError(f"Tracker row must have 5 cells: {line!r}")
        if not REQUIREMENT_ID_RE.match(cells[0]):
            raise ValueError(
                f"Invalid requirement ID {cells[0]!r}: IDs must match "
                "<WAVE>-<NUMBER> like A-01 or R1-02 (uppercase wave token with "
                "optional trailing digits, a dash, then digits). Rows are never "
                "skipped silently — fix the ID or remove the row."
            )
        row = Requirement(
            requirement_id=cells[0],
            requirement=cells[1],
            verify=cells[2],
            status=cells[3],
            notes=cells[4],
        )
        if row.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {row.status!r} on {row.requirement_id}; "
                f"expected one of {sorted(VALID_STATUSES)}."
            )
        if row.status == "dropped" and not row.notes:
            raise ValueError(
                f"{row.requirement_id} is dropped without a reason in Notes."
            )
        rows.append(row)
    identifiers = [row.requirement_id for row in rows]
    duplicates = sorted({rid for rid in identifiers if identifiers.count(rid) > 1})
    if duplicates:
        raise ValueError(f"Duplicate requirement IDs: {', '.join(duplicates)}.")
    return rows


def waves_summary(rows: Iterable[Requirement]) -> list[dict[str, Any]]:
    waves: dict[str, list[Requirement]] = {}
    for row in rows:
        waves.setdefault(row.wave, []).append(row)
    return [
        {
            "wave": wave,
            "done": sum(1 for row in wave_rows if row.finished),
            "total": len(wave_rows),
            "pending": [
                {
                    "id": row.requirement_id,
                    "status": row.status,
                    "requirement": row.requirement,
                }
                for row in wave_rows
                if not row.finished
            ],
        }
        for wave, wave_rows in waves.items()
    ]


def load_audit_rounds(paths: LoopPaths) -> list[AuditRound]:
    if not paths.audit.exists():
        return []
    rounds: list[AuditRound] = []
    current: AuditRound | None = None
    for line in paths.audit.read_text(encoding="utf-8").splitlines():
        round_match = _AUDIT_ROUND_RE.match(line)
        if round_match:
            current = AuditRound(number=int(round_match.group(1)))
            rounds.append(current)
            continue
        if current is None:
            continue
        confirmed = re.match(
            r"^- CONFIRMED \[([A-Z]+\d*-\d+)\](?:\s+\((critical|major|minor)\))?",
            line,
        )
        if confirmed:
            current.confirmed_ids.append(confirmed.group(1))
            current.confirmed_severities.append(
                confirmed.group(2) or DEFAULT_FINDING_SEVERITY
            )
            continue
        refuted = re.match(r"^- REFUTED (.+?):", line)
        if refuted:
            current.refuted_titles.append(refuted.group(1))
    return rounds


def contract_digest(charter: Charter) -> str:
    """Digest of the charter's semantic contract (name, gates, audit policy).

    The prose body is deliberately excluded: it may evolve freely, but changing
    the definition of green mid-loop must be an explicit, recorded act.
    """
    audit_payload: dict[str, Any] | None = None
    if charter.audit_enabled:
        audit_payload = {
            "auditor": charter.auditor,
            "max_rounds": charter.max_audit_rounds,
        }
        # Non-default policy fields join the digest only when set, so loops
        # locked before these keys existed keep their digests; changing a
        # policy is still always a recorded drift + re-lock.
        if charter.audit_convergence != "zero":
            audit_payload["convergence"] = charter.audit_convergence
        if charter.audit_notes is not None:
            audit_payload["notes"] = charter.audit_notes
    payload: dict[str, Any] = {
        "audit": audit_payload,
        "gates": list(charter.gates),
        "name": charter.name,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def locked_contract_digest(paths: LoopPaths) -> str | None:
    """The most recently locked contract digest (init or explicit lock event)."""
    if not paths.history.exists():
        return None
    locked: str | None = None
    for line in paths.history.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = cast(dict[str, Any], json.loads(line))
        if event.get("event") in {"init", "lock"} and "contract_digest" in event:
            locked = str(event["contract_digest"])
    return locked


def audit_converged(charter: Charter, rounds: list[AuditRound]) -> bool:
    """Whether the last completed round satisfies the convergence policy.

    Policy "zero" (default): the round confirmed nothing new. Policy
    "no_major": the round confirmed nothing of blocking severity — a
    minor-only round converges (the lesson of large surfaces, where a
    max-effort auditor can surface esoteric minor findings indefinitely).
    Unlabeled findings count as DEFAULT_FINDING_SEVERITY (major).
    """
    if not charter.audit_enabled:
        return True
    if not rounds:
        return False
    last = rounds[-1]
    if charter.audit_convergence == "no_major":
        return not any(
            severity in BLOCKING_SEVERITIES for severity in last.confirmed_severities
        )
    return not last.confirmed_ids


def append_history(paths: LoopPaths, event: Mapping[str, Any]) -> None:
    with paths.history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(event), sort_keys=True) + "\n")


def render_tracker_row(row: Requirement) -> str:
    return (
        f"| {row.requirement_id} | {row.requirement} | {row.verify} "
        f"| {row.status} | {row.notes} |"
    )


def charter_template(
    name: str,
    gates: Iterable[str],
    auditor: str | None,
    max_audit_rounds: int,
    audit_convergence: str = "zero",
) -> str:
    frontmatter: dict[str, Any] = {
        "name": name,
        "gates": list(gates),
    }
    if auditor:
        audit_block: dict[str, Any] = {
            "auditor": auditor,
            "max_rounds": max_audit_rounds,
        }
        if audit_convergence != "zero":
            audit_block["convergence"] = audit_convergence
        frontmatter["audit"] = audit_block
    rendered = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    return f"""---
{rendered}
---

# Charter: {name}

State the goal to the bar of a good goal definition: a concrete outcome, the
evidence that proves it, measurable thresholds, explicit scope bounds, and
stop conditions that end the loop early.

## Outcome

<what will exist when this loop is done>

## Evidence

<which artifacts/tests/checks prove it — the tracker's Verify column cites these>

## Scope bounds

<what is explicitly out of scope>

## Stop conditions

<hard blockers that end the loop and escalate to a human instead of iterating>
"""


def tracker_template(name: str) -> str:
    return f"""# {name} — requirements tracker

Single source of execution state for this goal loop. Every row is a checkable
requirement. `goalloop goal` exits 0 only when every row is `done` (or
`dropped` with a reason in Notes), every charter gate passes, and — when an
auditor is configured — the adversarial audit has converged.

Status vocabulary: `todo` | `doing` | `done` | `blocked` | `dropped`.

Checkpoint protocol: work lands in small clusters; each cluster is committed
only when its verification is green, flipping its rows to `done` in the same
commit. On interruption, resume from this file.

## Wave A — <title>

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A-01 | <first requirement> | <deterministic verification> | todo | |
"""


HANDOFF_PROTOCOL: Final = """\
You are continuing a goal loop. Protocol:
1. Read the tracker below — it is the single source of execution state.
2. Pick the next small cluster of pending rows (prefer one wave at a time).
3. Implement, then run each row's verification, then run the charter gates
   capturing REAL exit codes (never pipe a gate into tail/head — that masks
   the exit code).
4. Commit the cluster with the tracker rows flipped to done in the SAME
   commit. Small clusters: an interruption should never lose more than one.
5. When all rows are done, run `goalloop goal`. If an auditor is configured,
   run the audit phase (`goalloop audit prompt`, triage findings by attempting
   reproduction, `goalloop audit ingest`) until it converges.
6. Stop conditions from the charter override everything: on a hard blocker,
   record it in the tracker Notes and escalate instead of iterating.
"""


AUDIT_PROMPT_TEMPLATE: Final = """\
You are an independent auditor for the goal loop "{name}". The implementing
agent believes the work is complete. Your job is to refute that.

Audit the repository at the current working directory, read-only. The charter
and requirement tracker follow. For every problem you believe you found, you
MUST attempt reproduction (run the failing case, cite exact file:line, show
the wrong output) — unreproduced suspicions must be labeled as such.

Do not re-raise previously refuted findings (the refutation log follows); if
you disagree with a refutation, present NEW evidence.

Output STRICT JSON only, matching:
{{"findings": [{{"title": str, "severity": "critical|major|minor",
"claim": str, "reproduction": str, "files": [str]}}], "summary": str}}

An empty findings list is a valid and welcome answer if the work holds up.

--- CHARTER ---
{charter}

--- TRACKER ---
{tracker}

--- PRIOR REFUTATIONS ---
{refutations}
"""
