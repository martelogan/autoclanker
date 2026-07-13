#!/usr/bin/env python3
"""Deterministic goal checker for the clankerprof v2 effort.

Parses docs/CLANKERPROF_V2_REQUIREMENTS.md and reports/enforces completion.

Modes:
  --status            (default) print per-wave progress; always exits 0
  --assert-done IDS   exit 1 unless every named row/wave prefix is done/dropped
  --gate              run ./bin/dev check and cargo test --workspace; exit nonzero on failure
  --goal              assert ALL rows done/dropped, then run the gate
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACKER = REPO / "docs" / "CLANKERPROF_V2_REQUIREMENTS.md"
VALID_STATUS = {"todo", "doing", "done", "blocked", "dropped"}
ROW_RE = re.compile(r"^\|\s*([A-Z]+\d*-\d+)\s*\|")


def parse_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in TRACKER.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            raise SystemExit(f"malformed row (need 5 cells): {line!r}")
        row = {
            "id": cells[0],
            "requirement": cells[1],
            "verify": cells[2],
            "status": cells[3],
            "notes": cells[4],
        }
        if row["status"] not in VALID_STATUS:
            raise SystemExit(f"invalid status {row['status']!r} on {row['id']}")
        if row["status"] == "dropped" and not row["notes"]:
            raise SystemExit(f"{row['id']} is dropped without a reason in Notes")
        rows.append(row)
    if not rows:
        raise SystemExit(f"no requirement rows parsed from {TRACKER}")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({rid for rid in ids if ids.count(rid) > 1})
        raise SystemExit(f"duplicate requirement IDs: {dupes}")
    return rows


def wave_of(row_id: str) -> str:
    return row_id.rsplit("-", 1)[0]


def print_status(rows: list[dict[str, str]]) -> None:
    waves: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        waves.setdefault(wave_of(row["id"]), []).append(row)
    total_done = 0
    for wave, wave_rows in waves.items():
        done = sum(1 for r in wave_rows if r["status"] in {"done", "dropped"})
        total_done += done
        marker = "OK " if done == len(wave_rows) else "   "
        print(f"{marker}{wave:4} {done}/{len(wave_rows)}")
        for row in wave_rows:
            if row["status"] not in {"done", "dropped"}:
                print(f"       {row['id']} [{row['status']}] {row['requirement'][:80]}")
    print(f"TOTAL {total_done}/{len(rows)}")


def assert_done(rows: list[dict[str, str]], selectors: list[str]) -> int:
    failing = [
        row
        for row in rows
        if any(row["id"] == sel or wave_of(row["id"]) == sel for sel in selectors)
        and row["status"] not in {"done", "dropped"}
    ]
    for row in failing:
        print(f"NOT DONE: {row['id']} [{row['status']}] {row['requirement']}")
    return 1 if failing else 0


def run_gate() -> int:
    for cmd in (["./bin/dev", "check"], ["cargo", "test", "--workspace"]):
        print(f"gate: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=REPO, check=False)
        if result.returncode != 0:
            print(f"GATE FAILED ({result.returncode}): {' '.join(cmd)}")
            return result.returncode
    print("GATE PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--assert-done", nargs="+", metavar="ID_OR_WAVE")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--goal", action="store_true")
    args = parser.parse_args()

    rows = parse_rows()
    if args.goal:
        pending = [r for r in rows if r["status"] not in {"done", "dropped"}]
        if pending:
            print(f"GOAL NOT MET: {len(pending)} requirement(s) pending")
            print_status(rows)
            return 1
        return run_gate()
    if args.assert_done:
        return assert_done(rows, args.assert_done)
    if args.gate:
        return run_gate()
    print_status(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
