# clankerprof-v2-adversarial-review — requirements tracker

Single source of execution state for this goal loop. Every row is a checkable
requirement. `goalloop goal` exits 0 only when every row is `done` (or
`dropped` with a reason in Notes), every charter gate passes, and — when an
auditor is configured — the adversarial audit has converged.

Status vocabulary: `todo` | `doing` | `done` | `blocked` | `dropped`.

Checkpoint protocol: work lands in small clusters; each cluster is committed
only when its verification is green, flipping its rows to `done` in the same
commit. On interruption, resume from this file.

Audit rounds append `R<N>` waves below; those rows are the review's output
and this loop's real work.

## Wave V — validated baseline (the work under review)

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| V-01 | All 71 clankerprof v2 requirements implemented and gate-green on this branch | `python3 scripts/clankerprof_v2_goal.py --goal` exits 0 | done | 21 commits `b23528d^..e9a1c2c`; each landed behind a green `./bin/dev check` |
| V-02 | Baseline gates green in this audit worktree before round 1 | `goalloop gate` exits 0 | doing | fresh worktree: cold cargo target dir on first run |
