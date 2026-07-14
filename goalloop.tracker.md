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
| V-02 | Baseline gates green in this audit worktree before round 1 | `goalloop gate` exits 0 | done | first run hit a transient sdist Errno 2 (worktree startup contention); standalone build + full gate rerun green |

## Wave R1 — audit round 1 findings

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| R1-01 | Compare accepts truncated or corrupt reports and fails open | Reproduced. Rust coerces non-numeric pct to 0.0 -> false-green exit 0 where Python exits 2 (parity break, fail-open gate); missing slices/boundaries key accepted as empty in both. Full evidence: .goalloop-support/round1.triage.json#F1 | done | audit round 1; fixed: missing slices/boundaries arrays and non-numeric pct/total fields exit 2 with identical envelopes in both languages |
| R1-02 | NaN thresholds disable regression gating in both implementations | Reproduced in both languages: --threshold-abs/rel nan turns a real 10->20pct regression into stable/exit 0. No finiteness validation anywhere. #F2 | done | audit round 1; fixed: non-finite thresholds rejected at option validation (exit 2 envelope) in both languages |
| R1-03 | Duplicate function names overwrite frame data in compare reports | Reproduced byte-for-byte in both: compare keys frames by function only, overwriting (f,/one)+(f,/two); +5 regression vanishes from top_regressions. #F3 | done | audit round 1; fixed: per-function frame pcts now sum across duplicates in both languages, byte-identically |
| R1-04 | Facts-v2 numeric handling rejects valid IDs, coerces floats, and can panic | Reproduced, worse than claimed: both exporters emit u64 IDs > i64::MAX from real .pb, so Rust rejects its own export (round-trip violation, misleading error); Python truncates 7.9->7; Rust i64 sum overflow panics (debug). #F4 | todo | audit round 1 |
| R1-05 | The promised JSON error boundary does not cover CLI usage or Python conversion errors | Reproduced: argparse/clap usage errors print prose (exit 2, no JSON envelope); Python OverflowError from top:.inf escapes as traceback exit 1. #F5 | todo | audit round 1 |
| R1-06 | `--output` violates the documented receipt and global-option contract | Reproduced: Python targets --format json --output prints full artifact instead of receipt; Rust prints nothing on all --output writes; compare accepts no --output at all in either language despite spec. #F6 | todo | audit round 1 |
| R1-07 | Published Rust CLI parity omits tracked slice-config and compare-alias surfaces | Reproduced: Rust lacks slices --config and compare --focus-scopes despite PARITY.md claims and B3-05 done; tracker note admits deferral. #F7 | todo | audit round 1 |
| R1-08 | Rust silently ignores invalid scope predicates and `--by-slice` values | Reproduced: Rust .ok().flatten() drops configured cost-kind predicate errors (all cost -> Other, exit 0) and ignores unparsable --by-slice; Python exits 2 on both. #F8 | todo | audit round 1 |
| R1-09 | Rust path/glob matching is not compatible with Python | Reproduced: Rust tokenizes [ as glob but escapes brackets in regex conversion -> path:/app/[ab].rb matches in Python (Application), never in Rust (Other); also affects scopes path predicates. #F9 | todo | audit round 1 |
| R1-10 | Rust silently drops later members from valid concatenated gzip profiles | Reproduced: Rust GzDecoder reads only first gzip member; Python decodes all. Valid RFC1952 multi-member profile -> Rust silently returns empty profile, exit 0. #F10 | todo | audit round 1 |
