---
name: clankerprof-v2-adversarial-review
gates:
- ./bin/dev check
- cargo test --workspace
audit:
  auditor: codex exec --sandbox read-only --cd /Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit
    -
  max_rounds: 3
---

# Charter: clankerprof-v2-adversarial-review

An independent adversarial review of the completed clankerprof v2 effort
(21 commits, `b23528d^..e9a1c2c` on branch `clankerprof-v2`), run as a
convergence loop: an external max-effort auditor (GPT-5.6-sol via
`codex exec`, read-only) tries to refute completion; the implementor
(Claude, with full context of the v2 effort) triages every finding by
attempting its reproduction, ingests the triage, and fixes confirmed
findings behind the standard gates until a round confirms nothing new.

## Outcome

The clankerprof v2 work survives (or is fixed to survive) adversarial
review along three lenses:

1. **The plan and requirements themselves** — are the 71 requirement rows
   in `docs/CLANKERPROF_V2_REQUIREMENTS.md` the right set for the goal
   stated in `docs/CLANKERPROF_V2_PLAN.md`? Were any mis-specified,
   under-specified, or satisfiable by gaming (vacuous tests, weakened
   assertions, doc-only "fixes")?
2. **The implementation against plan + requirements** — correctness of the
   semantic fixes (D1–D15 defect classes), the architecture split, the
   facts v2 contract, the Rust port, and the byte-parity claims between
   Python and Rust.
3. **The auditor's own independent findings** — bugs, design flaws, test
   blind spots, unhandled edge cases, or performance regressions the plan
   never anticipated, discovered by reading the code and running probes.

Done means: every confirmed finding is fixed (in both Python and Rust
where applicable) behind green gates, every refuted finding has recorded
proof, and an audit round completes with zero new confirmed findings.

## Evidence

- `docs/CLANKERPROF_V2_PLAN.md` — the plan, defect inventory, decisions,
  and benchmark appendices.
- `docs/CLANKERPROF_V2_REQUIREMENTS.md` — the 71-row requirements tracker;
  `python3 scripts/clankerprof_v2_goal.py --goal` exits 0.
- `docs/CLANKERPROF_SPEC.md` (normative contracts: sample_facts.v2,
  runtime_rules.v1, compare gate, CLI stream/error contract) and
  `docs/CLANKERPROF_PARITY.md` (capability matrix).
- The implementation: `clankerprof/`, `crates/clankerprof-core/`, their
  tests in `tests/` (notably `tests/test_clankerprof_rust_parity.py`,
  byte-level flag-matrix goldens) and `crates/clankerprof-core/src/`
  unit tests.
- This loop's gates: `./bin/dev check` (the full required CI gate) and
  `cargo test --workspace`.

## Scope bounds

- Audit surface is clankerprof only: the `clankerprof/` Python package,
  `crates/clankerprof-core/`, their tests, docs, rule packs, and the v2
  plan/requirements documents. `autoclanker` core, `bigbets`, and
  `goalloop` itself are out of scope (goalloop was reviewed separately).
- No history rewriting; fixes land as new commits on this audit branch.
- Any behavior fix must land in Python and Rust together, preserve the
  byte-parity contract (or update goldens + `docs/CLANKERPROF_PARITY.md`
  explicitly), and keep `./bin/dev check` deterministic and non-billed.
- No new runtime dependencies without recording the justification in the
  tracker Notes.
- The auditor runs read-only; only the implementor writes code, and only
  after reproducing a finding.

## Stop conditions

- Audit exceeds max_rounds (3) without convergence → stop and escalate to
  Logan with `goalloop audit status` and the open tracker rows.
- A confirmed finding requires a breaking contract change (e.g. a facts v3
  schema, changed CLI exit-code semantics, altered compare gate contract)
  → mark the row `blocked` with notes and escalate rather than change a
  published contract unilaterally.
- Gate failures that are environmental (toolchain, network, cargo cache)
  rather than caused by the work → record in Notes, fix the environment,
  never weaken a gate to pass.
- Usage-limit interruption is NOT a stop condition: all state lives in the
  goalloop files and the worktree; resume from the tracker.
