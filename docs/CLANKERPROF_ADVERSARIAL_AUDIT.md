# clankerprof v2 adversarial audit record

This document is the durable record of the adversarial convergence review
that hardened clankerprof v2 after the initial 21-cluster effort shipped.
The raw loop artifacts (charter, 112-row tracker, append-only audit log,
per-round auditor prompts, findings, and reproduction-triage evidence)
are deliberately not part of `main`'s tree; the complete record is
preserved under the annotated tag `clankerprof-v2-audit-loop`.

## Purpose

An independent, maximally adversarial review of the completed v2 work
along three lenses: the plan and requirements themselves (mis-specified or
gameable rows), the implementation against them (semantic fixes, facts
contract, Rust port, byte-parity claims), and the auditor's own
independent findings — run as a convergence loop rather than a one-shot
review, so every confirmed finding was fixed and re-audited.

## Mechanics

- The loop was driven by `goalloop` (see `docs/GOALLOOP.md`): charter
  gates `./bin/dev check` and `cargo test --workspace`, findings ingested
  as `R<N>` tracker waves, refutations recorded so later rounds could not
  re-raise them without new evidence.
- Reviewer: `codex exec` (GPT-5.6-sol, max reasoning) in a read-only
  sandbox, emitting strict-JSON findings with mandatory reproductions.
- Implementor: Claude, with the constraint that no finding was believed
  until independently re-reproduced (exact commands, both languages,
  against the cited spec text) and no fix landed without the full gates.
- Every behavior change landed in Python and Rust together; the parity
  suite pinned byte-identical artifacts or byte-identical error envelopes
  for every fix (~60 new cross-language tests over the ten rounds).

## The numbers

- **10 rounds**, findings per round: 10, 14, 8, 14, 11, 12, 8, 14, 5, 5.
- **101 findings confirmed** by independent reproduction — all fixed.
- **1 finding refuted** with recorded proof (the compare gate's
  percentage trust model is the documented contract, not a fail-open).
- **2 auditor-adjacent false alarms** disproven empirically and pinned as
  verified non-divergences instead of "fixed" (a rejection guard would
  have created the divergence it claimed to close).
- Three proactive hardening waves closed sibling defects before the
  auditor could raise them.

## Defect-class themes

1. **Compare-gate fail-opens** — non-numeric and missing fields coerced
   to zero, NaN/negative thresholds disabling gating, duplicate rows and
   duplicate JSON members making the gate order-dependent, unknown focus
   names silently matching nothing, signed rows never able to regress.
2. **Numeric range and exactness contracts** — uint64 IDs, exact integer
   aggregates with a documented bound (no panic, wrap, or float
   approximation on any input), occurrence-mode re-enforcement, i64::MIN
   edges, zero-total rendering arms.
3. **Cross-engine dialect alignment** — YAML scalar typing, tags, merge
   keys, and duplicate keys; strict RFC 8259 JSON including duplicate
   members and lone surrogates; regex dialect (numeric-backref rewriting,
   documented engine-native matching edges); CSV quoting; CPython-`repr`
   float spelling and unicode escaping in artifacts.
4. **Signed-data rendering rules** — negative aggregates render, zero
   rows are omitted only when their entire rendered subtree is zero (at
   every rollup level: categories, buckets, domains, caller/leaf pairs,
   pseudo-slices), magnitude-aware noise gates.
5. **Input-grammar strictness** — reserved names (`(gc)`,
   `(uncollapsible)`, `(all)`, the implicit `Other` bucket), unknown-key
   allowlists at every config level, empty predicates, strict numeric
   option grammars, the comma-reserved focus grammar, empty-path
   rejection, last-wins repeated options.
6. **Protobuf strictness** — field number 0 and the 2^29-1 maximum,
   group wire types skipped balanced, invalid UTF-8 rejected, singular
   embedded-message merge semantics, multi-member and zero-padded gzip.

All resulting behavior contracts are normative in
`docs/CLANKERPROF_SPEC.md`; coverage rows live in
`docs/CLANKERPROF_PARITY.md`.

## Resolution

Accepted as materially converged on 2026-07-17. A zero-confirmation round
never occurred: a maximum-effort frontier auditor with a fresh session
each round kept finding progressively more esoteric edges, and the
severity trajectory — critical gate bypasses in round 1, input-grammar
corner cases by round 10 — showed the defect classes that matter had been
exhausted well before the cap. Nothing found after round 4 threatened
gate semantics or data integrity for tool-produced artifacts. The audit
loop's charter records the acceptance and closeout; `goalloop goal`
exits 0 on the preserved branch.

Operational lessons from running the loop at this scale are recorded in
`docs/GOALLOOP.md`.
