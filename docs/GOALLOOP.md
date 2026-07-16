# goalloop: deterministic goal loops for agent harnesses

`goalloop` turns a long implementation effort into a resumable, auditable
loop that any agent harness — Claude Code, Codex, pi — or a human can drive.
It generalizes the loop architecture that shipped clankerprof v2 (71
requirements, 21 gate-verified commits across 9 interrupted sessions) into a
host-neutral subpackage.

The design bet: the loop state must live in **plain files**, not in any
harness. A harness contributes only two things — a way to re-invoke the agent
(scheduler, stop-hook, supervisor turn) and a prompt for each iteration. Both
are commodity; the contract below is the durable part.

## Files at the loop root

| File | Role |
| --- | --- |
| `goalloop.charter.md` | The goal definition: YAML frontmatter (`name`, `gates`, optional `audit: {auditor, max_rounds, convergence, notes}`) plus an Outcome / Evidence / Scope bounds / Stop conditions body. Unknown audit keys are a validation error — a typo must never silently weaken the audit policy. |
| `goalloop.tracker.md` | Single source of execution state: wave-grouped requirement rows `| ID | Requirement | Verify | Status | Notes |`. IDs must match `<WAVE>-<NUMBER>` (uppercase wave token with optional trailing digits, dash, digits — e.g. `A-01`, `B2-03`, `R1-02`); malformed IDs are a hard error, never silently skipped, and `R<N>` waves are reserved for audit ingestion. Status vocabulary: `todo`, `doing`, `done`, `blocked`, `dropped` (dropped requires a reason in Notes). Duplicate IDs are rejected. |
| `goalloop.audit.md` | Append-only adversarial audit log: `## Round N` sections with `- CONFIRMED [R<N>-XX] …` and `- REFUTED <title>: <proof>` lines. |
| `goalloop.history.jsonl` | Append-only JSONL event stream (`init`, `lock`, `goal`, `audit_ingest`). |

## Commands

All commands are non-interactive, take `--root` (default `.`), print JSON to
stdout (except `handoff` and `audit prompt`, whose text output is the
artifact), and use exit codes: 0 success, 1 not-met/gate-failure
(gate exit codes propagate verbatim), 2 validation error. Also available as
`autoclanker goalloop …`.

- `goalloop init --name N [--gate CMD]... [--auditor CMD] [--max-audit-rounds K]
  [--audit-convergence zero|no_major]`
  — scaffold the charter and tracker; refuses to overwrite an existing loop.
  `K` defaults to 10: audits should run as long as needed for convergence,
  with the cap as a runaway backstop rather than an expected stop.
- `goalloop status` — progress summary: per-wave done/total with pending rows,
  gates, audit state.
- `goalloop assert ID_OR_WAVE...` — exit 1 unless the named rows/waves are
  finished; selectors matching no row or wave also fail (reported under
  `unknown`), so typos never pass vacuously. For wiring partial milestones
  into scripts.
- `goalloop gate` — run the charter gates in order, capturing each real exit
  code and output tail; stops at the first failure and propagates its code.
- `goalloop lock` — record the current charter contract digest (sha256 over
  name, gates, and audit policy — the prose body is excluded) as the locked
  definition of done. `init` locks automatically; after any intentional
  contract edit, re-lock. This mirrors the bayes-layer eval contract: eval
  results with a mismatched contract are rejected, and here a drifted
  contract blocks `goal` until re-locked — moving the goalposts mid-loop is
  always an explicit, history-recorded act.
- `goalloop goal` — the completion check. Exit 0 only when: the charter
  contract has not drifted from its lock, AND every tracker row is `done` or
  `dropped`-with-reason, AND (if an auditor is configured) the audit has
  converged, AND every gate passes.
- `goalloop handoff [--json]` — the next-iteration prompt: the loop protocol
  (small clusters, real exit codes, tracker-flip-in-same-commit, stop
  conditions), current progress, pending rows, and the charter body. Feed this
  to whatever continues the loop.
- `goalloop audit prompt` — the auditor charter: an independent, read-only
  auditor is told to refute completion, must attempt reproduction for every
  finding, must not re-raise refuted findings without new evidence, and must
  answer in strict JSON. When the charter carries `audit.notes`, the prompt
  ends with an `--- OPERATOR NOTES ---` section — the place for scope
  carve-outs and auditor-platform constraints (see "at scale" below); notes
  are part of the locked audit policy.
- `goalloop audit ingest FINDINGS.json` — ingest **triaged** findings (the
  implementor attempts reproduction first): a JSON array of
  `{"title", "verdict": "confirmed"|"refuted", "evidence"}` with an optional
  `"severity": "critical"|"major"|"minor"` per finding. Confirmed findings
  are appended to the tracker as a new `R<N>` wave (so the ordinary loop
  machinery drives the fixes), with the severity recorded on the tracker row
  and the audit-log line; refutations go to the audit log. Rounds past
  `max_rounds` are rejected with instructions to escalate to a human.
- `goalloop audit status` — per-round confirmed/refuted counts, the
  convergence policy, and convergence state.

**Convergence:** under the default policy (`zero`) the audit converges when a
completed round confirms nothing new. Under `audit.convergence: no_major` it
converges when a round confirms nothing of critical or major severity — a
minor-only round converges. A finding ingested without a severity counts as
major (fail-closed: an unlabeled finding must never be what lets the audit
converge). With no auditor configured, the audit phase is vacuously
converged. The convergence policy and notes join the contract digest, so
changing either mid-loop is goalpost-moving: `goal` blocks until an explicit
re-lock.

## The loop protocol (what makes it deterministic)

1. The tracker is the single source of execution state; harness memory is a
   cache, never the truth.
2. Work lands in small clusters; each cluster's rows flip to `done` in the
   same commit as the code, behind that cluster's verification. An
   interruption never loses more than one cluster.
3. Gates run with real exit codes — never piped into `tail`/`head`, which
   masks failures.
4. `goalloop goal` exit 0 is the only definition of done.
5. Charter stop conditions override everything: hard blockers are recorded in
   tracker Notes and escalated, not iterated on.

## Driving the loop per harness

- **Claude Code** — self-schedule: end each iteration by scheduling a wakeup
  whose prompt is the `goalloop handoff` output. Or use a stop-hook loop
  (e.g. the ralph-loop plugin) with `goalloop goal` exit 0 as the completion
  promise.
- **Codex** — run `codex exec` iterations seeded from `goalloop handoff`;
  optionally pin the effort with a native Codex goal whose completion
  criterion is `goalloop goal` exiting 0.
- **pi / pi-autoclanker** — the supervisor pattern: each turn's handoff prompt
  is `goalloop handoff`; the extension's loop continues until `goalloop goal`
  exits 0. pi-autoclanker ships a `goalloop_*` tool family (init/status/gate/
  goal/handoff/audit) wrapping these commands exactly as it wraps the
  `autoclanker` CLI.
- **Human** — read the tracker, do a cluster, flip rows, commit.

## Adversarial audit in practice

The audit phase pits two independent agents against each other:

1. Implementor believes the work is done (`goalloop goal` blocks, reason
   "audit not converged").
2. `goalloop audit prompt | codex exec --sandbox read-only …` (or any other
   auditor command) produces strict-JSON findings with reproductions.
3. The implementor triages each finding by attempting the reproduction,
   assigning `confirmed` or `refuted` with evidence.
4. `goalloop audit ingest` turns confirmed findings into a new tracker wave;
   the ordinary loop fixes them; the next round's auditor sees the refutation
   log and cannot re-raise dead findings without new evidence.
5. A round satisfying the convergence policy converges the audit;
   `goalloop goal` can now pass. Past `max_rounds`, the loop refuses to
   continue and escalates.

## Adversarial audits at scale

The clankerprof v2 review (10 rounds, 101 confirmed findings; see
`docs/CLANKERPROF_ADVERSARIAL_AUDIT.md`) is the reference run for large
surfaces. What it taught:

- **Batch the triage.** Reproduction triage is mechanical; a few batched
  verifier agents at moderate effort re-reproduce a whole round for a
  fraction of the cost of one high-effort agent per finding, with no loss —
  judgment lives in the fix clusters, not the triage.
- **Expect auditor-platform interference.** External auditor platforms may
  content-filter fuzzing-style probing (byte-mutation sweeps over binary
  inputs pattern-match to security tooling) and kill the session mid-audit,
  losing its findings. Use `audit.notes` to steer the auditor toward
  hand-built single-hypothesis fixtures, and keep the audit prompt compact —
  a tracker full of malformed-input language is itself trigger surface.
- **Choose the convergence policy by surface size.** `zero` suits small,
  bounded surfaces. On a large surface a max-effort frontier auditor keeps
  finding progressively more esoteric true positives indefinitely — the
  signal that matters is severity migration, which `no_major` encodes.
- **Interruptions are cheap by design.** All state is files, so network
  drops, credit exhaustion, and killed auditor sessions cost only the
  in-flight step; resume from the tracker.

Deferred candidates (judged not yet worth building; recorded so future
sessions don't re-derive them): direct ingestion of raw auditor JSON without
implementor triage (removes the reproduction discipline that caught real
auditor false alarms), per-round token budgets (the harness driving the loop
meters spend better than the loop can), and severity-weighted round caps
(max_rounds plus `no_major` covers the observed need).

## Bayesian guidance over a goal loop (autoclanker adapter)

The synthesis with autoclanker's core engine runs through the `goalloop`
adapter kind (`docs/INTEGRATIONS.md` §2): an autoclanker session can treat a
goal loop's charter gates as its locked eval surface. Configure
`kind: goalloop, mode: local_repo_path` with `repo_path` at the loop root and
`metadata.genes` declaring the tunable knobs; suggested candidates reach the
gates as `GOALLOOP_GENE_*` environment variables, gates report measurements
with a `GOALLOOP_METRICS={...}` JSON line, and results carry the loop's
contract digest. The two lock systems compose: the session's eval contract
rejects results from a changed adapter surface, and the adapter refuses to
evaluate at all while the loop's own charter contract is drifted from its
lock. Beliefs about which approaches are risky or promising flow through the
ordinary belief → posterior → suggest pipeline; the loop supplies the durable
definition of green. See `examples/adapters/goalloop.local.yaml`.

## Operator skill

`skills/goal-loop/SKILL.md` packages this workflow for agent harnesses
(agentskills.io format, consumable by Claude Code, Codex, and pi).
