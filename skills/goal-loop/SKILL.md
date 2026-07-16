---
name: goal-loop
description: Use when running a long, multi-hour implementation effort as a deterministic goal loop — enumerating requirements into a tracker, iterating in small verified clusters across sessions or harnesses, and optionally converging through adversarial audit rounds by a second agent.
---

# Goal Loop Operator

Use this skill when a task is too large for one sitting and must survive
interruptions, harness switches, or model handoffs: a charter defines done,
a tracker holds execution state, gates define green, and `goalloop goal`
is the single deterministic completion check. The full file and CLI contract
lives in `docs/GOALLOOP.md`.

The `goalloop` CLI is host-neutral: every artifact is a plain file at the
loop root, so Claude Code, Codex, pi, or a human can continue the same loop.
Run it standalone (`goalloop …`) or via the umbrella (`autoclanker goalloop …`).

## Workflow

1. **Charter the goal.** Run `goalloop init --name <loop> --gate '<command>'`
   (repeat `--gate` for each required check; add `--auditor '<command>'` to
   enable adversarial audit). Then edit `goalloop.charter.md` to a real goal
   definition: concrete Outcome, the Evidence that proves it, explicit Scope
   bounds, and Stop conditions that escalate instead of iterating. If you
   changed the gates or audit policy while editing, run `goalloop lock` — the
   contract digest is locked at init, and a drifted contract blocks
   `goalloop goal` until re-locked.
2. **Enumerate requirements.** Replace the skeleton in `goalloop.tracker.md`
   with wave-grouped rows (`A-01`, `A-02`, … `B-01`): one checkable
   requirement per row, each with a deterministic Verify command. IDs must
   match `<WAVE>-<NUMBER>` (uppercase wave token, dash, digits); `R<N>` waves
   are reserved for audit rounds; cell text must not contain `|`. The tracker
   is the single source of execution state — never track progress anywhere else.
3. **Iterate in small clusters.** Each iteration: run `goalloop handoff` to
   get the next-iteration prompt (protocol + pending rows + charter), pick one
   small cluster, implement, run each row's Verify plus the charter gates with
   REAL exit codes, and commit with the rows flipped to `done` in the same
   commit. An interruption should never lose more than one cluster.
4. **Drive the loop from your harness.** Claude Code: schedule a wakeup (or a
   stop-hook loop such as ralph-loop) whose prompt is the `goalloop handoff`
   output, with completion promise `goalloop goal` exit 0. Codex: run
   `codex exec` iterations seeded from `goalloop handoff`, optionally pinned
   by a native goal. pi: feed `goalloop handoff` as the supervisor handoff
   prompt each turn. Humans: just read the tracker.
5. **Check completion.** `goalloop goal` exits 0 only when every row is
   `done` (or `dropped` with a reason), every gate passes, and any configured
   audit has converged. Use `goalloop status` / `goalloop assert <wave-or-id>`
   for partial checks.
6. **Converge through audit (if configured).** On completion run
   `goalloop audit prompt` and feed it to the independent auditor (a different
   model/agent, read-only). Triage its findings by attempting reproduction,
   write the triaged JSON (`title` / `verdict: confirmed|refuted` /
   `evidence`), and run `goalloop audit ingest findings.json` — confirmed
   findings become a new `R<N>` tracker wave to implement; refutations enter
   the log so the auditor cannot re-raise them without new evidence. Loop
   until a round confirms nothing new; past `max_rounds`, escalate to a human.

## Bayesian guidance (optional)

When iterations are parameterized (tuning knobs, competing approaches), wire
the loop into an autoclanker session via the `goalloop` adapter kind: the
loop's gates become the session's locked eval surface, candidates arrive as
`GOALLOOP_GENE_*` environment variables, and gates report measurements with a
`GOALLOOP_METRICS={...}` JSON line. See docs/GOALLOOP.md and
`examples/adapters/goalloop.local.yaml`.

## Rules

- Never pipe a gate into `tail`/`head` — that masks the exit code. Capture it.
- Never weaken gates or audit policy to make `goal` pass — that is contract
  drift; if a change is genuinely warranted, make it and `goalloop lock` it
  as a visible, deliberate act.
- Flip tracker rows and land the code in the SAME commit.
- `dropped` requires a reason in Notes; duplicate IDs are rejected.
- Stop conditions from the charter override everything else.
