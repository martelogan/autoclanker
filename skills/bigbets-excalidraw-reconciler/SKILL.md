---
name: bigbets-excalidraw-reconciler
description: Use when reconciling an edited Excalidraw board back into a host-neutral bigbets registry, including node moves, renames, dependency edge edits, or issue/slug references visible in the board.
---

# Bigbets Excalidraw Reconciler

Use this skill after people modify a downloaded `big_bets.excalidraw` board and
want the durable registry to reflect the real planning changes.

## Workflow

1. Establish the baseline.

- Load the canonical registry with `bigbets validate --input <bigbets.yaml>`.
- Render a fresh baseline Excalidraw from that registry unless the user
  provided the exact baseline export.
- Treat the registry as source of truth; the edited board is an input proposal.

2. Extract board intent.

- Parse the edited `.excalidraw` JSON.
- Match big-bet nodes by stable text first, then by known IDs, slugs, issue
  numbers, or nearby baseline positions.
- Classify visible changes as: layer/order move, title rename, dependency edge
  add/remove, edge-label edit, issue-family reassignment, or new candidate.
- Ignore purely visual edits unless the user explicitly asks to preserve them in
  a regenerated artifact.

3. Apply conservative registry changes.

- Update `priority` only through the layer-depth contract: `P0` is wave 1,
  `P1` is wave 2, and so on.
- Update `rank` from left-to-right order within a layer.
- Update `depends_on`, `unlocks`, and `edge_labels` only when the edited board
  shows a clear directed relationship.
- Map visible issue numbers or stable slugs to existing idea-family rows before
  creating anything new.
- If a board edit is ambiguous, leave the registry unchanged and list the
  decision that needs human confirmation.

4. Validate and regenerate.

```bash
bigbets validate --input <bigbets.yaml>
bigbets render --input <bigbets.yaml> --output-dir <output-dir>
```

For generated sites, rerun `bigbets site scaffold` after the registry passes.

## Reference

- For reconciliation rules and expected output, read
  [`references/reconciliation.md`](references/reconciliation.md).
