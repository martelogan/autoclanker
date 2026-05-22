---
name: bigbets-idea-family-author
description: Use when creating or revising a normalized, self-contained idea-family issue for a bigbets portfolio from benchmark evidence, rough optimization ideas, run notes, or an autoclanker ideas artifact.
---

# Bigbets Idea-Family Author

Use this skill to turn a candidate optimization lane into a durable issue that a
local or remote harness can pick up without prior context. Treat issue authoring
as a research task, not a template fill.

## Workflow

1. Establish the evidence base.

- Identify the latest relevant benchmark baseline, locked corpus, run bundle,
  branch, artifact bucket/object-store path, or explicit user-provided reference.
- If no baseline is discoverable, ask one crisp question before fabricating a
  starting point.
- Inspect related issues, pull requests, run ledgers, design docs, and prior
  rejected attempts.
- For algorithmic or systems ideas, do a literature/prior-art pass when it can
  materially change the lane framing.

2. Shape the lane.

- Decide whether this is a new idea family or belongs inside an existing lane.
- Define candidate pathways that are as disjoint as practical.
- Write or update the `*.ideas.json` seed with constraints, evaluation gates,
  candidate pathways, relations, and an agent prompt nudge.
- Pin artifacts to immutable references when possible.

3. Write the issue as self-contained Markdown.

- Start with a short human summary and why the lane matters.
- Put a `bigbets:idea-family` metadata block near the top.
- Include the primary artifact link plus inline expandable JSON.
- Include evidence, related work, evaluation gates, kickoff instructions, and a
  run-ledger template.
- Include both local and remote/supervisor kickoff surfaces when the harness
  supports both.
- Keep stale history out of the top-level contract; fold durable learning into
  the current summary/status table.

4. Validate the issue.

```bash
bigbets issues import --input <issue.md>
```

If the issue is already exported as JSON, validate the export instead:

```bash
bigbets issues import --input <issues.json>
```

5. Keep issue metadata stable.

- `slug` should be short enough for board nodes and meeting notes.
- `big_bet` must name exactly one owning big bet.
- `priority` is the lane's current urgency, not the owning bet's layer.
- `role` should be one of `ideas-lane`, `wip`, `evidence`, `proof`,
  `follow-up`, `blocked`, or `shipped`.

## References

- For the normalized issue body shape, read
  [`references/normalized-idea-family.md`](references/normalized-idea-family.md).
- For the evidence-gathering checklist, read
  [`references/authoring-checklist.md`](references/authoring-checklist.md).
