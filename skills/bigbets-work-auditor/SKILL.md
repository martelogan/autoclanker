---
name: bigbets-work-auditor
description: Use when auditing repositories, issues, pull requests, users, projects, or run artifacts to discover candidate idea-family lanes and categorize them into big bets.
---

# Bigbets Work Auditor

Use this skill to convert scattered work into candidate idea-family issues and a
portfolio draft.

## Workflow

1. Inventory the requested sources.

- Pull requests, issues, project trackers, benchmark bundles, profile summaries,
  and user-authored notes.
- For each source, capture only durable facts: scope, evidence, status, owner,
  blockers, and related artifacts.
- Identify the newest credible baseline/evidence bundle for the workstream; if
  multiple baselines exist, keep the ambiguity visible instead of collapsing it.

2. Distill candidate lanes.

- Merge duplicates that test the same mechanism.
- Split lanes when they require different evidence, owners, or rollback paths.
- Prefer mutually distinct lanes that a harness can explore independently.
- Preserve rejected attempts as evidence, not as active pathways, unless new
  evidence changes the blocker.

3. Categorize lanes into big bets.

- Name the shared strategic bet.
- State the near-term win and long-term unlock.
- Rank by evidence strength, expected impact, scope clarity, and reversibility.
- Keep tooling/evidence hygiene as an underlay unless it is itself the product
  of the bet.

4. Produce normalized outputs.

- A candidate issue body per lane.
- A seed artifact or explicit artifact TODO per lane.
- A registry patch or full registry.
- A short audit note explaining merges, splits, and rejected lanes.

## References

- For the expected audit handoff, read
  [`references/audit-output.md`](references/audit-output.md).
