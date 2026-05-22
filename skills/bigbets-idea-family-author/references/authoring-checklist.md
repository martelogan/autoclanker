# Idea-Family Authoring Checklist

Complete this before publishing or materially revising an idea-family issue.

## Intake

- Capture the user's rough idea in their words.
- Ask for missing references only after checking discoverable local/repo/project
  artifacts.
- Identify the target harness, benchmark command, corpus, lockfile, and primary
  metric.
- Confirm whether the lane should optimize for near-term wins, long-term
  unlocks, or both.

## Baseline And Artifacts

- Find the newest relevant baseline run or ask the user to choose one.
- Prefer immutable references: commit SHA, artifact checksum, branch name with
  timestamp, object-store URI, or published bundle ID.
- Record how to copy or download the seed artifact locally.
- Verify the seed artifact parses before linking it.

## Related Work

- Search existing issues, PRs, branches, docs, and run ledgers for overlapping
  work.
- Classify related work as prerequisite, sibling, duplicate, rejected attempt,
  or evidence-only.
- For non-obvious algorithms or systems designs, search external literature or
  prior art and summarize only what changes the lane.

## Lane Contract

- Define one to five pathways. Each pathway needs a hypothesis, starting
  points, guardrails, and acceptance checks.
- State promotion gates and reject/blocker criteria.
- State what must stay fixed during evaluation: corpus, scoring, baseline,
  random seed, eval command, or other harness controls.
- Include observability requested from the run: progress, candidate state,
  metrics, confidence/noise, and logs.

## Kickoff Surface

- Provide the simplest local setup command the harness can support.
- Provide a concise remote-supervisor prompt when remote execution is expected.
- Require exhaustive exploration until every credible pathway is promoted,
  rejected, parked, or blocked with evidence.
- Require the final result to link branches, PRs, artifacts, logs, and decision
  summaries.

## Maintenance

- Include a current pathway-status table.
- Include a run-ledger comment template.
- Explain when to update the issue body versus append a comment.
- Explain how to supersede a seed artifact without losing history.
