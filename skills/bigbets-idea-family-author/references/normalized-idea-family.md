# Normalized Idea-Family Issue

Use this structure when an issue should be ingestible by `bigbets issues import`.

```markdown
# <clear lane title>

## Lane Summary

<Two or three sentences explaining the optimization lane, why it matters, and
the current best proof case.>

<!-- bigbets:idea-family
issue: 1001
slug: short_slug
big_bet: owning_big_bet_id
priority: P0
status: candidate
role: ideas-lane
artifact: artifacts/lane.autoclanker.ideas.json
next_action: Run the smallest evidence-backed candidate first.
links:
  - label: Evidence bundle
    url: https://example.invalid/evidence
    kind: evidence
-->

## Current Thesis

State the hypothesis, expected win shape, current baseline/corpus, and simplest
first candidate.

## Evidence

List durable benchmark, profile, trace, run, branch, and artifact references.
Prefer links and short interpretation over pasted logs.

## Related Work

List prerequisite, sibling, duplicate, rejected, and evidence-only references.

## Evaluation Contract

- Primary metric:
- Primary harness:
- Fixed baseline/corpus:
- Promotion gates:
- Reject/blocker criteria:
- Observability required:

## Current Pathway Status

| Pathway | Status | Latest durable evidence |
| --- | --- | --- |
| `<pathway>` | Active / parked / blocked / shipped | `<artifact>` |

## Run This Lane

### Remote supervisor prompt

```text
Explore this idea-family lane end to end. Use the linked evidence and ideas
artifact, test every credible pathway, keep the evaluator fixed, and stop only
when each pathway has a promote/reject/park/block decision backed by evidence.
```

### Local setup

```bash
# Fill in the repo, baseline branch/artifact, and harness command for this lane.
```

<details>
<summary>ideas.json</summary>

```json
{
  "ideas": []
}
```

</details>

## Acceptance

Define the measured improvement, parity, regression, and confidence bar.

## Maintenance Contract

- Keep the body as the current contract, not a chronological log.
- Add one run-ledger comment per durable run.
- Promote a new ideas artifact only after validation.
- Update the pathway-status table when work ships, rejects, parks, or blocks.

<details>
<summary>Run ledger comment template</summary>

```md
### Run ledger: YYYY-MM-DD - <lane> - <run id>

Mode:
Seed branch:
Result branch:
Draft PR:
Artifact version:
Evaluator/corpus:

Pathways explored:
- <pathway id>: kept | rejected | blocked | parked | shipped

Benchmark evidence:
- <candidate>: <metric deltas, confidence/noise, confirmation status>

Decisions:
- Shipped/keep:
- Rejected:
- Blocked:
- Parked for follow-up:

Next contract update:
- <body/artifact changes needed before next run>
```

</details>
```
