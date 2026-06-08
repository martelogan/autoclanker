---
name: issue-seed-author
description: Use when turning evidence, benchmark snapshots, rough ideas, or clankergraph artifacts into a ready-to-run autoclanker issue seed.
---

# Issue Seed Author

Use this skill when a user wants an optimization issue that can directly seed a
long-running agent or benchmark run.

## Workflow

1. Gather the smallest durable seed.

- Goal and target repo.
- Candidate lanes or rough ideas.
- Constraints, rollback rules, and fixed-eval requirements.
- Benchmark snapshots, corpus snapshots, clankergraph files, profiler notes, or
  prior run artifacts.
- Optional `bigbets` mapping fields: `big_bet`, `priority`, and `next_action`.

2. Write a seed JSON file.

```json
{
  "title": "Explore lower-cost pipeline execution",
  "goal": "Reduce pipeline cost without reducing correctness.",
  "target_repo": "owner/repo",
  "ideas": ["Batch repeated data fetches."],
  "artifacts": [
    {
      "name": "evidence-graph",
      "kind": "clankergraph",
      "url": "https://example.invalid/evidence.clankergraph.json"
    }
  ]
}
```

3. Generate artifacts.

```bash
autoclanker issue-seed generate --input seed.json --output-dir tmp/issue-seed
```

4. Validate the generated seed before handing it off.

- `issue_body.md` starts with `Start Here`.
- `autoclanker.ideas.json` validates with `autoclanker beliefs validate`.
- `run-contract.json` names the fixed eval and stopping rules.
- `lane-ledger.md` has explicit active lanes.
- Pi and headless prompts both say setup is not execution; the supervising
  agent must execute the returned handoff prompt.

## Guardrails

- Do not put secrets, raw private payloads, customer data, or unredacted
  benchmark rows in issue bodies.
- Link durable artifacts instead.
- Evidence artifacts are search inputs, not promotion proof.
- Do not stop at one local optimum while other seeded lanes remain plausible.
- Keep hosted adapters optional; LLM/provider keys belong behind server-side
  adapters or CLI environment variables.

## References

- [`docs/ISSUE_SEEDER.md`](../../docs/ISSUE_SEEDER.md)
- [`docs/HOST_ADAPTERS.md`](../../docs/HOST_ADAPTERS.md)
- [`examples/issue_seeder/pipeline_optimization.seed.json`](../../examples/issue_seeder/pipeline_optimization.seed.json)
