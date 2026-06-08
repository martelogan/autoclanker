# Issue Seeder

The issue seeder turns benchmark evidence, clankergraph artifacts, rough ideas,
and run constraints into a ready-to-run optimization issue. It exists for the
handoff boundary between evidence collection and long-running agent execution:
the issue should be enough for a human or supervisor to hydrate the workspace,
read the current evidence, and start a multi-lane run without stale side
context.

The source of truth is the CLI generator:

```bash
autoclanker issue-seed generate \
  --input examples/issue_seeder/pipeline_optimization.seed.json \
  --output-dir tmp/issue-seed
```

For a request-rendering workload, use the companion example:

```bash
autoclanker issue-seed generate \
  --input examples/issue_seeder/request_rendering.seed.json \
  --output-dir tmp/issue-seed-rendering
```

That command emits one JSON response and, when `--output-dir` is provided,
materializes a portable artifact bundle:

- `issue_body.md`
- `autoclanker.ideas.json`
- `artifact-manifest.json`
- `run-contract.json`
- `lane-ledger.md`
- `pi.prompt.txt`
- `headless-command.sh`
- `host-adapter-contract.md`

## Seed Input

Minimal seed input:

```json
{
  "title": "Explore lower-cost pipeline execution",
  "goal": "Reduce pipeline cost without reducing correctness.",
  "target_repo": "owner/repo",
  "ideas": [
    "Batch repeated data fetches.",
    "Avoid repeated decode work."
  ]
}
```

Useful optional fields:

- `label`: suggested issue label.
- `era_id` and `session_id`: default autoclanker session context.
- `run_intensity`: `standard`, `deep`, or `mega`.
- `constraints`: operator rules and rollback constraints.
- `artifacts`: evidence URLs, immutable object keys, or checked-in paths.
- `benchmark_snapshot` and `corpus_snapshot`: shortcut artifact references.
- `adapter_config_path`: path used by headless commands.
- `canonicalization_mode`: `deterministic`, `hybrid`, or `llm`. Defaults to
  `deterministic` so generated commands do not require provider credentials.
- `canonicalization_model`: optional provider identifier or import path used
  only when the chosen mode needs a model.
- `big_bet`, `priority`, and `next_action`: optional `bigbets` mapping metadata.

Artifacts should be references, not sensitive payload dumps. Keep raw payloads,
credentials, customer data, and private rows out of issue bodies.

## Generated Contract

Generated issues start with a compact "Start Here" section and then use
expandable sections for:

- Local Pi kickoff prompt.
- Headless CLI kickoff command.
- Seed artifacts and expected workspace files.
- Seed lanes.
- Constraints.
- Run contract.
- Lane ledger.
- Optional host adapter contract.

Every generated prompt preserves these rules:

- Setup commands prepare or resume the workspace.
- The supervising agent must execute the returned handoff prompt.
- Evidence artifacts are search inputs, not proof.
- The eval surface stays fixed during measurement.
- Multi-lane evals must bind measurements to explicit candidate identity.
- One small local optimum is not enough to stop while seeded lanes remain
  plausible and runnable.
- Proposals, blockers, and lane decisions must be written before exit.
- Default headless commands use deterministic canonicalization; hybrid or LLM
  canonicalization must be selected deliberately with a model/provider.

In short: setup commands prepare or resume the workspace; execution still
belongs to the supervising agent that reads the handoff and runs the measured
loop.

Before handoff, validate generated bundles with:

```bash
autoclanker beliefs validate --input tmp/issue-seed/autoclanker.ideas.json
test -x tmp/issue-seed/headless-command.sh
```

## Static Or Hosted UX

The CLI generator is the stable, testable core. A static site or hosted issue
manager should call into the same seed JSON contract instead of inventing a new
issue shape.

Recommended local-first static behavior:

- Store draft seeds in browser local storage.
- Import/export seed JSON.
- Render the same issue body and artifact files as the CLI.
- Avoid GitHub, database, or LLM dependencies by default.

Recommended optional host adapter methods:

- `loadSeeds({ appId })`
- `saveSeed({ appId, seed })`
- `uploadArtifact({ appId, name, contentType, body })`
- `fetchIssue({ repo, issue, url })`
- `createIssue({ repo, title, body, labels })`
- `canonicalizeIdeas({ seed, artifacts })`

LLM provider keys belong behind `canonicalizeIdeas` or in the CLI environment,
not in the static browser app. Artifact uploads should return immutable URLs or
object keys that can be copied into `artifact-manifest.json`.

## Relation To Bigbets

`bigbets` maintains portfolio-level strategy and can import normalized
idea-family issue metadata. The issue seeder owns per-issue run kickoff. They
compose through the optional `bigbets:idea-family` metadata block in generated
issue bodies:

```bash
bigbets issues import --input issue_body.md
bigbets issues merge \
  --registry examples/bigbets/basic_portfolio.yaml \
  --input issue_body.md \
  --output tmp/portfolio.with-issue.json
```

Use `bigbets` when deciding which issue-family lanes matter most. Use
`autoclanker issue-seed generate` when preparing a single issue so an agent can
run it.
