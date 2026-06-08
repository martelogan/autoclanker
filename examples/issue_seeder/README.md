# Issue Seeder Example

This directory contains a portable issue-seed input for the repo-native
generator:

```bash
autoclanker issue-seed generate \
  --input examples/issue_seeder/pipeline_optimization.seed.json \
  --output-dir tmp/issue-seed
```

Additional generic seeds:

- `pipeline_optimization.seed.json`: default cross-domain data-pipeline example.
- `request_rendering.seed.json`: request-rendering example for render plans,
  component lookups, view-model shape, and response correctness without
  framework-specific assumptions.

The command writes:

- `issue_body.md`
- `autoclanker.ideas.json`
- `artifact-manifest.json`
- `run-contract.json`
- `lane-ledger.md`
- `pi.prompt.txt`
- executable `headless-command.sh`
- `host-adapter-contract.md`

The generated issue body is designed to be copied into an issue tracker. The
generated JSON/Markdown files are designed to be uploaded as artifacts, checked
into a seed branch, or copied into a preseeded workspace.

The default path is static and credential-free. A hosted deployment may add a
storage or LLM adapter, but provider keys should stay behind a server-side
adapter and never in browser-local seed JSON.

The adjacent `index.html` is a static browser preview for the same seed shape.
It stores drafts in browser local storage only. For a real handoff, use the CLI
generator so the generated files, validation behavior, and shell permissions
match the package contract.

The example uses deterministic canonicalization by default. Add
`canonicalization_model` and switch `canonicalization_mode` to `hybrid` or
`llm` only when the receiving environment has the intended provider configured.
