# Static Host Adapters

`autoclanker` and `bigbets` both default to local files and browser
localStorage. Hosted deployments are optional adapter layers, not required
runtime dependencies.

## Principles

- Keep the CLI contract runnable without a hosted service.
- Keep static-site drafts importable/exportable as plain JSON.
- Keep provider keys, GitHub tokens, and artifact write credentials out of the
  browser.
- Treat uploaded artifacts as immutable references that can be copied into issue
  bodies.
- Prefer server-side LLM canonicalization through the same typed belief contract
  used by `autoclanker beliefs canonicalize-ideas`.

## Bigbets Site Adapter

`bigbets site scaffold` can copy a host-specific `storage-adapter.js` beside the
generated site:

```bash
bigbets site scaffold \
  --input examples/bigbets/basic_portfolio.yaml \
  --output-dir tmp/bigbets-site \
  --storage-adapter-file path/to/storage-adapter.js
```

The generated app uses `window.bigbetsStorageAdapter` when present and falls
back to browser localStorage otherwise.

Supported optional methods:

- `load({ appId })`
- `save({ appId, registry })`
- `listSnapshots({ appId })`
- `saveSnapshot({ appId, registry, name })`
- `loadSnapshot({ appId, snapshotId })`
- `writeArtifacts({ appId, artifacts })`
- `fetchIssue({ url, issue })`

## Issue Seeder Adapter

A static issue-seeder UI should use the seed JSON consumed by:

```bash
autoclanker issue-seed generate --input seed.json
```

Recommended optional methods:

- `loadSeeds({ appId })`
- `saveSeed({ appId, seed })`
- `uploadArtifact({ appId, name, contentType, body })`
- `fetchIssue({ repo, issue, url })`
- `createIssue({ repo, title, body, labels })`
- `canonicalizeIdeas({ seed, artifacts })`

`canonicalizeIdeas` should call server-side code that invokes an LLM provider or
the `autoclanker` CLI. It must return inspectable typed ideas, metadata-only
proposals, or a validation error. It must not write directly into posterior
state.

The generated seed remains valid without this method. Default generated
headless commands use deterministic canonicalization so shared demos and local
handoffs do not accidentally depend on provider credentials. Hosted deployments
may rewrite the seed with `canonicalization_mode: "hybrid"` or
`canonicalization_mode: "llm"` only when the server-side adapter has the
intended provider configured.

## Supabase Reference Shape

Supabase is a good optional reference adapter because it can cover:

- Postgres tables for seed catalogs and bigbets registries.
- Storage buckets for artifact bundles.
- Auth and row-level security for shared teams.
- Edge Functions for GitHub issue creation and provider-backed canonicalization.
- Local development with the Supabase CLI.

Suggested tables:

```sql
create table issue_seeds (
  id uuid primary key default gen_random_uuid(),
  app_id text not null,
  title text not null,
  seed jsonb not null,
  updated_at timestamptz not null default now()
);

create table bigbets_registries (
  app_id text primary key,
  registry jsonb not null,
  updated_at timestamptz not null default now()
);
```

Suggested storage layout:

```text
artifact-bundles/
  <app-id>/
    <seed-id>/
      autoclanker.ideas.json
      artifact-manifest.json
      run-contract.json
      lane-ledger.md
      evidence.clankergraph.json
```

Suggested Edge Functions:

- `issue-seed-canonicalize`: reads seed + artifact summaries, calls the
  configured provider, returns typed ideas/proposals.
- `issue-seed-create-github-issue`: creates or updates one issue from generated
  body and labels.
- `artifact-upload-sign`: returns short-lived upload targets for large
  artifacts when direct authenticated upload is not appropriate.

This adapter should live beside a deployment, not in the core Python package,
unless it can be kept credential-free and optional.
