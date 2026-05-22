---
name: bigbets-site-operator
description: Use when maintaining a host-neutral generated bigbets static site, storage adapter, snapshots, exported artifacts, or schema-versioned generated files.
---

# Bigbets Site Operator

Use this skill when updating or validating a generated bigbets site.

## Workflow

1. Treat the registry as canonical.

- Edit the source registry, not generated artifacts, unless debugging the site
  generator itself.
- Preserve any host-specific `storage-adapter.js` unless explicitly replacing
  it.
- If the desired change affects schema, generated UI, artifact shape, storage
  hooks, or validation behavior, update the generic `bigbets` package first;
  then regenerate the site from the package. Do not hand-patch generated files
  as the durable fix.

2. Regenerate and validate.

```bash
bigbets validate --input <bigbets.yaml>
bigbets site scaffold --input <bigbets.yaml> --output-dir <site-dir> --app-id <app-id>
node --check <site-dir>/app.js
```

3. Snapshot before large edits.

```bash
bigbets snapshot create --input <bigbets.yaml> --output-dir <site-dir> --name "<label>"
```

4. Keep generated artifacts together.

- `registry.seed.json` is the input snapshot for the browser.
- `big_bets.registry.json`, CSV, Markdown, Mermaid, SVG, and Excalidraw are
  regenerated from the same registry.
- `SITE.md` should describe the local preview and adapter boundary.
- Portfolio-level `metadata.links` are the right place for shared boards,
  meeting notes, or source dashboards that belong to the whole plan.

## References

- For site operation details, read [`references/site-ops.md`](references/site-ops.md).
