# Big Bets Portfolio Compiler

`bigbets` is a small portfolio compiler for ranked optimization bets. It is
designed for long-running agent programs where individual idea-family issues,
benchmark artifacts, and run ledgers need to roll up into a small meeting-ready
set of strategic bets.

The package is intentionally generic:

- GitHub issues are links, not a hard dependency.
- `autoclanker.ideas.json` files are links, not a required schema.
- Static hosts can serve the generated HTML/JSON; persistence is adapter-based.
- `autoclanker bigbets ...` is a convenience alias; `bigbets ...` is the
  standalone CLI.

## Model

Use one canonical registry and generate every view from it.

```yaml
schema_version: bigbets.registry.v1

big_bets:
  - id: data_plane_batching
    title: Batch the request data plane
    priority: P0
    rank: 1        # serialized position inside P0
    wave: 1        # serialized P-depth; P0 == wave 1
    status: active
    narrative: Collapse repeated request-level data fan-out now.
    near_term_win: Ship one measured batch-loader win.
    long_term_unlock: Enables static planning and adaptive prefetching.
    next_action: Run the strongest concrete batch surface first.
    unlocks: [native_planning]
    edge_labels:
      - target: native_planning
        label: proven contracts

idea_families:
  - issue: 1001
    slug: collection_bulk
    title: Bulk-resolve repeated collection projections
    big_bet: data_plane_batching
    priority: P0
    rank: 1
    status: active
    artifact: artifacts/collection_projection.autoclanker.ideas.json
    url: https://github.com/example/org/issues/1001
    links:
      - label: Tracking project
        url: https://example.invalid/projects/collection-bulk
        kind: project
```

The user-facing planning model is intentionally simple: big bets live in
`P0`, `P1`, `P2`, ... layers, and idea families are ordered lanes inside their
owning bet. The serialized registry still carries `wave` and `rank` for stable
machine artifacts:

- Big-bet `priority` is the layer shown to humans. P0 is the current front, P1
  is the next unlock layer, and so on.
- Big-bet `wave` is the internal dependency depth. Wave 1 serializes P0, wave
  2 serializes P1, and so on.
- Big-bet `priority` is derived from `wave`; the validator rejects mismatches
  so board drags cannot leave stale labels behind.
- `rank` is the internal position inside a layer or bet. The UI makes that
  implicit through drag/drop, keyboard movement, and table ordering rather than
  labeling every row with an order number.
- Idea-family `priority` is the lane's own P-level. It does not change the
  owning bet's layer.
- Idea-family `slug` is the short human label shown in compact views. It should
  be stable enough for meeting notes, but it is not a replacement for the issue
  or PR URL.
- Idea-family `artifact` points at the durable lane artifact, often an
  `*.autoclanker.ideas.json` file. `ideas-lane` is a role; `ideas.json` is a
  concrete artifact link label.
- Idea-family `links` is an optional list of related project trackers, docs,
  PRs, or evidence links. Link `kind` is generic; hosts can render private
  trackers without coupling `bigbets` to that tracker.
- Big-bet `edge_labels` is optional. Use it sparingly for short dependency
  reasons like `safe contracts` or `profile heat`; omit labels when the edge is
  already obvious.
- Idea-family `role` should use the supported vocabulary: `ideas-lane` for an
  issue-backed exploration lane, `wip` for an active implementation or PR,
  `evidence` for measurement/background material, `proof` for a crisp proof
  candidate, `follow-up` for deferred work, or `blocked`/`shipped` when the lane
  state is explicit.

## Invariants

`bigbets validate` enforces the constraints that keep the portfolio maintainable:

- explicit `schema_version` values must match the supported registry schema;
- every idea-family issue maps to exactly one big bet;
- every mapped big bet exists;
- active big bets have at least one idea family;
- P0 big bets are capped by `metadata.max_p0_big_bets`;
- P0/P1 big bets have a `next_action`;
- big-bet `priority` matches its wave-derived priority;
- idea-family roles use the supported vocabulary;
- idea-family link kinds use the supported vocabulary;
- dependency and unlock edges reference valid big-bet ids;
- dependency labels reference an actual dependency/unlock edge;
- big bets define both `near_term_win` and `long_term_unlock`.

Tooling, observability, and benchmark hygiene should usually be modeled as an
always-on underlay rather than as a separate priority layer. Each layer should still
pursue concrete near-term wins while paving longer-term unlocks.

## CLI

Validate a registry:

```bash
bigbets validate --input examples/bigbets/basic_portfolio.yaml
autoclanker bigbets validate --input examples/bigbets/basic_portfolio.yaml
```

Render every artifact:

```bash
bigbets render \
  --input examples/bigbets/basic_portfolio.yaml \
  --output-dir tmp/bigbets-site
```

Generate an editable static site:

```bash
bigbets site scaffold \
  --input examples/bigbets/basic_portfolio.yaml \
  --output-dir ~/bigbets-site \
  --app-id bigbets-braintrust \
  --storage-adapter local-storage

python3 -m http.server --directory ~/bigbets-site 8000
```

Create a dated plan snapshot beside a rendered or scaffolded site:

```bash
bigbets snapshot create \
  --input examples/bigbets/basic_portfolio.yaml \
  --output-dir ~/bigbets-site \
  --name "Weekly plan 2026-05-10"

bigbets snapshot list --output-dir ~/bigbets-site
```

Emit one artifact:

```bash
bigbets emit --input examples/bigbets/basic_portfolio.yaml --format metadata
bigbets emit --input examples/bigbets/basic_portfolio.yaml --format mermaid
bigbets emit --input examples/bigbets/basic_portfolio.yaml --format excalidraw
bigbets emit --input examples/bigbets/basic_portfolio.yaml --format html --output index.html
```

Generated artifacts:

- `big_bets.artifact_metadata.json`: generator version plus registry/artifact
  schema versions;
- `big_bets.registry.json`: normalized machine-readable graph;
- `big_bets.rankings.csv`: sortable table;
- `big_bets.md`: human-readable snapshot;
- `big_bets.mmd`: Mermaid graph;
- `big_bets.excalidraw`: Excalidraw-compatible editable board;
- `big_bets.svg`: polished static visual;
- `index.html`: static page.

`bigbets site scaffold` writes a richer static app instead of only a snapshot.
The generated app is intentionally board-first: the dependency graph is a
top-down Excalidraw-style prototype board, the table is a compact
spreadsheet-like view where each row is one idea family, and big-bet grouping is
derived from the registry. Dragging a board node updates only the bet's priority
and implicit position; dependency edges stay explicit and do not change as a
side effect of layout. Dragging a sheet row updates the idea family's big-bet
membership and implicit position. Single-clicking a node or row selects it
without opening an editor; double-click, Enter, or an explicit edit action opens
the focused inspector. Inline cell edits, keyboard row moves, drag/drop moves,
inspector edits, canonical JSON edits, and snapshot restores all regenerate the
JSON, CSV, Markdown, Mermaid, SVG, and Excalidraw artifacts from the same
registry.

The generated app can open a focused inspector from table rows or graph nodes,
validate the same maintenance invariants, download regenerated artifacts, and
persist through `window.bigbetsStorageAdapter` when the host provides one.
Without an adapter it falls back to browser localStorage.

Storage adapters are deliberately pluggable. `bigbets site adapters` lists
built-ins. Use `--storage-adapter none` to omit `storage-adapter.js`, or
`--storage-adapter-file /path/to/storage-adapter.js` to copy a private
host-specific adapter beside the generated site without adding that adapter to
the `bigbets` package.

Static hosts may optionally expose `window.bigbetsStorageAdapter.fetchIssue`.
When present, the generated site can hydrate an idea-family row from an issue
URL without making `bigbets` depend on any specific issue tracker.

Static hosts may also expose snapshot methods:

- `listSnapshots({ appId })`
- `saveSnapshot({ appId, registry, name })`
- `loadSnapshot({ appId, snapshotId })`

When those methods are absent, the site stores snapshots in browser
localStorage. The CLI `bigbets snapshot create/list` commands provide a
host-neutral way to keep dated registry backups under `snapshots/`.

## Versioning

Registries may include `schema_version: bigbets.registry.v1`. The field is
optional for older unversioned registries, but recommended for any durable
portfolio source. If a registry declares an unsupported schema, `bigbets
validate` fails before rendering so stale inputs do not quietly generate
misleading views.

Every generated machine-readable artifact carries enough metadata to audit its
producer:

- normalized JSON includes `schema_version`, `artifact_schema_version`, and a
  `generator` object with name and version;
- CSV includes `schema_version` and `generator_version` columns;
- Markdown, Mermaid, Excalidraw, SVG, and HTML include embedded
  schema/generator metadata;
- static-site scaffolds additionally declare `bigbets.site.v1` in generated
  page metadata, `SITE.md`, and `big_bets.artifact_metadata.json`.

When a future schema changes, bump the corresponding version constant in
`bigbets.version`, add a migration or explicit rejection path, and regenerate
the affected artifacts. This makes stale meeting decks, static sites, and
registry snapshots easy to identify without coupling `bigbets` to any specific
host or issue tracker.

## Recommended Usage

Keep individual idea-family issues as the durable work/evidence units. Use
`bigbets` for the portfolio view:

1. update issue status tables and run-ledger comments after durable runs;
2. update the registry only when ranking, ownership, dependencies, or next
   actions change;
3. run `bigbets validate`;
4. render a fresh Markdown/SVG/HTML view;
5. deploy the rendered directory to a static host if a meeting-ready surface is
   needed.
