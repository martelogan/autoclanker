# clankerprof

`clankerprof` is a language-neutral pprof CPU profile analyzer packaged with
`autoclanker`. Its primary compatibility target is target-function attribution.
Slice attribution is a separate, additive migration surface:

- raw and gzipped `.pb` profile decoding;
- target-function attribution with an `Other` catch-all;
- optional Ruby runtime rules for native/core semantic labels;
- runtime-internal folding and proportional CPU attributables;
- slice/path attribution, filters, collapse rules, and attribution overrides;
- JSON outputs suitable for comparison gates and agent handoff artifacts.

The tested compatibility contract is tracked in
[`CLANKERPROF_PARITY.md`](CLANKERPROF_PARITY.md). Treat that file as the source
of truth for what compatibility is claimed versus intentionally not claimed.

## Target attribution

Use this mode when you know the parent function or request/rendering boundary
you want to explain.

```bash
clankerprof targets \
  --profile profile.pb.gz \
  --config target_config.json \
  --format csv \
  --output slices.csv
```

`target_config.json` maps parent function names to category regexes:

```json
{
  "Target#render": {
    "Application": "[/\\\\]app[/\\\\]",
    "Gems": "[/\\\\]gems[/\\\\]"
  }
}
```

Every sample whose stack contains `Target#render` is attributed by the leaf
self-time frame. If no configured category matches, time goes to `Other`, so the
target total stays fully accounted for.

For request-rendering investigations, use the same shape with the request
boundary as the parent and neutral categories for app code, component rendering,
client libraries, native engines, or data-shape work:

```json
{
  "RequestHandler#render_response": {
    "View Model": "[/\\\\]app[/\\\\]view_models[/\\\\]",
    "Components": "[/\\\\]app[/\\\\]components[/\\\\]",
    "Cache Client": "[/\\\\]gems[/\\\\]cache-client[/\\\\]"
  }
}
```

This is intentionally framework-neutral: the parent can be an HTTP handler,
RPC method, background job, renderer, or any other stack frame that represents
the boundary you want to cost.

## Ruby runtime rules

Ruby support is opt-in and data-driven. Pass the runtime and the core-class CSV
instead of relying on hardcoded application or framework assumptions. If
`--ruby-core-classes` is omitted, `clankerprof` uses the packaged Ruby core
class list:

```bash
clankerprof targets \
  --profile ruby-profile.pb.gz \
  --config target_config.json \
  --runtime ruby \
  --ruby-core-classes ruby_core_classes.csv \
  --fold-ruby-internals \
  --format simple-csv \
  --output ruby-slices.csv
```

The Ruby rules classify common native/core frames such as `String#gsub`,
`Marshal.load`, `JSON.parse`, OpenTelemetry, StatsD, I/O clients, and
serialization/compression helpers. Non-verbose mode rolls these into broad
overhead families; `--verbose-ruby-internals` keeps raw categories and folds the
verbose-only native categories when folding is enabled.

For compatibility with older target-attribution runs, pass `--no-enhanced`.
That disables semantic runtime labels and uses the legacy native/delegated
caller fallback before category regex matching.

To reproduce the old two-file CSV artifact layout from
`--format csv --output slices.csv`, add:

```bash
clankerprof targets \
  --profile ruby-profile.pb.gz \
  --config target_config.json \
  --runtime ruby \
  --format csv \
  --output slices.csv \
  --legacy-target-csv-layout
```

That writes `output/slices.csv` and `output/verbose/slices.csv`, matching older
two-file target report artifact locations while keeping ordinary `csv` and
`simple-csv` as explicit single-output formats.

## Slice attribution

Use this mode for slice-based responsibility or code-area views. This mode is
kept separate from target attribution: slice CLI/config semantics such as
duplicate scalar validation apply here, not to `clankerprof targets`.

```bash
clankerprof slices \
  --profile profile.pb.gz \
  --slices slices.yml \
  --filter "<name:Target#render" \
  --collapse "gem:statsd-instrument" \
  --attribute "gem:renderer,to:rendering" \
  --output profile-slices.json
```

Slice definitions use path patterns:

```yaml
slices:
  - name: app
    paths:
      - app/**
    metadata:
      owner: rendering-platform
      docs:
        - https://example.invalid/rendering
    contacts:
      - "#rendering-performance"
  - name: default
    default: true
```

Filters support `name:`, `path:`, `gem:`, and descendant prefix `<`. Collapse
rules skip matching frames when choosing the attribution frame. Attribute rules
override slice assignment, including descendant rules such as
`<name:GraphQL::Execute,to:graphql`.
Unknown slice keys are preserved as generic JSON-compatible `metadata` in
slice output. A nested `metadata:` object is flattened into that same payload,
so callers can attach labels, contacts, docs, escalation hints, or any other
domain metadata without teaching `clankerprof` application-specific concepts.

Common slice options can also live in a YAML config:

```yaml
slices: ./slices.yml
filters:
  - <name:RequestHandler#render_response
collapse:
  - gem:statsd-instrument
  - gem:opentelemetry-sdk
attribute:
  - name:TemplateEngine::Native,to:rendering-native
by_slice: 5
top: 10
show_paths: true
```

Run it with:

```bash
clankerprof slices --profile profile.pb.gz --config clankerprof-slices.yml
```

CLI array flags such as `--filter`, `--collapse`, and `--attribute` append to
config-file arrays. Single-value fields such as `slices`, `top`, and `by_slice`
fail if supplied in both places so analysis inputs do not drift silently.
TOML config files are also accepted, and
`./slices.yml` is discovered automatically when slice-aware options are used.
Attribute targets must name a configured slice by default so typos are caught
early. If you intentionally want an output-only virtual slice, pass
`--allow-virtual-attribute-slices`.

## Compare

Compare two JSON slice outputs in CI or in a review artifact:

```bash
clankerprof compare \
  --before before.json \
  --after after.json \
  --threshold-abs 2.0 \
  --threshold-rel 15.0
```

The command returns JSON with `has_regression`, per-slice deltas, and top
function regressions/improvements. The CLI exits `2` when a regression exceeds
the configured thresholds for gate-friendly automation. `autoclanker pprof ...`
exposes the same subcommands as a convenience alias.
