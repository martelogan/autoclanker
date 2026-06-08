# clankerprof

`clankerprof` is a language-neutral pprof CPU profile analyzer packaged with
`autoclanker`. It turns raw profile samples into a typed call graph and stable
artifacts that can survive handoff into issue seeds, benchmark loops, CI gates,
and code review.

<p align="center">
  <img src="assets/clankerprof-sample-facts-hero.png" width="840" alt="clankerprof call graph to sample facts visual">
</p>

The core strategy is call-graph first. Leaf frames often describe CPU
mechanics: `Object#new`, `String#gsub`, `JSON.parse`, native runtime work,
template execution, compression, or I/O clients. Those are useful clues, but
the actionable caller is usually higher in the stack: `HotelSearch#rank_results`,
`CalendarExport#to_json`, or `MapView#load_tiles`.

`clankerprof` keeps those views connected. It can show the low-level CPU
mechanics, then fold or attribute that cost back to the caller, boundary,
semantic bucket, or responsibility slice that made the work happen.

The same sample-facts model can be rendered as target-boundary reports,
responsibility slices, semantic caller exports, or before/after regression
gates.

<p align="center">
  <img src="assets/clankerprof-sample-facts.svg" width="760" alt="clankerprof sample-facts architecture">
</p>

## Why use it

- Decode raw and gzipped `.pb` profiles without generated protobuf runtime files.
- Preserve pprof IDs and inline frames in a typed call graph model.
- Explain a known parent boundary with complete target-function attribution.
- Opt into runtime rule packs such as Ruby core/native semantic labeling.
- Fold runtime-internal cost into meaningful callers when that answers the
  investigation question.
- Emit CSV and JSON artifacts that agents, CI gates, and review tools can
  consume without scraping terminal prose.
- Keep slice attribution separate from target attribution so ownership views do
  not redefine parent-boundary cost accounting.

## Command surfaces

| Command | Use it for | Primary output |
| --- | --- | --- |
| `clankerprof targets` | Explain CPU under one or more configured parent functions, request handlers, renderers, jobs, or RPC boundaries. | JSON, CSV, simple CSV, or text target reports. |
| `clankerprof slices` | Attribute selected CPU to path-based responsibility slices with filters, collapse rules, and metadata. | JSON slice reports for humans, agents, and compare gates. |
| `clankerprof compare` | Compare two slice JSON outputs with absolute and relative regression thresholds. | JSON delta report plus exit code `2` on regression. |
| `autoclanker pprof ...` | Run the same subcommands through the main `autoclanker` CLI. | Same payloads as `clankerprof`. |

## MVP contract

The current MVP is already useful as a standalone library and CLI. It is not yet
a claim that every historical profile report can be deleted without real-profile
golden verification.

The tested compatibility contract is tracked in
[`CLANKERPROF_PARITY.md`](CLANKERPROF_PARITY.md). Treat that file as the source
of truth for what compatibility is claimed versus intentionally not claimed.

The next phase should deepen the sample-facts engine: make the fact graph more
explicit as a reusable library API, add richer projection composition, and bring
additional runtime rule packs online without making any one runtime or ownership
system part of the core model.

## Quickstart

Use any pprof CPU profile exported as raw `.pb` or gzipped `.pb.gz`.

```bash
clankerprof targets \
  --profile profile.pb.gz \
  --config examples/clankerprof/target_config.json \
  --runtime ruby \
  --fold-runtime-internals \
  --track-semantic-callers \
  --semantic-callers-csv tmp/semantic-callers.csv \
  --format json \
  --output tmp/profile-targets.json

clankerprof slices \
  --profile profile.pb.gz \
  --config examples/clankerprof/clankerprof-slices.yml \
  --output tmp/profile-slices.json
```

See [`../examples/clankerprof`](../examples/clankerprof) for copyable target and
slice configs.

## Target Attribution

Use this mode when you know the parent function or request/rendering boundary
you want to explain.

```bash
clankerprof targets \
  --profile profile.pb.gz \
  --config target_config.json \
  --format csv \
  --output slices.csv
```

`target_config.json` maps parent function names to category path patterns:

```json
{
  "Target#render": {
    "Application": "app/**",
    "Cache Client": "library:cache-client"
  }
}
```

Every sample whose stack contains `Target#render` is attributed by the leaf
self-time frame. If no configured category matches, time goes to `Other`, so the
target total stays fully accounted for.

Prefer path patterns such as `app/**` or `app/components/**` in new configs.
Use `library:cache-client` or `dependency:cache-client` for versioned
third-party paths. `gem:` is retained as a compatibility selector for older
Ruby-oriented configs, and `regex:...` can make intentional regex matching
explicit.

For request-rendering investigations, use the same shape with the request
boundary as the parent and neutral categories for app code, component rendering,
client libraries, native engines, or data-shape work:

```json
{
  "RequestHandler#render_response": {
    "View Model": "app/view_models/**",
    "Components": "app/components/**",
    "Cache Client": "library:cache-client"
  }
}
```

This is intentionally framework-neutral: the parent can be an HTTP handler,
RPC method, background job, renderer, or any other stack frame that represents
the boundary you want to cost.

### Output Example: Boundary Cost

Use `simple-csv` when you want a compact summary for an issue, PR, or agent
handoff. If you pass `--cpu-attributables`, each category also gets a
proportional estimate for that metric.

```bash
cat > request_metrics.json <<'JSON'
{
  "p90_ms": {
    "RequestHandler#render_response": 96.0
  }
}
JSON

clankerprof targets \
  --profile profile.pb.gz \
  --config target_config.json \
  --runtime ruby \
  --fold-runtime-internals \
  --cpu-attributables request_metrics.json \
  --format simple-csv \
  --output tmp/request-summary.csv
```

`simple-csv` writes one CSV row per category. The same rows are wrapped below
as a table so the important shape is easy to scan:

| Parent | Category | CPU % | p90 ms est. | Main app callsites | Low-level work |
| --- | --- | ---: | ---: | --- | --- |
| `RequestHandler#render_response` | Components | 34.9 | 33.5 | `ComponentRenderer#render` 19.8%; `CardPresenter#render` 10.1% | `TemplateEngine::Native#render` 14.7%; `String#gsub` 8.4% |
| `RequestHandler#render_response` | I/O Overhead | 22.3 | 21.4 | `CacheClient#get_multi` 12.8%; `InventoryClient#batch_fetch` 9.5% | `Net::HTTP#request` 8.9%; `CacheClient#get_multi` 7.1% |
| `RequestHandler#render_response` | Serialization Overhead | 18.1 | 17.4 | `ResponseEnvelope#to_json` 9.3%; `Money#as_json` 5.7% | `JSON.generate` 10.8%; `Marshal.load` 3.9% |
| `RequestHandler#render_response` | Third-party Libraries | 13.6 | 13.1 | `TelemetryWrapper#call` 7.4%; `StatsClient#increment` 3.5% | `TraceSDK::Span#record` 7.4%; `StatsClient#increment` 3.5% |
| `RequestHandler#render_response` | Other | 11.1 | 10.7 | `PathHelpers#normalize` 4.2%; `Timezone#convert` 3.1% | `Object#new` 4.9%; `Hash#[]` 2.0% |

Interpretation:

- `CPU %` is sampled CPU share below the configured parent boundary.
- `p90_ms` is `CPU % * parent p90_ms`, so it is a prioritization estimate, not
  direct per-category latency measurement.
- `Main app callsites` tells you where in the application stack the cost was
  introduced.
- `Low-level work` tells you the CPU mechanics or library frames that burned the
  samples.

## Ruby Runtime Rules

Ruby support is opt-in and data-driven. Pass the runtime and the core-class CSV
instead of relying on hardcoded application or framework assumptions. The Ruby
rule pack declares stdlib paths, native path patterns, and library extraction
patterns such as versioned `gems/<name>-<version>` directories. If
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
caller fallback before category matching.

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

## Slice Attribution

Use this mode for slice-based responsibility or code-area views. This mode is
kept separate from target attribution: slice CLI/config semantics such as
duplicate scalar validation apply here, not to `clankerprof targets`.

```bash
clankerprof slices \
  --profile profile.pb.gz \
  --slices slices.yml \
  --filter "<name:Target#render" \
  --collapse "library:telemetry-client" \
  --attribute "library:renderer,to:rendering" \
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

Filters support `name:`, `path:`, `library:`, `dependency:`, and descendant
prefix `<`. `gem:` remains a compatibility selector for older Ruby-oriented
configs.
Collapse rules skip matching frames when choosing the attribution frame.
Attribute rules override slice assignment, including descendant rules such as
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
  - library:telemetry-client
  - library:trace-sdk
attribute:
  - name:TemplateEngine::Native,to:rendering-native
by_slice: 5
top: 10
show_paths: true
unattributed_libraries: 5
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

### Output Example: Responsibility Slices

Slice output is the compact ownership view. It answers which configured code
areas still carry cost after filters, collapse rules, and explicit attribution:

```json
{
  "tool": "clankerprof_slices",
  "summary": {
    "matching_time_ns": 128000000,
    "total_time_ns": 180000000,
    "matching_pct": 71.1
  },
  "slices": [
    {
      "name": "components",
      "time_ns": 52000000,
      "pct": 40.6,
      "metadata": {
        "owner": "rendering-platform"
      },
      "frames": [
        {
          "function": "ComponentRenderer#render",
          "filename": "/app/components/card.rb",
          "line": 43,
          "time_ns": 25400000,
          "pct": 19.8
        }
      ],
      "unattributed_libraries": []
    },
    {
      "name": "default",
      "time_ns": 17400000,
      "pct": 13.6,
      "is_default": true,
      "frames": [
        {
          "function": "TraceSDK::Span#record",
          "filename": "/vendor/trace-sdk-2.4.1/lib/span.rb",
          "time_ns": 7400000,
          "pct": 5.8
        }
      ],
      "unattributed_libraries": [
        {
          "name": "trace-sdk",
          "time_ns": 7400000,
          "pct": 5.8
        }
      ]
    }
  ]
}
```

The default slice is a triage aid. If code falls through configured slices,
`unattributed_libraries` makes unclaimed dependency CPU visible without forcing
the dependency concept into the core analyzer.

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

## Library Shape

The public Python modules are small enough to use directly when a CLI subprocess
is not the right integration point:

```python
from clankerprof.analysis import TargetAnalysisOptions, analyze_targets, load_json_mapping
from clankerprof.proto import load_profile
from clankerprof.render import render_target_json

profile = load_profile("profile.pb.gz")
config = load_json_mapping("examples/clankerprof/target_config.json")
results = analyze_targets(profile, config, TargetAnalysisOptions())
payload = render_target_json(results)
```

For advanced integrations, treat `Profile.stack_for_sample(...)` as the current
typed call graph seam. The next phase should make this seam more explicit as a
first-class sample-facts API, but the existing model already preserves the facts
that matter for safe projection: leaf-to-root stack order, sparse IDs, inline
frames, leaf self-time, target containment, slice labels, folded-from lineage,
and semantic callers.

## Integration Guidance

Prefer this order when adopting `clankerprof` in another harness:

1. Produce `targets --format json` for parent-boundary diagnosis.
2. Produce `slices --output profile-slices.json` for ownership or code-area
   triage.
3. Use `compare` only on slice JSON outputs generated from stable configs.
4. Keep runtime-specific behavior in rule packs and config files.
5. Keep old profile tools available until real-profile golden outputs match the
   compatibility contract you need.
