# clankerprof

`clankerprof` turns pprof CPU profiles into a typed call graph and a small set
of durable reports: target-boundary cost ledgers, scope decomposition,
responsibility slices, semantic caller exports, and before/after regression
gates.

<p align="center">
  <img src="./sample-facts-hero.png" width="840" alt="clankerprof call graph to sample facts visual">
</p>

`clankerprof` is call-graph first: keep the sampled stack intact, then choose
the attribution level that answers the question.

Leaf frames often describe CPU mechanics: `Object#new`, `String#gsub`,
`JSON.parse`, native runtime work, template execution, compression, or I/O
clients. Those are useful clues, but the actionable caller is usually higher in
the stack: `HotelSearch#rank_results`, `CalendarExport#to_json`, or
`MapView#load_tiles`.

The tool keeps those views connected. It can show the low-level CPU mechanics,
then fold or attribute that cost back to the caller, measured scope, semantic
rollup, owner frame, or responsibility slice that made the work happen. New
configs should prefer `scope`, `cost_kind`, `rollup`, `owner`, `runtime_label`,
and `selector`; older `boundary`, `category`, `bucket`, `domain`,
`runtime_category`, and `match` names remain accepted compatibility aliases.

## Start With Defaults

Start with a broad profile read. Add config only when the useful projection is
clear.

```bash
# Broad "what frames own CPU?" view. No slice config required.
clankerprof slices --profile profile.pb.gz

# Explain CPU below one known parent frame. No JSON config required.
clankerprof targets \
  --profile profile.pb.gz \
  --target HotelSearch#rank_results

# Explain one parent denominator with rollups, cost kinds, and owners.
clankerprof scopes \
  --profile profile.pb.gz \
  --config examples/clankerprof/scopes.toml

# Write stable JSON once the query is useful.
clankerprof slices \
  --profile profile.pb.gz \
  --output tmp/profile-slices.json

# Export decoded facts once, then replay several projections from that artifact.
clankerprof facts \
  --profile profile.pb.gz \
  --output tmp/profile-facts.json
clankerprof targets \
  --facts tmp/profile-facts.json \
  --target HotelSearch#rank_results
```

Add config for stable labels, path categories, filters, collapse rules, runtime
semantics, or metadata:

```bash
clankerprof targets --profile profile.pb.gz --config target_config.json
clankerprof scopes --profile profile.pb.gz --config scopes.toml
clankerprof slices --profile profile.pb.gz --config clankerprof-slices.yml
clankerprof slices --facts tmp/profile-facts.json --config clankerprof-slices.yml
```

The same commands are also available through the umbrella CLI:

```bash
autoclanker pprof slices --profile profile.pb.gz
autoclanker pprof targets \
  --profile profile.pb.gz \
  --target HotelSearch#rank_results
autoclanker pprof facts --profile profile.pb.gz --output tmp/profile-facts.json
```

## How It Thinks

`clankerprof` decodes pprof samples into one sample-facts model, then projects
that model into the view you need: target cost, slice ownership, semantic
callers, or compare gates.

## What It Answers

- Which parent boundary accumulated this CPU?
- Is the cost allocation, string processing, serialization, I/O,
  instrumentation, template execution, or application logic?
- Within one parent denominator, which owner domain drove each cost kind?
- Which responsibility slice carries the cost after filters, collapse rules,
  and attribution rules?
- Did a before/after run regress a focused slice enough to fail a benchmark or
  CI gate?

## Inputs and Outputs

`clankerprof` is intentionally boring at the edges:

| Input | Purpose |
| --- | --- |
| `profile.pb` or `profile.pb.gz` | pprof CPU profile for direct decoding. |
| sample-facts JSON | Versioned decoded profile surface accepted by `--facts`. |
| `--target <function>` | Minimal target mode for explaining CPU below a known parent frame. |
| `target_config.json` | Optional richer target categories for path-based attribution. |
| `scopes.toml` | Scope decomposition config with cost kinds, rollups, owners, and same-scope attributables. |
| `boundaries.toml` | Compatibility name for older scope decomposition configs. |
| `clankerprof-slices.yml` / `slices.yml` | Optional slice ownership, filters, collapse rules, and metadata. |
| runtime rule pack | Optional semantic labels and runtime-internal folding. |

| Output | Purpose |
| --- | --- |
| target JSON/CSV/text | Parent-boundary cost ledger. |
| boundary JSON | Compatible payload for parent scopes split into rollups, atomic cost kinds, owners, and evidence. |
| slice JSON | Responsibility view with top frames and metadata. |
| sample-facts JSON | Stable decoded facts that target and slice projections can replay. |
| semantic callers CSV | Runtime/native leaves such as `Object#new` mapped back to callers. |
| compare JSON + exit code | Before/after regression gate for benchmark or CI use. |

## Architecture

The core model preserves pprof IDs, mappings, inline frames, sample values, and
call paths. Runtime behavior is loaded from rule packs. Domain ownership or
responsibility labels are loaded from slice configs. Neither concept is baked
into the profile decoder.

<p align="center">
  <img src="./sample-facts.svg" width="760" alt="clankerprof sample-facts architecture">
</p>

## Why This Shape

Profile tooling tends to drift when one report mode owns the whole analysis
strategy. `clankerprof` keeps the pieces separate:

| Layer | Responsibility | Why it matters |
| --- | --- | --- |
| Decode | Parse raw and gzipped pprof payloads without generated protobuf files. | Profiles remain portable across environments and package builds. |
| Sample facts | Represent samples, frames, mappings, and paths with typed objects. | Every projection works from the same accounting surface. |
| Runtime rules | Label native/core frames and fold runtime internals when requested. | Language-specific insight stays data-driven and opt-in. |
| Projections | Render target, slice, semantic-caller, and compare outputs. | Different investigation questions do not redefine the core facts. |
| Artifacts | Emit JSON, CSV, and text output with stable fields and exit codes. | Agents, CI, and review tools can consume evidence without scraping prose. |

## Command Map

```bash
clankerprof targets --profile profile.pb.gz --target TripPlanner#rank_itineraries
clankerprof scopes --profile profile.pb.gz --config scopes.toml
clankerprof slices --profile profile.pb.gz --config clankerprof-slices.yml
clankerprof facts --profile profile.pb.gz --output profile-facts.json
clankerprof slices --facts profile-facts.json --config clankerprof-slices.yml
clankerprof compare --before before-slices.json --after after-slices.json
```

The same commands are exposed through the umbrella CLI:

```bash
autoclanker pprof targets --profile profile.pb.gz --target TripPlanner#rank_itineraries
autoclanker pprof slices --profile profile.pb.gz --config clankerprof-slices.yml
autoclanker pprof facts --profile profile.pb.gz --output profile-facts.json
autoclanker pprof compare --before before-slices.json --after after-slices.json
```

## Target Attribution

Use `targets` when the question starts with a known boundary: a request handler,
background job, renderer, RPC method, compiler pass, query executor, or any
other parent frame that represents the work you want to explain.

```bash
clankerprof targets \
  --profile profile.pb.gz \
  --target TripPlanner#rank_itineraries \
  --format json \
  --output tmp/profile-targets.json
```

That reports all cost under the target as `Other`, which is useful for a first
pass because it proves the boundary and total before adding categories.

Add a target config when you want stable category labels:

```json
{
  "TripPlanner#rank_itineraries": {
    "Ranking": "app/trip_ranking/**",
    "Availability": "lib/availability/**",
    "Maps": "lib/maps/**",
    "Cache Client": "library:cache-client"
  }
}
```

Every sample whose stack contains the configured parent is attributed by the
leaf self-time frame. If no category matches, the time remains accounted for as
`Other`; target totals do not silently disappear.

Prefer path patterns such as `app/trip_ranking/**` or `lib/maps/**` in new
target configs. Use `library:cache-client` or `dependency:cache-client` for
versioned third-party paths. `gem:` is retained as a compatibility selector for
older Ruby-oriented configs, and `regex:...` can make intentional regex
matching explicit when a category genuinely needs it.

### Reading Target Output

For a crisp first-pass diagnosis, render target attribution as `simple-csv`.
It keeps the fields people usually need during performance triage:
category-level CPU share, a proportional latency estimate, and the hottest
application callsites or leaf functions that explain that share.

```bash
cat > request_metrics.json <<'JSON'
{
  "p90_ms": {
    "TripPlanner#rank_itineraries": 142.0
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
  --output tmp/target-summary.csv
```

`simple-csv` writes one CSV row per category (categories other than `Other`
whose share magnitude is below 0.1% are omitted as noise; negative shares of
any magnitude are rendered). The same rows are wrapped below
as a table so the signal is visible in a README:

| Parent | Category | CPU % | p90 ms est. | Main app callsites | Low-level work |
| --- | --- | ---: | ---: | --- | --- |
| `TripPlanner#rank_itineraries` | Ranking | 38.4 | 54.5 | `TripRanker#score` 21.0%; `CrowdCalendar#blend` 10.3% | `RankingModel#score` 16.4%; `Array#sort_by` 9.2% |
| `TripPlanner#rank_itineraries` | I/O Overhead | 24.8 | 35.2 | `WeatherClient#forecast_batch` 18.9%; `CacheClient#get_multi` 5.9% | `Net::HTTP#request` 12.1%; `CacheClient#get_multi` 5.9% |
| `TripPlanner#rank_itineraries` | Serialization Overhead | 17.6 | 25.0 | `ItineraryPresenter#as_json` 9.6%; `TicketBundle#to_json` 8.0% | `JSON.generate` 10.4%; `String#gsub` 4.1% |
| `TripPlanner#rank_itineraries` | Third-party Libraries | 11.5 | 16.3 | `TelemetryWrapper#call` 7.2%; `Flags#enabled?` 4.3% | `TraceSDK::Span#record` 6.8%; `StatsClient#increment` 3.1% |
| `TripPlanner#rank_itineraries` | Other | 7.7 | 10.9 | `RouteHelpers#normalize` 4.0%; `Timezone#convert` 3.7% | `Object#new` 3.5%; `Hash#[]` 2.4% |

`p90_ms` is computed as `category CPU share * p90_ms` for the parent boundary.
That makes it useful for prioritization: the example says ranking owns 38.4%
of sampled CPU below `TripPlanner#rank_itineraries`, which corresponds to
about 54.5 ms of a 142 ms p90 request if CPU share tracks the request latency
shape. It is not a direct per-category latency measurement.

Use full `csv` or `text` output when you need deeper callsite evidence. Full
CSV includes sample counts, file counts, top caller-to-leaf pairs, and both raw
nanoseconds and human-readable time.

## Scope Decomposition

Use `scopes` when the investigation needs more structure than a flat target
category table: one parent denominator, scope-specific display rollups, atomic
cost kinds, optional owner rows, and proportional attributable estimates.

```bash
clankerprof scopes \
  --profile profile.pb.gz \
  --config examples/clankerprof/scopes.toml \
  --output tmp/scopes.json
```

Minimal TOML shape:

```toml
[cost_kind]
"Components" = "path:app/components/**"
"Cache Client" = "library:cache-client"
"Serialization" = ["name:JSON", "name:MessagePack"]
"Application outside components" = { all = ["path:app/**"], not = "path:app/components/**" }

[owner]
"Rendering" = ["path:app/components/**", "path:app/view_models/**"]
"Application fallback" = { patterns = ["path:app/**"], fallback = true }

[[scope]]
label = "Request render"
function = "RequestHandler#render_response"

[scope.attributables]
p90_ms = 142.0

[scope.rollup]
"Application code" = ["Components"]
"Mechanics" = ["Cache Client", "Serialization"]
```

Terminology:

| Concept | Meaning |
| --- | --- |
| Scope | The stack frame that defines the denominator. |
| Cost kind | The atomic kind of work sampled at the leaf or folded caller. |
| Rollup | Scope-specific display grouping of cost kinds. |
| Owner | Observed frame below the scope; owner rows preserve cost-kind sub-buckets. |
| Slice | Optional path ownership source that `owner` predicates can reference with `slice:<name>`. |

Predicate values can be a string, an array of strings with OR semantics, or a
table with `any`, `all`, and `not`. Predicate keys include `name:`,
`name_eq:`, `path:`, `regex:`, `native`, dependency selectors such as
`library:`, `slice:<name>` when a slice file is loaded, `cost_kind:<label>` for
configured cost-kind labels, and `runtime_label:<label>` for labels produced by
the selected runtime rule pack. Compatibility aliases remain accepted:
`category:<label>` for `cost_kind:<label>`, `runtime_category:<label>` for
`runtime_label:<label>`, `[category]` for `[cost_kind]`, `[domain]` for
`[owner]`, `[[boundary]]` for `[[scope]]`, and `[boundary.bucket]` for
`[scope.rollup]`. Cost-kind definitions cannot use `cost_kind:` or `category:`
recursively; use rollups for display groupings.

The hot loop caches predicate matches per unique frame and streams each sample
stack leaf-to-root once. That keeps the common shape close to
`O(samples x frames + unique_frames x configured_predicates)`, while still
emitting owner files and caller -> hot leaf evidence for the rows that were
actually observed.

## Slice Attribution

Use `slices` when the question is responsibility-oriented: which component,
library, subsystem, or code area owns selected CPU after filters and collapse
rules?

```bash
clankerprof slices --profile profile.pb.gz

clankerprof slices \
  --profile profile.pb.gz \
  --config clankerprof-slices.yml \
  --output tmp/profile-slices.json
```

Slice config:

```yaml
slices: ./slices.yml
filters:
  - <name:TripPlanner#rank_itineraries
collapse:
  - library:telemetry-client
attribute:
  - name:MapTileEngine::Native,to:maps-native
by_slice: 5
top: 10
show_paths: true
```

Slice definitions:

```yaml
slices:
  - name: search-ranking
    paths:
      - app/trip_ranking/**
      - lib/availability/**
    metadata:
      owner: journey-platform
      docs:
        - https://example.invalid/search
  - name: default
    default: true
```

Unknown slice keys are preserved as generic metadata in the JSON output, so
callers can attach owners, contacts, docs, escalation hints, or any other
domain-specific labels without changing `clankerprof`.

### Reading Slice Output

Slice JSON is better when the question is "which code area owns this cost after
filters and collapse rules?" It keeps summary percentages, top frames, metadata,
and optional unattributed dependency lists in one stable payload:

```json
{
  "tool": "clankerprof_slices",
  "summary": {
    "matching_time_ns": 91000000,
    "total_time_ns": 128000000,
    "matching_pct": 71.1
  },
  "slices": [
    {
      "name": "search-ranking",
      "time_ns": 42000000,
      "pct": 46.2,
      "metadata": {
        "owner": "journey-platform"
      },
      "frames": [
        {
          "function": "TripRanker#score",
          "filename": "/app/trip_ranking/ranker.rb",
          "line": 84,
          "time_ns": 21000000,
          "pct": 23.1
        }
      ],
      "unattributed_libraries": []
    },
    {
      "name": "default",
      "time_ns": 16000000,
      "pct": 17.6,
      "is_default": true,
      "frames": [
        {
          "function": "TraceSDK::Span#record",
          "filename": "/vendor/trace-sdk-2.4.1/lib/span.rb",
          "time_ns": 6800000,
          "pct": 7.5
        }
      ],
      "unattributed_libraries": [
        {
          "name": "trace-sdk",
          "time_ns": 6800000,
          "pct": 7.5
        }
      ]
    }
  ]
}
```

The default slice is intentionally useful: if third-party or vendor code falls
through your configured slices, `unattributed_libraries` gives you a quick list
of dependencies still carrying CPU that nobody has explicitly claimed.

## Runtime Rules

The core analyzer is not Ruby-specific, Go-specific, Python-specific, or tied
to one ownership system. Runtime behavior is opt-in:

```bash
clankerprof targets \
  --profile ruby-profile.pb.gz \
  --config target_config.json \
  --runtime ruby \
  --fold-runtime-internals \
  --track-semantic-callers \
  --semantic-callers-csv tmp/semantic-callers.csv
```

The packaged Ruby rule pack can label native/core frames, recognize Ruby
library paths, and fold runtime-internal cost into meaningful callers.
Project-local runtimes can use the same machinery without changing the
package:

```bash
clankerprof targets \
  --profile service-profile.pb.gz \
  --config target_config.json \
  --runtime-rules runtime-rules.yml \
  --core-classes core_classes.csv \
  --fold-runtime-internals
```

Use `--runtime-rules` for semantic labels, native path markers, dependency path
extraction, selector-specific dependency paths, caller fallback prefixes,
simplification maps, and foldable categories that are specific to your
application or language runtime. Packaged runtimes are conveniences, not the
extension boundary.

Rule order is intentional. If your runtime reports dynamically created
constructors as native frames, put the most specific folded-caller labels
before broad fallbacks:

```yaml
native_name_category_rules:
  - category: Presentation Model
    name_patterns:
      - '^(?=.*View)[A-Z][A-Za-z0-9_]*(::[A-Z][A-Za-z0-9_]*)*\\.new$'
  - category: Application
    name_patterns:
      - '^[A-Z][A-Za-z0-9_]*(::[A-Z][A-Za-z0-9_]*)*\\.new$'
library_selector_path_patterns:
  plugin:
    - regex:/plugins/([^/]+)/
caller_fallback_name_prefixes:
  - Delegator.
```

Use `--target-csv-layout compat` only when reproducing the two-file
`output/<name>` and `output/verbose/<name>` artifact layout from older target
reports. The older `--legacy-target-csv-layout` flag is still accepted as an
alias for existing scripts.

## Compare Gates

`compare` consumes two slice or boundary JSON reports and fails with exit code
`2` when a focused row crosses both absolute and relative thresholds:

```bash
clankerprof compare \
  --before tmp/before.json \
  --after tmp/after.json \
  --threshold-abs 2.0 \
  --threshold-rel 15.0
```

This makes profile evidence usable in CI or benchmark harnesses without
depending on terminal-only summaries.

## Library Map

| Module | Purpose |
| --- | --- |
| `proto.py` | Minimal pprof protobuf decoder. |
| `model.py` | Typed profile, sample, frame, location, function, and mapping models. |
| `facts.py` | Versioned sample-facts JSON, import/export helpers, and projection-neutral indexes. |
| `analysis.py` | Call graph construction plus target, boundary, and slice attribution. |
| `rules.py` | Runtime rule packs, slice configs, filters, collapse rules, and attribute rules. |
| `compare.py` | Before/after slice and boundary delta checks. |
| `render.py` | JSON, CSV, simple CSV, and text renderers. |
| `cli.py` | `clankerprof` command surface. |

## Design Contract

- Decode once; project many times.
- Let CLI and library callers replay target, boundary, and slice projections
  from the same versioned fact artifact.
- Keep target accounting and slice attribution separate.
- Preserve raw profile identity while adding semantic labels as projections.
- Keep runtime and ownership knowledge declarative.
- Prefer stable JSON and CSV artifacts over terminal prose.
- Treat compatibility output shapes as explicit modes, not hidden defaults.

## More

- Full guide: `../docs/CLANKERPROF.md`
- Compatibility notes: `../docs/CLANKERPROF_PARITY.md`
- Example configs: `../examples/clankerprof/`
