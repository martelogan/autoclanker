# clankerprof sample-facts specification

This document is the normative contract for `clankerprof` as a Python package
and as a future cross-language profile-analysis core. Implementation details
can change, but the fact shape and projection semantics below are the
compatibility target.

## Design goals

- Decode one pprof profile into one durable sample-facts model.
- Keep target-boundary attribution and slice attribution as projections over
  the same facts, not as separate decoders with separate accounting.
- Keep runtime and dependency knowledge declarative through rule packs.
- Keep domain ownership metadata declarative through slice config.
- Emit machine-readable artifacts that can be compared by tests, CI, agents, or
  another implementation in a different language.

## Decoding contract

`clankerprof` accepts raw `.pb` and gzipped `.pb.gz` pprof profiles. The decoder
must preserve:

- sparse pprof function IDs and location IDs;
- sample order and stable zero-based `sample_index`;
- all sample values, with `values[0]` as the primary CPU value;
- pprof leaf-to-root location order;
- all inline frames from each location, in pprof line order;
- function name, system name, filename, start line, sample line, and folded
  location marker when present.

The decoder must not assume IDs are contiguous array indexes. Optimized
implementations can build indexes, but indexes must be derived from profile IDs
without changing visible facts.

## Sample-facts JSON

`clankerprof facts --profile profile.pb.gz` exports the stable JSON shape. The
current schema version is:

```json
"clankerprof.sample_facts.v1"
```

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exact schema identifier. Readers must reject unknown versions. |
| `tool` | Producer identifier, currently `clankerprof_facts`. |
| `summary.sample_count` | Number of decoded samples. |
| `summary.empty_sample_count` | Samples whose stack could not be decoded. |
| `summary.non_empty_sample_count` | Decoded samples with at least one frame. |
| `summary.total_primary_value` | Sum of primary sample values. |
| `samples` | Ordered sample fact array. |

Each sample contains:

| Field | Meaning |
| --- | --- |
| `sample_index` | Stable zero-based index from the original profile order. |
| `primary_value` | First sample value, used as CPU time by current projections. |
| `values` | All raw sample values. |
| `location_ids` | Raw pprof location IDs for the sample. |
| `is_empty` | Whether the decoded stack is empty. |
| `stack` | Leaf-to-root decoded frames. |

Each frame contains:

| Field | Meaning |
| --- | --- |
| `location_id` | Original pprof location ID. |
| `function_id` | Original pprof function ID. |
| `name` | Function name. |
| `filename` | Source or pseudo path. |
| `line` | Sample line number when available, otherwise `0`. |
| `location_is_folded` | pprof folded-location marker. |

Round-tripping this JSON through `loads_sample_facts` must preserve target and
slice projection outputs.

## Fact index contract

`ProfileFactIndex` is a convenience layer over `ProfileFacts`. It can speed up
or centralize shared stack operations, but it must not own projection policy.

The index may provide:

- target-frame lookup for a sample;
- leaf, caller, descendant, and arbitrary frame predicate helpers;
- bottom-frame selection from caller-supplied eligibility and collapse
  predicates;
- total primary value and empty-stack accounting.

The index must not own:

- runtime labels;
- dependency extraction rules;
- target category matching;
- slice ownership;
- collapse or filter syntax;
- output formatting.

Those remain projection or rule-pack concerns.

## Runtime rule contract

Runtime rule packs own language- or runtime-specific interpretation:

- semantic function labels;
- native/core/stdlib detection;
- runtime-internal folding categories;
- ordered native-name rules for folded native callers whose pseudo path cannot
  carry the semantic category;
- simplified category maps;
- dependency/library path extraction;
- selector-specific dependency path extraction for names such as `library:`,
  `dependency:`, `package:`, `vendor:`, or project-local selector names;
- caller fallback prefixes used when semantic runtime categorization is
  intentionally disabled;
- compatibility aliases.

The generic runtime must remain neutral. New configs should prefer
`library:`, `dependency:`, `package:`, `vendor:`, or selector names declared in
the active runtime rule pack. Runtime-specific selectors such as `gem:` can
remain as compatibility aliases, but they are not special to the fact model.

Callers must be able to load runtime rule packs from project-local YAML files.
Packaged rule packs are conveniences, not the extension boundary. This keeps
domain-specific categories, native labels, dependency-path conventions, and
foldability decisions outside the core package while preserving the same
projection machinery and tests.

Rule-pack schema keys are intentionally declarative. Prefer
`caller_fallback_name_prefixes` for native/delegated caller fallback behavior
and `library_selector_path_patterns` for selector-specific dependency paths.
Older alias keys may remain accepted for migration, but aliases must normalize
to the same `RuntimeRuleSet` fields before analysis begins.

## Target projection contract

`targets` answers: “inside this parent boundary, what leaf work consumed CPU,
and which higher-level caller made it matter?”

For each sample:

- find every configured parent frame contained in the stack;
- attribute the full primary sample value to each matching parent;
- start from the leaf frame as the self-time owner;
- optionally label the leaf using runtime rules;
- optionally fold runtime-internal leaves into the first meaningful caller;
- match configured target categories against the chosen frame path;
- place unmatched time in `Other`;
- track leaf functions, categorized files, folded-from totals, semantic callers,
  and caller-to-leaf pairs.

Target projection must preserve complete target accounting. Category totals for
one parent must sum to the total CPU time under that parent.

## Slice projection contract

`slices` answers: “after filters and collapse rules, which responsibility area
owns this sample?”

For each sample:

- choose a bottom attribution frame from leaf to root;
- skip native/runtime frames unless `--no-collapse-native` is set;
- skip collapsed frames when selecting the bottom frame;
- if every eligible frame is collapsed, count the sample and report an
  uncollapsible pseudo-output;
- apply non-descendant filters conjunctively to the selected bottom frame;
- apply descendant filters as an OR across the stack;
- apply attribute rules before path-based slice matching;
- use the configured default slice or `(all)` for unmatched samples;
- report GC pseudo-slice time for `(marking)` and `(sweeping)`;
- report unattributed dependency/library summaries for default-slice time when
  requested.

Slice projection must not redefine target-boundary semantics.

## Compare contract

`compare` consumes two slice JSON payloads. It reports:

- total before/after primary time from summaries;
- per-slice before percent, after percent, absolute delta, relative delta, and
  status;
- per-function deltas within each slice;
- top regressions and improvements;
- `has_regression`;
- exit code `2` for regression through the CLI.

A slice is a regression only when it exceeds both the configured absolute and
relative thresholds and is within the focus set when one is provided.

## Compatibility and validation

Public tests must cover:

- raw and gzipped pprof decoding;
- sparse pprof IDs;
- inline frames;
- folded locations;
- sample-facts JSON round-trip;
- fact-index helpers used by projections;
- target output stability from profile facts and imported facts;
- slice output stability from profile facts and imported facts;
- runtime folding, semantic callers, attributables, and compatibility aliases;
- external runtime rule-pack loading without package changes;
- slice filters, collapse, attributes, pseudo-slices, and compare gates.

Real-profile parity must be optional and local-only. The helper under
`scripts/clankerprof` compares caller-provided local reference artifacts against
current `clankerprof` output and intentionally commits no profile data. Its
`--check-rust-core` mode additionally compares Rust facts and generic target
projection output against Python for the same caller-provided local inputs.

## Rust core parity

`crates/clankerprof-core` is the Rust compatibility implementation for this
spec. The crate must treat Python `clankerprof` as the reference until the Rust
core has equivalent coverage for every public projection. Its stable surfaces
currently include `clankerprof-rs facts`, generic `targets`, generic `slices`,
and `compare`. The facts command emits the same `clankerprof.sample_facts.v1`
payload as Python for raw profiles, gzipped profiles, inline frames, folded
locations, and sparse pprof IDs.

Projection work must build on the same Rust fact model rather than introducing
separate tree or opportunity accounting. Any future Rust target, slice, compare,
tree, or opportunity command must prove parity against Python fixtures before
being used as an integration boundary by downstream tools. Runtime-specific
rule-pack parity remains the expansion point after the generic projection
surface; downstream integrations should keep calling this out explicitly when
they depend on runtime-specific folding or semantic categorization.
