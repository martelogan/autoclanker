# clankerprof sample-facts unification status

`clankerprof` now has one decoded sample-facts core and several projections.
This document records the implemented seam so target attribution, scope
decomposition, slice attribution, semantic caller exports, and compare gates
keep shared stack facts without merging their distinct accounting policies.

For the normative artifact shape and projection semantics, see
[`CLANKERPROF_SPEC.md`](CLANKERPROF_SPEC.md). For tested compatibility status,
see [`CLANKERPROF_PARITY.md`](CLANKERPROF_PARITY.md).

## Implemented

- `Profile.to_sample_facts()` emits a `ProfileFacts` aggregate with stable
  sample indexes, primary CPU values, raw sample values, leaf-to-root frames,
  inline frames, pprof IDs, folded-location markers, total primary value, and
  empty-stack accounting.
- `clankerprof.facts` exports versioned sample-facts JSON with strict schema
  version checks, summary validation, import/export helpers, and a
  projection-neutral `ProfileFactIndex`.
- `clankerprof facts --profile ...` writes the JSON fact artifact from raw or
  gzipped pprof input.
- `clankerprof targets`, `clankerprof scopes`/`boundaries`, and `clankerprof slices`
  accept either `--profile` or `--facts`, so a decoded fact artifact can be
  replayed by multiple projections.
- `autoclanker pprof ...` exposes the same fact export and fact replay surface.
- Target, boundary, and slice library APIs consume `ProfileFacts` or iterables
  of `SampleFact` through `analyze_target_facts(...)`,
  `analyze_boundary_facts(...)`, and `analyze_slice_facts(...)`.

## Fact Core Boundary

The core sample-facts layer owns decoded pprof facts only:

- raw `Sample` identity and all sample values;
- primary sample value used as CPU time;
- leaf-to-root frame order;
- expanded inline frames from each pprof location;
- profile, location, and function IDs;
- function name, file path, line number, and location folded marker;
- total profile CPU time;
- empty-stack accounting.

The core does not own:

- runtime-specific labels;
- target category matching;
- slice ownership;
- scope rollup/owner decomposition;
- collapse/filter semantics;
- domain metadata such as owners, docs, contacts, or escalation hints;
- terminal formatting or comparison thresholds.

Those remain projection, rule-pack, config, or rendering concerns.

## Projection Boundary

Target attribution owns parent-boundary accounting:

- find every target frame contained in a sample stack;
- attribute sample self-time by the leaf frame or folded caller;
- preserve `Other` catch-all accounting;
- preserve runtime-rule folding, semantic callers, folded-from summaries,
  caller-to-leaf pairs, compatibility fallbacks, proportional attributables,
  and explicit compatibility CSV layout.

Slice attribution owns bottom-frame accounting:

- choose the first non-native, non-collapsed eligible frame unless
  `no_collapse_native` is set;
- apply bottom filters conjunctively and descendant filters as OR;
- support `name:`, `path:`, `library:`, `dependency:`, `package:`, `vendor:`,
  rule-pack selector keys, compatibility selector aliases, and `slice:` filter
  keys where valid;
- support collapse rules and explicit attribute rules;
- preserve default-slice unattributed dependency summaries, GC pseudo-slices,
  uncollapsible pseudo-output, by-slice limits, and compare JSON compatibility.

Scope decomposition owns richer parent-denominator accounting:

- keep cost-kind totals additive under each configured scope;
- group cost kinds into scope-specific display rollups;
- optionally map the same samples to owner frames below the scope;
- preserve cost-kind sub-buckets under each owner;
- support residual scopes with descendant exclusions;
- scale caller-supplied attributables proportionally without inferring metrics
  from pprof samples.

Runtime rules own language-specific interpretation:

- semantic labels;
- simplification maps;
- foldable runtime categories;
- stdlib/native path detection;
- library/dependency path extraction and selector-specific dependency paths;
- compatibility aliases such as `gem:`.

## Why This Seam

The projection styles answer different questions:

- target attribution asks: "inside this parent boundary, what leaf work
  consumed CPU, and which higher-level caller made that leaf work happen?"
- scope decomposition asks: "inside this parent denominator, which cost kinds
  and owners explain the CPU?"
- slice attribution asks: "after filtering and collapse rules, which code area
  should own this sample?"

They are complementary, not interchangeable. A shared sample-facts model lets
both strategies stay correct while eliminating duplicated stack construction.
Indexes can accelerate fact lookup, but must not alter projection semantics.

## Covered By Tests

- raw and gzipped pprof decoding;
- inline frames and non-contiguous pprof IDs;
- folded-location markers;
- sample-facts JSON export/import and malformed frame rejection;
- fact-index helpers used by projections;
- target output stability from profile facts and imported facts;
- scope/boundary output stability from profile facts and imported facts;
- slice output stability from profile facts and imported facts;
- CLI fact export plus target/scope/slice replay through `clankerprof` and
  `autoclanker pprof`;
- runtime folding, semantic callers, attributables, compatibility aliases,
  external runtime rule packs, scope owner decomposition, cached predicate
  matching, residual exclusions, slice filters, collapse, attributes,
  pseudo-slices, and compare gates.

## Remaining Confidence Boundary

The self-contained test suite proves the fact contract and projection parity on
fixture profiles. Before retiring any older profile workflow, run real-profile
golden comparisons against representative local `.pb` / `.pb.gz` files using
`scripts/clankerprof/check_real_profile_parity.py`. That helper covers target,
boundary, slice, and sample-facts artifacts, is intentionally opt-in, and
commits no profile data.

Future implementation work can add faster precomputed indexes or a
cross-language implementation of the same fact contract without changing
projection outputs.
