# clankerprof sample-facts unification plan

`clankerprof` should have one decoded sample-facts core and several projections.
This plan defines that seam so target attribution, slice attribution, semantic
caller exports, and compare gates do not each invent their own stack traversal
or accounting rules.

## Goal

Make the profile fact model explicit enough that:

- target-boundary attribution can keep its parent-contained, leaf-self-time
  semantics;
- slice/filter/collapse attribution can keep its bottom-frame and ownership
  semantics;
- both projections consume the same per-sample stack facts;
- future optimizations or Rust ports can operate on the same declarative fact
  shape without redefining output behavior.

The first implementation milestone is deliberately conservative: introduce the
sample-facts API and move existing projections onto it without changing public
CLI outputs.

## Fact Core Contract

The core sample-facts layer owns decoded pprof facts only:

- raw `Sample` identity and all sample values;
- primary sample value used as CPU time;
- leaf-to-root frame order;
- expanded inline frames from each pprof location;
- profile, location, and function IDs;
- function name, system name, file path, and line number;
- location folded marker;
- total profile CPU time;
- empty-stack accounting.

The core must not own:

- runtime-specific labels such as Ruby core/native categories;
- target category matching;
- slice ownership;
- collapse/filter semantics;
- metadata such as owners, docs, contacts, or team concepts;
- terminal formatting or comparison thresholds.

Those are projections layered over the facts.

## Projection Contract

Target attribution owns parent-boundary accounting:

- find every target frame contained in a sample stack;
- attribute sample self-time by the leaf frame or folded caller;
- preserve `Other` catch-all accounting;
- preserve runtime-rule folding, semantic callers, folded-from summaries,
  caller-to-leaf pairs, legacy no-enhanced fallback, proportional attributables,
  and legacy CSV layout.

Slice attribution owns bottom-frame accounting:

- choose the first non-native, non-collapsed eligible frame unless
  `no_collapse_native` is set;
- apply bottom filters conjunctively and descendant filters as OR;
- support `name:`, `path:`, `library:`, `dependency:`, `gem:`, `package:`,
  `vendor:`, and `slice:` filter keys where valid;
- support collapse rules and explicit attribute rules;
- preserve default-slice unattributed dependency summaries, GC pseudo-slices,
  uncollapsible pseudo-output, by-slice limits, and compare JSON compatibility.

Runtime rules own language-specific interpretation:

- semantic labels;
- simplification maps;
- foldable runtime categories;
- stdlib/native path detection;
- library/dependency path extraction;
- compatibility aliases such as `gem:`.

## Why This Seam

The two historical analysis styles answer different questions:

- target attribution asks: "inside this parent boundary, what leaf work consumed
  CPU, and which higher-level caller made that leaf work happen?"
- slice attribution asks: "after filtering and collapse rules, which code area
  should own this sample?"

They are complementary, not interchangeable. A shared sample-facts model lets
both strategies stay correct while eliminating duplicated stack construction and
making future indexes safe: indexes can accelerate facts, but must not alter
projection semantics.

## Compatibility Checklist

Before this refactor is considered complete, tests must prove:

- existing target JSON/CSV/text output semantics are unchanged;
- existing slice JSON output semantics are unchanged;
- inline frames and non-contiguous pprof IDs remain supported;
- Ruby runtime folding and semantic caller behavior remain supported;
- legacy regex configs and `gem:` selectors remain supported;
- simplified path, `library:`, and `dependency:` selectors remain supported;
- default slice `unattributed_libraries` and legacy `unattributed_gems` remain
  present;
- compare gate behavior remains unchanged;
- the new sample-facts API exposes stable, inspectable per-sample facts for
  library users.

## Future Work

After the conservative API refactor, the next improvements can be layered on
without changing outputs:

- precomputed frame and library indexes for faster filter/collapse projections;
- serialized JSON sample-facts export for cross-language golden tests;
- direct golden comparisons against real profiles from prior workflows;
- a Rust implementation of the same fact contract for high-throughput
  integrations.
