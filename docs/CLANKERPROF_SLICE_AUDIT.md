# clankerprof slice-tool audit

This audit records what was learned while comparing `clankerprof` with an
older Rust slice analyzer. It is intentionally written in generic terms: the
implementation direction is declarative pprof analysis, not a copy of any
application-specific responsibility system.

## Goal

`clankerprof` should absorb the useful strategies from both earlier tool
families:

- keep target-function attribution, semantic runtime labels, runtime-internal
  folding, semantic caller summaries, and CSV compatibility from the legacy
  target-attribution workflow;
- keep the useful slice-analysis affordances from the older slice analyzer:
  path slices, bottom and descendant filters, collapse rules, attribution
  overrides, default-slice unattributed gem reporting, pseudo-slices, and
  regression comparison;
- preserve a language-neutral core and move runtime/application specifics into
  data files, slice configs, or caller-provided rule packs.

## Architecture comparison

| Area | clankerprof | Older slice analyzer | Audit conclusion |
| --- | --- | --- | --- |
| Runtime | Python package and CLI, shipped with `autoclanker` | Rust CLI | Rust had a faster single-purpose slice path. Python is better aligned with the surrounding optimization and evidence tooling. |
| Primary abstraction | Decoded pprof call graph model plus independent `targets`, `slices`, and `compare` commands | Slice-first profile analyzer | Separate command surfaces are more correct: target attribution semantics should not inherit slice-specific CLI behavior. |
| Protobuf handling | Minimal runtime decoder for raw or gzipped pprof profiles | Generated protobuf bindings | Generated bindings are complete, but the typed decoder avoids generated runtime artifacts and is easier to package. |
| pprof IDs | Dictionary lookup by profile IDs | Assumes contiguous 1-based IDs, with validation | `clankerprof` is more generally correct for legal pprof profiles. The older fast path is acceptable only when validated. |
| Inline frames | Expands every `Location.line` frame in traversal order | Older tests mainly assume one line per location in analysis | `clankerprof` is the safer architecture for inlined call stacks. |
| Target attribution | Parent-frame containment, leaf self-time, category regexes, `Other` catch-all | Not the primary capability | Keep this as the first compatibility surface. |
| Runtime semantics | Rule-pack-driven labels, simplification, foldability, stdlib markers, core-class CSV | Native/path heuristics only | `clankerprof` is strictly more expressive for runtime-specific semantic attribution. |
| Slice attribution | Path patterns, default slices, bottom/descendant filters, collapse, attributes, pseudo-slices | Same core concepts with optimized location flags | Semantics are now mirrored in tests; implementation can later optimize with indexes without changing output contracts. |
| Metadata | Generic declarative slice metadata in JSON output | Hardcoded responsibility/contact metadata | Generic metadata is the correct merged design. Domain-specific responsibility fields should pass through as data, not become core concepts. |
| Compare | Structured JSON output and exit code `2` for regression | Structured JSON plus terminal text report | JSON gate compatibility matters most; exact prose is not part of the stable contract. |
| Output philosophy | Machine-readable JSON/CSV first; target text report available | Terminal report first, JSON optional | Agent and CI workflows benefit from `clankerprof`'s machine-readable default. |

## Behavioral findings

The older slice analyzer contributed several important correctness rules that
are now represented in `clankerprof` tests:

- Filters parse repeated `!` and `<` prefixes in any order.
- Non-descendant filters are applied to the selected bottom attribution frame.
- Descendant filters match if any configured descendant filter matches the
  stack.
- Bottom `slice:<name>` filters honor descendant attribute rules that claim the
  sample for the requested slice.
- Collapse rules do not use descendant-attribute rescue and reject `!` or `<`
  prefixes.
- Attribute rules use `[<]<filter>,to:<slice>`, reject `!`, reject `slice:`,
  and treat duplicate filters as duplicates even when only `<` differs.
- Native frames are skipped when choosing the bottom attribution frame unless
  `--no-collapse-native` is set.
- If every eligible frame is collapsed, time is still counted and the
  `(uncollapsible)` pseudo-output reports the root eligible frame.
- `(marking)` and `(sweeping)` are attributed to a `(gc)` pseudo-output.
- Default-slice gem time can be reported as unattributed dependency cost.
- Slice comparison treats a regression as significant only when both absolute
  and relative thresholds are exceeded, and exits `2` through the CLI.

## Correctness tradeoffs

`clankerprof` is more correct for profile decoding because it does not require
profile location or function IDs to be contiguous. It also retains repeated
inline frames instead of collapsing a location to one function. Those two
properties are important for a language-neutral analyzer.

The older slice analyzer was more optimized for repeated slice queries. Its
location flags, bitsets, and precomputed slice context are good future
performance ideas, but they are not semantic requirements. `clankerprof` should
only adopt those optimizations behind the existing JSON contract and after
golden tests confirm unchanged output.

The older analyzer also carried domain metadata fields directly in the slice
model. The merged design should not preserve those names in core code. Instead,
`clankerprof` now preserves arbitrary JSON-compatible slice metadata under a
generic `metadata` field. Callers can pass labels, contacts, docs, escalation
rules, responsibility hints, or any equivalent concept through configuration.

## Implemented from this audit

- Added a generic `metadata` field to `SliceDefinition`.
- YAML slice files now preserve unknown slice keys as JSON-compatible metadata.
- A nested `metadata:` object is flattened into the output metadata payload.
- Slice JSON output includes per-slice metadata when available.
- Added coverage for non-contiguous pprof function IDs.
- Added coverage for repeated filter prefixes in arbitrary order.

## Not merged intentionally

- Terminal color, width wrapping, and exact text report wording are not part of
  the stable contract.
- Domain-specific responsibility fields are not modeled as first-class code
  fields.
- Rust-specific location bitset optimizations are not needed for correctness in
  the MVP.
- Exact historical CLI names and prose are not carried into public docs beyond
  the compatibility flag needed to reproduce the legacy two-file CSV artifact
  layout.

## Remaining validation before deleting older tools

Before retiring any production workflow, run golden comparisons on real `.pb`
or `.pb.gz` profiles:

1. Compare target CSV category totals, caller summaries, folded totals, and
   attributable columns against the older target-attribution workflow.
2. Compare slice JSON totals for representative filter/collapse/attribute
   combinations against the older slice analyzer.
3. Include at least one profile with inline frames and one profile whose IDs are
   not assumed to be array indexes.
4. Validate compare-gate behavior by asserting identical regression decisions
   for a baseline/experiment pair.

Passing those goldens would establish migration confidence beyond the
self-contained fixture tests.
