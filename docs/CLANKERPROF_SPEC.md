# clankerprof sample-facts specification

This document is the normative contract for `clankerprof` as a Python package
and as a future cross-language profile-analysis core. Implementation details
can change, but the fact shape and projection semantics below are the
compatibility target.

## Design goals

- Decode one pprof profile into one durable sample-facts model.
- Keep target-boundary attribution, scope decomposition, and slice
  attribution as projections over the same facts, not as separate decoders with
  separate accounting.
- Keep runtime and dependency knowledge declarative through rule packs.
- Keep ownership metadata declarative through slice config.
- Emit machine-readable artifacts that can be compared by tests, CI, agents, or
  another implementation in a different language.

## Decoding contract

`clankerprof` accepts raw `.pb` and gzipped `.pb.gz` pprof profiles. The decoder
must preserve:

- sparse pprof function IDs and location IDs;
- sample order and stable zero-based `sample_index`;
- all sample values, plus the profile's declared `sample_type` value types,
  `period_type`, `period`, and `default_sample_type`;
- pprof leaf-to-root location order;
- all inline frames from each location, in pprof line order;
- function name, system name, filename, start line, sample line, and folded
  location marker when present.

Signed pprof `int64` fields (sample values, line numbers, start lines,
`period`) decode as two's-complement 64-bit integers; a varint encoding of
`-1` must decode as `-1`, never as `2^64 - 1`.

The decoder is strict about malformed input: varints longer than 10 bytes,
length-delimited fields extending past the stream, and truncated fixed32/64
fields (even in skipped unknown fields) are decode errors, never silently
accepted. Varint bits past 63 drop, per protobuf 64-bit semantics.

### Primary-value selection

Projections aggregate exactly one value per sample, the *primary value*,
selected once per profile:

1. If `default_sample_type` is set and names a declared sample type, the
   primary value index is that type's position.
2. Otherwise the primary value index is the **last** declared sample type
   (pprof convention — e.g. Go CPU profiles declare
   `[samples/count, cpu/nanoseconds]` and mean nanoseconds).
3. Profiles that declare no sample types use index 0.

Samples missing a value at the selected index fall back to `values[0]`
(or `0` when the sample has no values at all).

The decoder must not assume IDs are contiguous array indexes. Optimized
implementations can build indexes, but indexes must be derived from profile IDs
without changing visible facts.

## Sample-facts JSON

`clankerprof facts --profile profile.pb.gz` exports the stable JSON shape. The
current schema version is:

```json
"clankerprof.sample_facts.v2"
```

The artifact is compact JSON with sorted keys by default; `--pretty` opts in
to indented output. Both encodings parse identically.

Top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Exact schema identifier. Readers must reject unknown versions. |
| `tool` | Producer identifier, currently `clankerprof_facts`. |
| `profile.value_types` | Declared sample value types, each `{"type", "unit"}`. |
| `profile.period_type` | Declared period type (`{"type", "unit"}`) or `null`. |
| `profile.period` | Declared sampling period, `0` when absent. |
| `profile.default_sample_type` | Declared default sample type name, `""` when absent. |
| `profile.primary_value_index` | Selected primary value index (see primary-value selection). |
| `summary.sample_count` | Number of decoded samples. |
| `summary.empty_sample_count` | Samples whose stack could not be decoded. |
| `summary.non_empty_sample_count` | Decoded samples with at least one frame. |
| `summary.total_primary_value` | Sum of primary sample values. |
| `strings` | Interned string table for frame names and filenames. |
| `frames` | Interned frame table; stacks reference rows by index. |
| `samples` | Ordered sample fact array. |

Each `frames` row is a six-element array:

```json
[location_id, function_id, name_index, filename_index, line, location_is_folded]
```

where `name_index` and `filename_index` point into `strings`, `line` is the
sample line number (otherwise `0`), and `location_is_folded` is the pprof
folded-location marker. Interning order is normative so exports are
byte-comparable across implementations: samples in order, stack frames in
order; each new frame interns its name, then its filename, then appends its
own row.

Each sample contains:

| Field | Meaning |
| --- | --- |
| `sample_index` | Stable zero-based index from the original profile order. |
| `values` | All raw sample values. |
| `location_ids` | Raw pprof location IDs for the sample. |
| `stack` | Leaf-to-root frame indexes into `frames`. |

Per-sample primary values and emptiness are derived on import:
`primary_value = values[profile.primary_value_index]` (with the fallbacks from
primary-value selection) and `is_empty = stack == []`. Importers must validate
frame and string indexes, reject non-integer stack entries, and reject
summaries that disagree with the reconstructed samples.

Numeric domains are strict and shared by both implementations. Location and
function IDs (in `location_ids` and frame rows) are unsigned 64-bit integers;
sample values, lines, and `period` are signed 64-bit integers; floats,
booleans, strings, and out-of-range numbers are validation errors, never
coerced or truncated. All JSON inputs are strict RFC 8259: the non-standard
`Infinity`/`-Infinity`/`NaN` tokens are validation errors in both languages.

Aggregates over sample values are exact, with a documented bound enforced at
import and at profile decode: the sum of positive primary values must fit
`u64` and the sum of negative primary values must fit `i64` (error:
"Aggregate sample values exceed the supported integer range."). Every subset
sum a projection can produce then lies in `[i64::MIN, u64::MAX]`, so derived
totals (which may exceed `i64::MAX`, e.g. two `i64::MAX` samples totalling
`18446744073709551614`) serialize as exact JSON integers in both languages —
never a panic, wrap, or float approximation.

Round-tripping this JSON through `loads_sample_facts` must preserve target and
slice projection outputs, including exports whose IDs exceed `i64::MAX`: each
implementation must be able to replay its own export of any decodable
profile.

### Legacy v1 imports

Readers must continue to accept `clankerprof.sample_facts.v1` payloads
(denormalized per-sample `primary_value`/`is_empty`/frame objects). v1
predates value-type metadata: v1 samples keep `values[0]` as their primary
value, preserving the meaning the artifact had when it was produced. Writers
always emit v2.

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

Rule packs are strict and versioned: unknown top-level keys and unknown
match-rule entry keys are validation errors (typos never silently disable a
rule), and the optional `schema_version` field must be
`clankerprof.runtime_rules.v1` when present (absent means v1).

### Rule matching semantics

Within one rule, `name_contains` entries match as substrings anywhere in the
frame name, `name_prefixes` entries anchor at the start of the name, and
`name_patterns` entries are regular expressions; a rule matches when any
entry matches and the frame path is not in `except_paths`. Rules evaluate in
pack order; the first matching rule wins.

Semantic rules may only claim frames on **runtime-owned paths**: native
pseudo-paths, runtime stdlib paths, runtime-internal paths, and
dependency/library paths per the active pack. Frames on plain application
paths are never claimed by semantic rules, no matter how their names read —
an application class whose name happens to contain a dependency's name stays
application code. A pack that declares no path-ownership configuration
(no native/stdlib/library path keys) cannot distinguish application paths,
so its semantic rules apply to every frame.

The `special_namespace_prefixes` guard blocks qualified names
(`OpenSSL::Cipher#encrypt`) from resolving through the core-class table. Bare
module-function names on guarded namespaces (`Zlib.inflate`,
`OpenSSL.fixed_length_secure_compare`) resolve through the pack's ordered
native-name rules **before** the core-class table when they appear on native
paths, so the core table's default category never swallows names the pack
labels semantically; off native paths, the core category maps
(`core_semantic_categories` and friends) keep their legacy claims.

The `--no-enhanced` flag disables enhanced runtime categorization for the
**active** runtime's rules; it never swaps rule packs. Under the generic
runtime, `--no-enhanced` keeps the generic pack.

## Target projection contract

`targets` answers: “inside this parent boundary, what leaf work consumed CPU,
and which higher-level caller made it matter?”

For each sample:

- find every configured parent frame contained in the stack;
- attribute the full primary sample value **once per matching parent**: a
  parent appearing multiple times in one stack (direct or indirect recursion)
  still receives the sample value exactly once, so a parent's total can never
  exceed the profile total;
- start from the leaf frame as the self-time owner;
- optionally label the leaf using runtime rules;
- optionally fold runtime-internal leaves into the first meaningful caller;
  the fold heuristic's caller window spans the next two **distinct locations**
  below the leaf, so the outcome is independent of inline expansion of the
  leaf's own location;
- match configured target categories against the chosen frame path;
- place unmatched time in `Other`;
- track leaf functions, categorized files, folded-from totals, semantic callers,
  and caller-to-leaf pairs.

Target projection must preserve complete target accounting. Category totals for
one parent must sum to the total CPU time under that parent.

## Scope decomposition contract

`scopes` answers: “inside this parent denominator, which rollups and cost
kinds consumed CPU, and which owner frame below the scope drove each cost
kind?”

Scope decomposition is a richer parent-scope projection, not a slice
replacement. The JSON payload keeps historical boundary field names for
compatibility, but the preferred authoring terminology is:

| Concept | Meaning |
| --- | --- |
| Scope | The parent frame that defines the denominator. |
| Cost kind | Atomic taxonomy used for sampled or folded work. |
| Rollup | Scope-specific display grouping of cost-kind rows. |
| Owner | Observed owner taxonomy below the scope; each owner preserves cost-kind sub-buckets. |
| Slice | Optional ownership source that `owner` predicates can reference via `slice:<name>`. |

Scope, cost-kind, owner, and exclusion selectors accept a plain selector,
an OR list of selectors, or an expression table with `any`, `all`, and `not`.
Supported selector keys are intentionally generic: function name contains,
function name equality, path/glob, regex, native frame, dependency selectors,
optional slice label, configured cost-kind label, and runtime-rule label.
The `native:` selector takes `true` (or a bare `native`) to match native
frames and `false` to match non-native frames; any other value is a
validation error. A
`cost_kind:<label>` selector matches the configured `[cost_kind]` label for a
frame and is valid for owners, scopes, and exclusions. A
`runtime_label:<label>` selector matches labels produced by the selected
runtime rule pack. Compatibility aliases remain accepted: `boundaries`,
`[category]`, `[domain]`, `[[boundary]]`, `[boundary.bucket]`, `category:`, and
`runtime_category:`. Cost-kind definitions cannot reference `cost_kind:` or
`category:` recursively; use rollups for display groupings.

For each sample:

- compute the cost kind once from the leaf or folded caller;
- walk the stack leaf-to-root once;
- maintain the current owner below the current frame;
- attribute the full primary sample value to every matching scope occurrence;
- skip a scope occurrence when a configured `exclude_descendants` predicate
  has already matched lower in the stack;
- count repeated/recursive scope frames by occurrence by default, or once
  per sample when configured;
- collect cost-kind totals, rollup totals, owner totals, owner cost-kind
  totals, top owner files, and representative caller-to-leaf pairs.

Predicate matching should be cached by unique frame identity so the hot path is
approximately `O(samples x frames + unique_frames x configured_predicates)`.
Implementations must avoid rescanning each sample prefix for every boundary,
cost-kind, and owner pairing.

Scope totals for one scope must remain internally additive: rollup cost-kind
rows sum to the scope total, and owner cost-kind rows sum to their owner totals.
Calibrated attributables are caller-supplied same-scope metrics; the analyzer
only scales them proportionally by scope CPU share.

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

Bottom `slice:<name>` filters evaluate the sample's **effective** attribution
in both polarities: a sample rescued into a slice by a descendant attribute
rule matches `slice:<name>` and is excluded by `!slice:<name>`. Collapse
`slice:` rules intentionally do not use the descendant-attribute rescue.

At most one slice may set `default: true`; declaring several is a validation
error (previously attribution silently used the last while tracking used the
first).

Slice projection must not redefine target-boundary semantics.

## Compare contract

`compare` consumes two slice JSON payloads or two boundary JSON payloads,
dispatching on their shared `tool` field. Both inputs must carry the same
`tool`, and it must be `clankerprof_slices` or `clankerprof_boundaries`;
anything else (including two facts exports) is a validation error, never a
silent "no regression". Each report must contain its projection's row array
(`slices` or `boundaries`); a payload missing it is a validation error, not
an empty comparison. Numeric report fields (`pct`, `pct_of_profile`,
`total_time_ns`) must be JSON numbers: malformed values are a validation
error in both implementations, never coerced to zero. Compare thresholds
must be finite numbers; a NaN or infinite threshold is an option-validation
error (a non-finite threshold would silently disable gating). For slice
payloads, it reports:

- total before/after primary time from summaries;
- per-slice before percent, after percent, absolute delta, relative delta, and
  status;
- per-function deltas within each slice (frames sharing a function name are
  summed);
- top regressions and improvements;
- `has_regression`;
- exit code `2` for regression through the CLI.

For boundary payloads, it reports the same threshold semantics over stable
boundary, bucket, category, and domain rows. Boundary compare is intentionally
JSON-first; exact terminal prose is not part of the compatibility contract.

Compare artifacts are strict JSON. A row that is new (`before_pct == 0`,
`after_pct > 0`) has no finite relative delta; its `delta_rel` serializes as
`null`, never as a bare `Infinity` token or a string. New rows still
participate in regression gating: threshold math treats their relative delta
as unbounded, so a new row whose absolute delta exceeds the absolute
threshold gates. `top_regressions` orders rows by descending absolute delta;
`top_improvements` orders rows by ascending (most negative first) delta.

A slice is a regression only when it exceeds both the configured absolute and
relative thresholds and is within the focus set when one is provided. Focus
sets come from `--focus-slices` for slice reports and `--focus-boundaries`
(alias `--focus-scopes`) for scope reports; both take comma-delimited names
and gate only the named rows while still reporting every row.

## CLI stream and error contract

Successful JSON commands print exactly one JSON document to stdout (the
payload, or an `{"ok": true, "output": ...}` receipt when `--output` wrote the
artifact). Non-JSON formats (`csv`, `simple-csv`, `text`) without `--output`
print exactly the raw rendered payload to stdout — never mixed with a JSON
envelope; with `--output` they write the artifact and print the JSON receipt.
A global `--output` before the subcommand is equivalent to the subcommand's
own `--output` through both the standalone and umbrella CLIs.

Every subcommand, including `compare`, accepts `--output`. Receipts name the
writing tool in a `tool` key; the `facts` receipt additionally carries
`schema_version` and `summary`, and the `compare` receipt additionally
carries `has_regression`. `compare`'s regression exit code is unchanged by
`--output`: the artifact is written and the receipt printed before the
nonzero exit. `facts` without `--output` prints to stdout exactly the bytes
it would write to the artifact — compact by default, indented with
`--pretty`.

Every contracted failure — decode errors (including truncated or corrupt
gzip), missing or unreadable input files, malformed YAML/JSON/TOML configs and
rule packs, invalid facts payloads (wrong schema version, missing keys, index
or summary mismatches), and option validation — exits `2` and prints a single
`{"ok": false, "error": ...}` JSON envelope to stderr, never a traceback.
Usage errors (unknown flags or subcommands, missing required options, invalid
option values) follow the same envelope through the standalone `clankerprof`
and `clankerprof-rs` CLIs, while `--help` and `--version` stay human-readable
with exit `0`; the umbrella `autoclanker pprof` surface guarantees the
envelope for contracted runtime failures and exit code `2` for usage errors,
but its usage errors may print the host parser's prose message. Filter and
collapse shape validation always runs, with or without a slices config.

## Compatibility and validation

Public tests must cover:

- raw and gzipped pprof decoding;
- sparse pprof IDs;
- inline frames;
- folded locations;
- sample-facts JSON round-trip;
- fact-index helpers used by projections;
- target output stability from profile facts and imported facts;
- scope/boundary output stability from profile facts and imported facts,
  including owner cost-kind rows and cached selector matching;
- slice output stability from profile facts and imported facts;
- runtime folding, semantic callers, attributables, and compatibility aliases;
- external runtime rule-pack loading without package changes;
- slice filters, collapse, attributes, pseudo-slices, and compare gates.

Real-profile parity must be optional and local-only. The helper under
`scripts/clankerprof` compares caller-provided local facts, target, scope,
and slice reference artifacts against current `clankerprof` output and
intentionally commits no profile data. Its `--check-rust-core` mode
additionally compares Rust facts and generic target projection output against
Python for the same caller-provided local inputs.

## Rust core parity

`crates/clankerprof-core` is the Rust compatibility implementation for this
spec, with Python `clankerprof` as the reference implementation. Its parity
surfaces cover every public projection: `facts` (export and v1/v2 replay),
`targets` (all formats and runtime flags), `slices` (filters, collapse,
attributes, pseudo-slices), `scopes`/`boundaries` (full decomposition with
TOML/YAML configs), `compare` (slice and boundary gates), and the single-pass
`report` mode. Runtime rule packs load from the same packaged YAML files via
`include_str!`, so the vocabularies cannot drift. The facts command emits the
same `clankerprof.sample_facts.v2` payload byte-for-byte as Python for raw
profiles, gzipped profiles, inline frames, folded locations, sparse pprof
IDs, multi-value samples, and packed sample encoding.

Projection work must build on the same Rust fact model rather than introducing
separate ad hoc accounting. Any future Rust projection or comparison command
must prove parity against Python fixtures before being used as an integration
boundary by downstream tools. Runtime-specific
rule-pack parity remains the expansion point after the generic projection
surface; downstream integrations should keep calling this out explicitly when
they depend on runtime-specific folding or semantic categorization.
