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

`clankerprof` accepts raw `.pb` and gzipped `.pb.gz` pprof profiles. Gzip
streams may contain multiple RFC 1952 members (producers that append emit
them); every member must be decoded — silently truncating to the first member
is a defect. The decoder must preserve:

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

The `summary` block is a redundancy check: exporters always write it, and
importers cross-validate a present summary against the decoded samples
(mismatches are validation errors). An absent (or explicit `null`) summary
skips the cross-check identically in both implementations; it never affects
the imported facts. A present summary of any other type is a validation
error (`Sample facts summary must be an object.`) — wrong-typed is
malformed, never "absent".

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

All four sample fields are required in v2 payloads: a sample object missing
any of them is a validation error (`Sample facts payload missing required
key: '<field>'.`), never an empty default. Legacy v1 payloads keep their
historical leniency (missing per-sample arrays read as empty) in both
implementations.

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
`Infinity`/`-Infinity`/`NaN` tokens are validation errors in both languages,
and duplicate object member names are validation errors on every JSON input
surface — never silent last-wins, which would make the same multiset of
members change meaning with ordering (shared message core
`duplicate entry with key "<key>"`; the location suffix is engine-specific,
exactly like the YAML duplicate-key rule).
JSON integer literals outside `[i64::MIN, u64::MAX]` are representation-
divergent internally (serde_json falls back to `f64`; Python keeps an
unbounded `int`) but behaviorally identical by contract: float-domain fields
coerce both representations to the same `f64`, and integer-domain fields
reject both with the same messages — pinned by the parity suite.

Aggregates over sample values are exact, with a documented bound enforced at
import and at profile decode: the sum of positive primary values must fit
`u64` and the sum of negative primary values must fit `i64` (error:
"Aggregate sample values exceed the supported integer range."). Every subset
sum a projection can produce then lies in `[i64::MIN, u64::MAX]`, so derived
totals (which may exceed `i64::MAX`, e.g. two `i64::MAX` samples totalling
`18446744073709551614`) serialize as exact JSON integers in both languages —
never a panic, wrap, or float approximation. The import-time bound covers
subset sums only; occurrence-mode scope attribution counts a sample once per
matching frame occurrence and can therefore exceed it. Both implementations
re-enforce the same bound during scope accumulation and fail with the
identical validation error rather than emitting out-of-range integers.

Percentage fields divide with both integer operands rounded to `f64` first
(Rust `as f64` casts; Python mirrors them operand-for-operand). Below `2^53`
this is identical to exact integer division; above it, the f64-operand form
is the shared contract, so percentage bytes and `--by-slice` threshold
selection stay identical across implementations. Floats read from JSON
re-emit exactly (serde_json's `float_roundtrip`; CPython parses correctly
rounded natively).
Implementations must read rendered totals back through both signed and
unsigned integer views: a value in `(i64::MAX, u64::MAX]` is valid data, not
zero.

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

Library regex patterns name the library with their first capture group when
that group participates in the match; when the pattern declares no groups, or
its first group does not participate (e.g. an unmatched optional group), the
whole match names the library instead. The relative path always runs from the
naming component's start to the end of the normalized path. Both
implementations share these semantics exactly.

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

Rendering rules shared by both implementations:

- A zero parent total (valid signed samples can cancel to exactly zero)
  renders every dependent percentage as `0` — never a division error, `inf`,
  or `NaN` — and the text report's TOTAL row reports `0.00%`, not `100.00%`.
- The simplified (`simple-csv` and compat simplified) noise gate is
  magnitude-aware: categories with `|share| < 0.1%` other than `Other` may be
  omitted; rows with negative shares of any magnitude must be rendered —
  sample values are signed, and dropping them breaks additivity.
- Caller-to-leaf attribution picks the first non-pseudo, non-runtime-stdlib
  frame above the leaf, scanning to the root — never a fixed-depth window.
  The leaf's immediate caller is the last-resort fallback only when no
  eligible frame exists in the whole stack.
- Core-class CSV files (packaged or `--core-classes` overrides) parse their
  first field with CSV semantics (Python `csv.reader` default dialect):
  quoted fields unwrap, doubled quotes unescape, quoted commas stay in the
  field, and quoted fields may span newlines (records are scanned across the
  whole payload, never split at line boundaries first) — identical class
  sets in both implementations.

Parents emit in first-seen encounter order (the order the analysis first
attributed a sample to them) in every row-oriented format — `csv`,
`simple-csv`, `text`, and the compat CSV pair — matching the Python
reference; JSON objects remain sorted-key. Ranked arrays inside a parent keep
breaking ties by first-seen order. Alphabetical reordering of parents is a
compatibility bug, not a presentation choice.

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
An expression table that declares no predicates and no `any`/`all`/`not`
children is a validation error in both implementations, never a
never-matching predicate. A scope's `count` must be the string `occurrence`
or `once_per_sample`; any other value (including non-strings) is a validation
error, never a silent occurrence default. An owner's `fallback` flag follows
Python truthiness over the parsed value in both implementations.
Configured cost-kind and owner definitions are evaluated in declaration
order and the first matching definition wins, in YAML and TOML alike; a
sorted-map reordering of these tables is a correctness bug. A scope's
`label` and `name` values must be strings (`scope.label must be a string.`),
and `function` must be a string or an array of strings; other shapes are
validation errors in both implementations, because Python `str()` and Rust
`Display` spell non-string scalars differently. YAML inputs (configs and
rule packs) reject duplicate mapping keys in both implementations — never
silent last-wins. The envelope message contains
`duplicate entry with key "<key>"`; surrounding context or line/column
detail is engine-specific and not part of the byte contract. YAML mapping
keys must be strings (`YAML mapping keys must be strings.`): bool, number,
null, and sequence keys are validation errors on every YAML surface, because
Python `str()` and serde's `Display` have no shared spelling for them. The
YAML 1.1 timestamp resolver is not applied — date-like scalars such as
`2026-01-01` stay plain strings, matching serde_yaml's YAML 1.2 core schema.

Plain-scalar resolution follows serde_yaml's dialect in both implementations
(the shared table is pinned empirically by
`crates/clankerprof-core/tests/yaml_scalar_semantics.rs` and its Python
mirror): booleans are the `true`/`false` spellings only — `yes`/`no`/`on`/
`off` are plain strings; integers are signed decimal without leading zeros
plus signed `0x`/`0o`/`0b` forms, and YAML 1.1-only forms (underscores,
sexagesimal `1:2:3`, bare-`0` octal like `017`) stay strings; plain integers
outside `[-2^63, 2^64-1]` fail the parse itself with the shared message core
`` invalid type: integer `<value>` as u128|i128, expected any YAML value ``
(location detail engine-specific); floats require a dot or exponent, accept
unsigned exponents (`1e2`), never contain underscores, keep the unsigned
`.inf`/`.nan` spellings (signed NaN forms are strings), and literals that
would overflow to infinity (`1e309`) stay plain strings rather than becoming
infinite. String-typed integer fields (for example a quoted `top: "17"`)
share one grammar in both implementations: surrounding whitespace is
trimmed, then ASCII signed decimal within the i64 domain — underscores,
unicode digits, and prefixed forms are validation errors.

Attributable metric values — the `--cpu-attributables` JSON table and a
scope's `attributables` block — must be JSON numbers in both
implementations: booleans and numeric strings are rejected
(`Attributable column <name> values must be numbers.` /
`Boundary attributable <name> must be a number.`), never coerced. Attributable
estimates must also stay JSON-representable end to end: a finite metric scaled
by a >100% bucket share can overflow, and a non-finite estimate — input or
scaled — fails closed in both implementations with
`Attributable estimate for '<name>' is not finite.`, never a silent `null`.
This covers every estimate surface — scope buckets, categories, caller pairs,
and the target CSV layouts — and the load layer: a `--cpu-attributables`
value that is non-finite after parsing (e.g. an overflowing `1e309` literal)
is rejected before any scaling in both languages (Python emits the shared
estimate message; serde_json fails the parse itself — exit codes match,
message detail is engine-specific).
Selector and predicate arrays require string entries: a non-string entry in
a scope `selector`/`matcher`/`match` list is
`<section> selector values must be strings.`, and a non-string entry in any
string-or-array field (cost-kind patterns, rollup categories) is
`<field> must be a string or array of strings.` in both implementations.
The same string strictness covers the other configuration surfaces: slice
definition names and `paths` entries (`Slice name must be a string.` /
`Slice paths values must be strings.`), target config patterns
(`Target config pattern for <category> must be a string.`), and runtime rule
pack fields — scalar fields are
`Runtime rule field <key> must be a string.` and list/map entries are
`Runtime rule field <key> entries must be strings.` — because Python `str()`
and Rust would otherwise spell (or drop) non-string values differently.
Supported selector keys are intentionally generic: function name contains,
function name equality, path/glob, regex, native frame, dependency selectors,
optional slice label, configured cost-kind label, and runtime-rule label.
The `native:` selector takes `true` (or a bare `native`) to match native
frames and `false` to match non-native frames; any other value is a
validation error. An unsupported predicate key is a validation error wherever
the predicate is evaluated — including configured cost-kind and category
definitions — never a silent fall-through to `Other`. Path/glob selectors
follow CPython `fnmatch` semantics everywhere globs appear (target category
patterns, slice `paths`, scope predicates): `*`, `?`, and bracket classes —
`[seq]`, negation `[!seq]`, character ranges, a literal `]` in first
position — all match; an unterminated `[` is a literal; an inverted range
such as `[z-a]` never matches. Both implementations must attribute
identically for every supported glob form. Explicit `regex:` patterns (and
rule-pack `name_patterns` and `library_path_patterns` regexes) follow
Python's regular-expression dialect, including lookaround and
backreferences; the Rust implementation compiles them with an engine
(fancy-regex) that accepts that dialect, and configs must stay within the
common subset both engines accept. An explicit pattern that fails to compile
is a validation error — exit `2` with an envelope whose message starts with
`Invalid regex pattern '<pattern>':` (library patterns:
`Invalid library regex pattern '<pattern>':`); engine-specific detail may
follow the prefix and is not part of the byte contract. Pattern errors
surface lazily, when a frame first evaluates the pattern, identically in
both implementations — never as a silent no-match. Auto-mode patterns
(no `mode:` prefix) that fail to compile fall back to path matching when
they look like paths, exactly like the reference. Rule packs additionally
validate `name_patterns` at load in both implementations with the identical
message `Invalid runtime rule name pattern '<pattern>'.`. A
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
Zero-aggregate rows may be omitted from rendered rollups (they cannot affect
the sums), but rows with negative aggregates must be rendered — sample values
are signed, and dropping negative rows breaks additivity.
Calibrated attributables are caller-supplied same-scope metrics; the analyzer
only scales them proportionally by scope CPU share. The share is signed —
a `-10` row inside a `-10` scope is 100% of it — so estimates are emitted
for any nonzero scope total and suppressed only when the total is exactly
zero (undefined share), mirroring the zero-arm of the pct fields; the
finite-estimate guard still rejects any scaled result that overflows.

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
- apply descendant filters as an OR across the stack; a negated descendant
  filter (`<!pred`) matches only samples whose stack contains NO frame
  matching `pred` — negation binds to descendant existence, never per-frame
  (a stack containing the forbidden frame must not pass just because some
  other frame fails to match);
- apply attribute rules before path-based slice matching;
- use the configured default slice or `(all)` for unmatched samples;
- report GC pseudo-slice time for `(marking)` and `(sweeping)`;
- pseudo-outputs (`gc`, `uncollapsible`) are omitted only when their
  aggregate is exactly zero; negative aggregates are signed data and must be
  reported (mirrors the scope rollup rule);
- report unattributed dependency/library summaries for default-slice time when
  requested.

Bottom `slice:<name>` filters evaluate the sample's **effective** attribution
in both polarities: a sample rescued into a slice by a descendant attribute
rule matches `slice:<name>` and is excluded by `!slice:<name>`. Collapse
`slice:` rules intentionally do not use the descendant-attribute rescue.

At most one slice may set `default: true`; declaring several is a validation
error (previously attribution silently used the last while tracking used the
first). The `default` value must be a YAML boolean — absent and `null` read
as false, and any other type is a validation error in both languages
(`Slice default must be a boolean.`); truthiness coercion is forbidden
because it silently diverged between implementations.

Slice projection must not redefine target-boundary semantics.

## Compare contract

`compare` consumes two slice JSON payloads or two boundary JSON payloads,
dispatching on their shared `tool` field. Both inputs must carry the same
`tool`, and it must be `clankerprof_slices` or `clankerprof_boundaries`;
anything else (including two facts exports) is a validation error, never a
silent "no regression". Each report must contain its projection's row array
(`slices` or `boundaries`); a payload missing it is a validation error, not
an empty comparison. Rows are strict: every row must be an object carrying a
string `name` (frame rows a string `function`) and its numeric field (`pct`,
`pct_of_profile`) — a present row missing its numeric field is malformed
input, never a zero default. Duplicate top-level row names (a slice name, or
a boundary/bucket/category/domain row within one boundary) are a validation
error in both implementations — projections never emit duplicates, and
last-wins would make the gate order-dependent. Row-level absence is different
and legal: a name
present in only one report compares against `0.0` (the documented new/removed
row semantics). Nested row arrays (`frames`, `buckets`, `categories`,
`domains`) may be absent, but a present key must be an array of objects — a
present `null` is a wrong shape and a validation error, not an absent array
(conflating them would let a nulled-out array turn a real regression into an
apparent removal). Numeric report fields must be JSON numbers: malformed
values are a validation error in both implementations, never coerced to
zero. Derived compare values (frame-percentage sums and absolute deltas)
must also stay finite: overflow of finite inputs fails closed with
`Compare values for '<name>' are not finite.` in both languages, never a
silent `null`. Each report must carry
a `summary` object whose `total_time_ns` is an integer, accepted across the
full aggregate range `[i64::MIN, u64::MAX]`, exactly as projections emit
them; absent or out-of-range values are rejected like non-integers. Compare
thresholds must be finite, non-negative numbers spelled in the strict shared
float grammar
(Rust `f64::from_str` mirrored by Python): underscores, surrounding
whitespace, and non-ASCII digits are validation errors, and a NaN, infinite,
overflowing, or negative threshold is an option-validation error (a
non-finite threshold would silently disable gating; a negative one would
gate identical reports as regressions). For slice
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

Compare artifacts are strict JSON. Relative deltas are computed against the
magnitude of the baseline (`delta_abs / |before_pct| * 100`): sample values
are signed, so rows can carry negative percentages, and a `-10% -> -5%` row
is a +50% relative increase that gates like any other (positive baselines are
bit-identical to plain `delta/before`). A zero baseline has no finite
relative delta; its `delta_rel` serializes as `null`, never as a bare
`Infinity` token or a string (the same `null` encoding applies in the rare
case where a finite delta over a tiny baseline overflows the relative
computation itself), and threshold math treats it as unbounded in
the direction of the absolute delta — so a new row whose absolute delta
exceeds the absolute threshold gates, and a new negative row is an unbounded
improvement, symmetrically. `top_regressions` orders rows by descending absolute delta;
`top_improvements` orders rows by ascending (most negative first) delta.

A slice is a regression only when it exceeds both the configured absolute and
relative thresholds and is within the focus set when one is provided. Focus
sets come from `--focus-slices` for slice reports and `--focus-boundaries`
(alias `--focus-scopes`) for scope reports; both take comma-delimited names
and gate only the named rows while still reporting every row.

## CLI stream and error contract

Successful JSON commands print exactly one JSON document to stdout (the
payload, or an `{"ok": true, "output": ...}` receipt when `--output` wrote the
artifact). Non-facts JSON artifacts, receipts, and error envelopes use
Python's `json.dumps` lexical form in both implementations: sorted keys,
two-space indentation for artifacts and receipts (envelopes are single-line
with `", "`/`": "` separators), non-ASCII characters escaped as `\uXXXX`
(surrogate pairs for astral code points), and CPython `repr` float spelling
(`1e-06`, `1e+21`, integral floats keep `.0`). Facts artifacts are the
documented exception: they serialize non-ASCII raw (UTF-8), compact by
default. These are byte contracts verified by the parity suite. Non-JSON formats (`csv`, `simple-csv`, `text`) without `--output`
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
`--by-slice` values fail closed in both implementations: bare values must be
signed 64-bit integers (negative limits drop slices from the tail,
Python-slice style) and `%`-suffixed thresholds must be finite numbers —
unparsable or non-finite values are validation errors, never ignored.
Every integer-valued CLI flag (`--top`, `--unattributed-libraries`,
`--by-slice` bare values) shares one strict grammar in both implementations:
an optional sign followed by ASCII digits, within the signed 64-bit range.
Underscores, surrounding whitespace, non-ASCII digits, and out-of-range
magnitudes are validation errors (`--top values must be integers.`), never
lenient `int()` coercion. Signed limits keep Python `list[:n]` semantics
everywhere they rank rows, including `scopes --top`.
`slices --config` (TOML or YAML) is part of the shared CLI surface: config
and command line merge with identical duplicate-scalar rejection, value
coercion, and error ordering in both languages, and `compare` accepts
`--focus-scopes` as an alias of `--focus-boundaries` in both.
`--focus-slices` and `--focus-boundaries` each take a single comma-delimited
value; repeating a focus flag keeps the last occurrence in both
implementations rather than accumulating.

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
