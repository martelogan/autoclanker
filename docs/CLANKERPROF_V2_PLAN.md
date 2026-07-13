# clankerprof v2 plan — verified findings and the two-track roadmap

Status: planning document (2026-07-08). Produced from a multi-agent deep assessment:
9 lens-specific code reviews with every substantive finding adversarially verified by
reproduction, a spec/parity distillation, git-evolution analysis, and a 5-beat industry
landscape survey. Nothing here has been applied to the code yet.

## Verdict summary

- **Novelty**: the combination clankerprof ships — serverless versioned replayable
  sample-facts contract + data-driven YAML runtime-semantic bucketing + declarative
  ownership slice/scope attribution + thresholded nonzero-exit profile-shape compare
  gates + agent-first JSON CLI — has no direct equivalent in the pprof toolchain,
  continuous-profiling platforms, CI perf-gating tools, the Ruby ecosystem, or the Rust
  ecosystem. Closest partial overlaps: google/pprof (`-focus/-ignore/-diff_base`, but
  scrape-fragile text and no gates), Grafana Pyroscope (server-coupled), CodeGuru
  Profiler (hardcoded ML categories), CodSpeed (gates on timing, not profile shape),
  Skylight (semantic buckets, instrumentation-based SaaS). The Rust read-side pprof
  analysis space is essentially vacant; `clankerprof-core` is first-of-kind.
- **Architecture**: the layered pipeline (proto → model → facts → projections →
  render/compare) is genuinely clean and the sample-facts inflection was the right
  design. The scopes/boundaries projection shows the matured pattern (typed predicates,
  frame-identity caching, spec'd complexity target). The debt is concentrated in
  `analysis.py` (four modules in one), a stringly-typed slice DSL that predates the
  typed predicate model, and duplicated categorization pipelines.
- **Correctness**: ~15 confirmed defects (each reproduced), several semantic.
- **Performance**: everything is linear in samples × depth (good); large constant
  factors are left on the table (no frame interning, per-occurrence rule evaluation in
  slices/targets, facts JSON ~90× the pprof blob, Rust recompiles regexes per call).
- **Tests**: thorough at the projection/CLI altitude but built on toy fixtures
  (1–6 samples, ≤5 frames, no recursion, single value type, ASCII only). Packed-varint
  sample encoding — what real pprof producers emit — is never exercised. `cargo test`
  runs zero tests; parity pins one flag combination per subcommand.

## Confirmed defect inventory

Severity/kind as verified. P = also present in Rust port.

### Semantic correctness

1. **[major] Recursive target frames multiply-count samples** (P) —
   `analysis.py` `analyze_target_facts` via `ProfileFactIndex.target_frames`: a target
   appearing N times in one stack adds N× the sample value to the same parent's
   `cpu_time`/`sample_count`/`total_time_ns`. Needs a spec decision (count once per
   sample per parent) before fixing in both languages.
2. **[major] `primary_value` is always `values[0]`; `sample_type`/`period`/
   `default_sample_type` are discarded** — `proto.py` skips Profile fields 1/11/12/14;
   `model.py:34` types `values[0]` as TimeNs. Standard Go CPU profiles
   (`[samples/count, cpu/nanoseconds]`) report **counts labeled as nanoseconds**.
   pprof convention: honor `default_sample_type`, else last value type.
3. **[major] `--no-enhanced` under generic runtime silently loads the ruby pack** —
   `cli.py:287-296` `_runtime_rules` returns `ruby_rules(...)` when
   `runtime == "generic" and no_enhanced`.
4. **[major] Semantic rules are unanchored substring matches with no path
   constraint** — `rules.py:23-30`: any frame whose *name* contains "StatsD"/"I18n"/…
   is claimed as gem overhead even on app paths; every `name_prefixes` entry with a
   sibling `name_contains` is dead.
5. **[major] `special_namespace_prefixes` guard only applies to `::`-qualified
   names** — `analysis.py:675-697`: bare module-function names (`Zlib.inflate`,
   `OpenSSL.fixed_length_secure_compare`) bypass the guard and leak into
   Ruby Core (Native).
6. **[major] `slice:` filters evaluate membership without descendant attribute
   rules** — `analysis.py:1327-1358` vs attribution at 1548; `!slice:X` is a no-op for
   descendant-attributed samples.
7. **[minor] Multiple `default: true` slices: attribution uses the last, tracking
   uses the first** — no uniqueness validation; unattributed-library reporting silently
   lost.
8. **[minor] `native:<value>` predicate ignores its value** — `native:false` means
   "is native" (`analysis.py:418-419`, 555-556).
9. **[minor] Fold heuristic window `stack[1:3]` counts inline-expanded frames** —
   folding outcome depends on inline depth (`analysis.py:775`).

### Compare / gate correctness

10. **[major] Compare emits bare `Infinity` — invalid JSON** — `compare.py:39-45` +
    `json.dumps(allow_nan=True)`. Rust emits the string "Infinity" — type-divergent.
    Decide the contract (suggest: `null` delta_rel + `"status": "new"`), fix both.
11. **[major] `compare_boundary_json` orders `top_improvements` ascending and can
    drop the largest** — `compare.py:256` sorts once by `abs(delta_abs)` then reverses.
12. **[minor] `compare` exits 0 with an empty report on wrong payload types** —
    only checks `tool` equality; two facts exports compare as "no regression".

### CLI / error-contract

13. **[major] `targets --format csv/text` without `--output` mixes CSV/text and the
    JSON envelope on one stdout stream** — `cli.py:966` + `_emit_json`.
14. **[major] Error taxonomy leaks tracebacks** — `main()` catches only `ValueError`:
    truncated gzip (`EOFError`/`zlib.error`, `proto.py:193-196`), missing files
    (`FileNotFoundError`), malformed YAML (`yaml.YAMLError`), facts with missing keys
    (`KeyError`, `facts.py:241,275-279`) all exit 1 with tracebacks instead of the
    exit-2 JSON envelope.
15. **[minor] Standalone `clankerprof --output` silently ignored for subcommands with
    a local `--output`** — the umbrella CLI normalizes this; the standalone parser
    doesn't (`cli.py:1365`).
16. **[minor] Filter/collapse validation skipped entirely when no slices file is
    configured** — `cli.py:212-255` early-return; `name:` (empty value) then matches
    every frame.
17. **[minor] `--target-csv-layout=compat` silently ignored with `--format json`** —
    guard sits after the JSON early-return (`cli.py:938-944`).
18. **[minor] Decoder accepts truncated fixed32/64 silently; varints >64-bit
    accepted** — `proto.py:58-66` no bounds check on skip.

### Rust-specific (beyond parity gaps)

19. **[major] BTreeMap alphabetizes what Python treats as insertion-ordered** —
    target-config category precedence (first-match-wins in file order vs alphabetical)
    and tie order in every ranked array. Replace with insertion-ordered maps
    (`indexmap`) or explicit order-preserving structures.
20. **[minor] Varint shift-overflow: panics in debug, corrupts in release** —
    `proto.rs:59-75` guard runs after the 11th-byte shift.
21. **[minor] Signed int64 fields: Rust reinterprets two's-complement, Python keeps
    raw unsigned bigint** — `line = -1` decodes differently; pick Rust's (correct
    protobuf) semantics and fix Python.
22. **[major perf] Regexes compiled per frame per call** — `Regex::new` inside hot
    loops in `targets.rs:305,396-405,424-437`.
23. **[minor] Hand-rolled arg parsing silently absorbs what Python rejects** —
    `--by-slice` consumes option-like tokens; unparseable values ignored.
24. **[major] Rust `compare` ignores the `tool` field** — boundary reports silently
    produce an empty slice comparison.

### Performance (Python)

25. **[major] No frame interning: `to_sample_facts` allocates a Frame per occurrence**
    — `model.py:85-105`; the location→frames expansion is invariant per profile.
26. **[major] Boundary loop allocates a predicate expr per occurrence and always runs
    the exclude-descendants scan** — `analysis.py:449,516,1174-1176`.
27. **[minor] Slices/targets have zero unique-frame memoization** (boundaries has it) —
    `_is_native_path`/`_slice_for_frame`/library extraction re-run per occurrence.
28. **[minor] Facts export ~90× larger than the pprof blob; replay slower than
    re-decoding** — fully denormalized frames + `indent=2` forcing the pure-Python
    JSON encoder (`facts.py:200,263-271`).

### Design debt

29. `analysis.py` (1602 lines) fuses pattern primitives, the categorization engine,
    and all three projections; `_target_category`/`_boundary_category` are ~75-line
    near-duplicates.
30. Slice filter/collapse DSL is stringly-typed (parsed per filter per sample) while
    boundaries got the typed `FramePredicate` model.
31. Projection accumulators split across `model.py:129-287` and `analysis.py:107-231`
    at the wrong altitude.
32. Rule packs accept unknown keys silently and are unversioned (unlike facts);
    1191/1358 `ruby_core_classes.csv` entries contain `::` and are unreachable by the
    component-based lookup.
33. `__init__.py` exports facts I/O but none of the analysis/proto/compare surface.
34. scopes-vs-boundaries mid-rename; legacy compat paths (`--no-enhanced`, compat CSV
    layout) pending retirement per the sample-facts plan doc.
35. Spec/doc drift: PARITY.md has only one Rust row (scopes "not claimed") while
    SPEC prose claims Rust facts/targets/slices/compare; compare "focus set" flag
    undocumented; `(all)` pseudo-slice mentioned once; `tree`/`opportunity` referenced
    in RUST.md but unspecced.

### Test gaps (highest value first)

36. **[critical] Packed-varint sample encoding never exercised** — real producers emit
    packed; the decoder handles it but nothing pins it.
37. `cargo test` runs zero tests; parity pins one flag combination per subcommand and
    compares parsed JSON, not bytes.
38. No fixtures with recursion, multi-value samples, unicode names, deep stacks,
    >6 samples; unknown-schema-version rejection untested; `once_per_sample` untested;
    no `generic.yml` ↔ `RuntimeRuleSet::generic()` equivalence test.

## Track A — finish the ideal Python implementation

Ordering matters: semantic fixes (A1) land before the Rust port expands, so Rust
mirrors corrected semantics, not bugs.

**A1 — semantics + contract wave** (spec edits first, then code+tests together)
- Spec decisions: recursion counting for targets (once per sample per parent);
  value-type selection (`default_sample_type`, else last value type; carry
  `sample_type`/`period` into the model and facts v2); compare new/removed-row
  representation (finite JSON only); signed int64 semantics (two's-complement).
- Fix items 1–12 above in Python (and 10/19/21/24 in Rust where the parity contract
  is at stake). Bump `SAMPLE_FACTS_SCHEMA_VERSION` if facts shape changes; keep v1
  import support.

**A2 — CLI/error-contract wave**
- Single typed error boundary in both CLIs: decode/validation/usage → exit 2 JSON
  envelope; wrap gzip/file/YAML/KeyError paths (items 13–18).
- Validation always on: filter shape checks regardless of slices config; default-slice
  uniqueness; strict rule-pack keys + a `runtime_rules.v1` version field.

**A3 — architecture wave**
- Split `analysis.py`: `patterns.py` (path/regex/library primitives),
  `categorize.py` (one categorization engine parameterized over cache strategy —
  deduplicates `_target_category`/`_boundary_category`), `targets.py`, `slices.py`,
  `scopes.py`. Keep `analysis.py` as a re-export shim for compatibility.
- Retype the slice DSL on the `FramePredicate` model — one predicate layer, parsed
  once, memoized per unique frame (kills item 27 as a side effect).
- Move accumulators out of `model.py`; make `__init__.py` export the real public API.
- Finish the scopes rename (boundaries = documented legacy alias everywhere).

**A4 — performance wave**
- Intern frames at decode: unique Frame table per profile, occurrence lists reference
  it (item 25); memoize `Profile.sample_facts()`.
- Facts v2 compact encoding: string/frame tables + per-sample index lists;
  `indent=None` separators-compact by default (`--pretty` opt-in) (item 28).
- Fix boundary per-occurrence churn (item 26): normalize predicate exprs once at
  config parse; skip the exclude-descendants pass when no boundary defines exclusions.

**A5 — test wave**
- Extend `PprofFixtureBuilder`: packed varints, recursion, multi-value samples,
  unicode, deep stacks, inline+folded combos, truncated/corrupt inputs.
- Property tests for reconciliation invariants (slice totals = profile total;
  category sums = parent totals; facts round-trip identity under random profiles).
- Cover every strict-validation branch in facts import; add
  `generic.yml` ↔ Rust equivalence test.

## Track B — spec-complete Rust migration

Prerequisite: A1 lands first. Definition of done: every row in
`docs/CLANKERPROF_PARITY.md` has a Rust column entry that is either "covered" with a
pinned fixture or an explicit "not claimed", and the checklist below is empty.

**B0 — fix the confirmed Rust defects** (items 19–24): insertion-ordered maps
(`indexmap`), varint guard, compare `tool` dispatch, finite-JSON compare contract,
`clap` for argument parsing (matches Python's strict rejection), precompiled regexes.

**B1 — rule system**: YAML rule-pack loading (serde_yaml is already a dependency);
packaged `generic.yml`/`ruby.yml` via `include_str!`; core-classes CSV loading with
`--core-classes` override; `--runtime ruby`; `--runtime-rules` external packs;
semantic labels, simplification maps, foldability categories, ordered
`native_name_category_rules`; delete hardcoded `RuntimeRuleSet::generic()` (replaced
by parsing the packaged YAML — guarantees no drift by construction).

**B2 — scopes/boundaries projection**: typed predicate model + expression parser
(`any`/`all`/`not`, `cost_kind:`, `runtime_label:`, `slice:`), TOML/YAML config with
preferred+legacy section aliases and mixed-section rejection, cost kinds, rollups,
owners + sub-buckets, `exclude_descendants`, `once_per_sample`, attributables,
frame-identity predicate caching, facts replay, boundary compare.

**B3 — target/slice completeness**: `--fold-runtime-internals` + folded-from
accounting, `--track-semantic-callers` + semantic-caller CSV, caller-to-leaf pairs,
callsite stdlib-skipping, `--no-enhanced`/caller-fallback prefixes,
`--cpu-attributables`, minimal `--target` mode, csv/simple-csv/text formats,
`--target-csv-layout compat`, slice config files + `./slices.yml` discovery,
`--by-slice` grammar, attribute/virtual-slice validation, GC/uncollapsible
pseudo-slices, unattributed libraries.

**B4 — parity harness v2**: shared fixture corpus with Python-generated goldens;
byte-level output comparison; a flag-matrix sweep per subcommand; native Rust unit
tests (currently zero); wire a cargo lane into `bin/dev`/CI (currently absent — Rust
is unvalidated by `./bin/dev check`).

**B5 — performance (exploit Rust, don't transliterate)**: string-interning arena for
frame names/paths, `regex::RegexSet` for rule matching, single-pass multi-projection
mode (decode once, emit targets+slices+scopes in one run — realizing the design's
promise in-process), benchmark suite vs Python on large synthetic profiles.

## Governance notes

- Every behavior change: Python + Rust together, `SAMPLE_FACTS_SCHEMA_VERSION`
  symmetric, PARITY.md row updated (add the missing Rust column while at it).
- Resolve the spec/doc drift items (35) in the same PRs that touch the behavior.
- clankerprof is not a coverage target and its docs are not doc-sync-tested today, but
  `skills/clankerprof-operator` documents the CLI surface — update it with CLI changes.
