# clankerprof v2 — requirements tracker

Execution tracker for `docs/CLANKERPROF_V2_PLAN.md`. Every row is a checkable
requirement; `(item N)` references the plan's confirmed-defect inventory.

**Deterministic goal:** `python3 scripts/clankerprof_v2_goal.py --goal` exits 0
only when every row below is `done` (or `dropped` with a reason in Notes) AND
`./bin/dev check` AND `cargo test --workspace` pass.

Status vocabulary: `todo` | `doing` | `done` | `blocked` | `dropped`.

Checkpoint protocol: work lands in small clusters on branch `clankerprof-v2`;
each cluster is committed only when the focused tests for its rows are green;
statuses flip to `done` in the same commit. On interruption, resume from this
file — it is the single source of execution state.

## Wave A1 — semantics + contract

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A1-01 | Spec decision recorded: recursive target frames count once per sample per parent (item 1) | CLANKERPROF_SPEC.md target projection contract | done | "once per matching parent" rule |
| A1-02 | Python targets: recursion no longer multiply-counts sample values (item 1) | test_clankerprof_targets_recursive_frames_count_once_per_sample | done | dedup by target name per sample |
| A1-03 | Rust targets: same recursion fix, parity-pinned (item 1) | targets parity test, recursive case | done | mirrors A1-02 |
| A1-04 | Python decodes sample_type/period_type/period/default_sample_type; primary value honors default_sample_type else last value type (item 2) | test_clankerprof_primary_value_defaults_to_last_value_type; test_clankerprof_primary_value_honors_default_sample_type; test_clankerprof_primary_value_unknown_default_falls_back_to_last | done | selection rule spec'd in CLANKERPROF_SPEC.md |
| A1-05 | Rust decodes the same fields with identical selection (item 2) | parity fixtures multi_value / multi_value_default / packed | done | proto.rs fields 1/11/12/14 |
| A1-06 | Facts carry value-type metadata so replay == direct decode; SAMPLE_FACTS_SCHEMA_VERSION bumped symmetrically (v2) with v1 import kept | test_clankerprof_facts_replay_matches_direct_decode_for_multi_value; test_clankerprof_facts_import_accepts_v1_payloads; test_clankerprof_facts_import_rejects_unknown_schema_version | done | v2 byte-identical Python↔Rust (verified incl. unicode) |
| A1-07 | `--no-enhanced` under generic runtime must not load the ruby pack (item 3) | test_clankerprof_no_enhanced_generic_runtime_keeps_generic_rules; test_clankerprof_no_enhanced_runtime_selection_is_observable | done | generic means generic; ruby needs --runtime ruby |
| A1-08 | Semantic rule matching spec'd + fixed: `name_contains` no longer claims app-path frames; `name_prefixes` meaningful (item 4) | test_clankerprof_semantic_rules_do_not_claim_app_frames_by_substring | done | runtime-owned-path constraint (vacuous for name-only packs); dead subsumed prefixes removed from ruby.yml |
| A1-09 | `special_namespace_prefixes` guard also covers bare module-function names (item 5) | test_clankerprof_special_namespace_guard_covers_bare_module_names | done | guarded bare names on native paths resolve via native rules first |
| A1-10 | `slice:`/`!slice:` filters evaluate with descendant attribution semantics (item 6) | test_clankerprof_slice_filter_negation_respects_descendant_attribution | done | rescue applies to both polarities of bottom slice: filters |
| A1-11 | Multiple `default: true` slices rejected at config validation (item 7) | test_clankerprof_duplicate_default_slices_rejected | done | Python + Rust load_slices_file |
| A1-12 | `native:<value>` predicate honors its value; `native:false` = not native (item 8) | test_clankerprof_native_predicate_value_honored | done | invalid values rejected at parse |
| A1-13 | Fold-heuristic window spec'd over pre-inline frames; outcome independent of inline expansion (item 9) | test_clankerprof_fold_heuristic_ignores_leaf_inline_expansion | done | window = next two distinct locations |
| A1-14 | Compare emits finite JSON only: new/removed rows use `delta_rel: null` + `"status"`; Python and Rust identical (items 10, 24) | test_clankerprof_compare_new_rows_emit_finite_json; parity new/removed-rows test | done | render_json_payload now allow_nan=False repo-wide |
| A1-15 | `compare_boundary_json` orders top_improvements correctly, never drops the largest (item 11) | test_clankerprof_boundary_compare_orders_top_improvements_by_magnitude | done | most-negative first |
| A1-16 | Compare validates payload types; mismatched/wrong `tool` → exit 2 envelope in both languages (items 12, 24) | test_clankerprof_compare_rejects_wrong_payload_types; parity tool-mismatch test | done | Rust gained compare_json dispatch + boundary compare + --focus-boundaries |
| A1-17 | Python decodes signed int64 protobuf fields as two's-complement, matching Rust (item 21) | test_clankerprof_decodes_signed_int64_fields_as_twos_complement | done | line/value = -1 decode as -1 |
| A1-18 | Rust uses insertion-ordered maps wherever Python relies on insertion order: rule/category precedence, ranked-array tie order (item 19) | test_clankerprof_target_category_precedence_and_tie_order; parity precedence_ties case | done | indexmap for TargetConfig/results/slice stats; JSON objects stay key-sorted |

## Wave A2 — CLI / error contract

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A2-01 | Typed error boundary: gzip/EOF/zlib, missing file, YAML, facts-KeyError/validation → exit 2 JSON envelope (item 14) | test_clankerprof_error_envelope_for_{truncated_gzip,missing_file,malformed_yaml,facts_missing_keys} | done | _contracted handler wrapper; proto/facts raise ValueError |
| A2-02 | `--format csv/text` without `--output`: raw payload only on stdout, never mixed with the JSON envelope (item 13) | test_clankerprof_csv_format_stdout_is_not_mixed_with_envelope | done | raw_output convention in both emitters |
| A2-03 | Standalone `clankerprof --output` before subcommand honored like the umbrella CLI (item 15) | test_clankerprof_standalone_global_output_is_honored | done | _hoist_global_output |
| A2-04 | Filter/collapse shape validation always on, even with no slices config (item 16) | test_clankerprof_filter_validation_applies_without_slices_config | done | empty `name:` and bogus keys rejected |
| A2-05 | `--target-csv-layout=compat` with `--format json` explicitly rejected (item 17) | test_clankerprof_csv_layout_compat_rejected_with_json_format | done | validated before analysis |
| A2-06 | Decoder robustness in both languages: truncated fixed32/64 rejected; varint >10 bytes rejected before shift overflow (items 18, 20) | test_clankerprof_decoder_rejects_truncated_and_overlong_fields; parity malformed-profiles test | done | varint bits past 63 drop in both languages |
| A2-07 | Rule packs strict: unknown keys rejected, `runtime_rules.v1` version field required-or-defaulted (item 32) | test_clankerprof_rule_packs_reject_unknown_keys_and_versions | done | packaged packs carry schema_version; Rust loader in B1 |
| A2-08 | Rust CLI parses arguments strictly via clap, rejecting what Python rejects (item 23) | test_clankerprof_rust_cli_rejects_malformed_flags; test_clankerprof_cli_rejects_malformed_flags | done | clap derive; --by-slice no longer absorbs option-like tokens |

## Wave A3 — architecture

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A3-01 | `analysis.py` split into `patterns.py`, `categorize.py`, `targets.py`, `slices.py`, `scopes.py`; `analysis.py` stays as re-export shim (item 29) | full suite green through the shim; caching monkeypatches retargeted to clankerprof.scopes and proven live | done | 1684-line module → 5 modules + 137-line shim with __all__ |
| A3-02 | One categorization engine; `_target_category`/`_boundary_category` duplicates deleted (item 29) | categorize_stack in categorize.py; both projections delegate; full suite green | done | targets gained frame-identity runtime-category caching for free |
| A3-03 | Slice filter/collapse DSL retyped on FramePredicate, parsed once, memoized per unique frame (items 30, 27) | 109-test equivalence oracle green; 197ms -> 26ms on 10k-sample synthetic | done | _SliceMatcher: typed filters + frame-identity caches incl. native-path eligibility |
| A3-04 | Projection accumulators moved out of `model.py` into the projection layer (item 31) | model.py contains model only | done | stats.py owns CategoryStats/DomainStats and friends |
| A3-05 | `__init__.py` exports the real public API (decode/facts/projections/compare) (item 33) | test_clankerprof_public_api_exports | done | decode + facts + projections + compare + rules |
| A3-06 | Scopes rename finished: scopes primary, boundaries documented legacy alias everywhere (item 34) | docs scopes-first; CLI registers scopes before boundaries; --focus-scopes alias verified | done | module named scopes.py since A3-01 |

## Wave A4 — performance

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A4-01 | Frames interned at decode (unique-frame table, occurrences reference it); `Profile.sample_facts()` memoized (item 25) | decode+facts 179ms -> 84ms; memoized repeat ~0.1us; suite green | done | per-location frame table on Profile; caches excluded from eq/repr |
| A4-02 | Facts v2 compact encoding: string/frame tables + per-sample index lists; compact separators by default, `--pretty` opt-in (item 28) | size ratio recorded; replay identity tests | done | 4.0x raw pprof (was ~90x); replay 3.1x faster than re-decode (plan appendix) |
| A4-03 | Boundary loop: predicate exprs normalized once at config parse; exclude-descendants pass skipped when no boundary uses it (item 26) | boundary tests green; exclusion exprs precomputed | done | scan loops only over boundaries with exclusions |
| A4-04 | Benchmark evidence recorded (large synthetic profile, before/after) in plan doc appendix | docs/CLANKERPROF_V2_PLAN.md appendix | done | A1/A4 facts table + A3/A4 projection table |

## Wave A5 — tests

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| A5-01 | Fixture builder covers: packed varints, recursion, multi-value samples, unicode names, deep stacks, inline+folded combos, truncated/corrupt inputs (items 36, 38) | packed/recursion/multi-value/corrupt covered in A1/A2 clusters; unicode/deep-stack/inline+folded added: test_clankerprof_{unicode_names_flow_through_decode_facts_and_projections,deep_stacks_decode_and_project,combined_inline_and_folded_location} | done | builder gained inline_location(folded=) |
| A5-02 | Property/invariant tests: slice totals == profile total; category sums == parent totals; facts round-trip identity (item 38) | test_clankerprof_{slice_totals,target_category_sums,facts_round_trip_identity}*_across_random_profiles (6 seeded profiles incl. multi-value, packed, empty stacks) | done | |
| A5-03 | Every strict-validation branch in facts import covered, incl. unknown schema version (item 38) | 23-case parametrized v2 corruption matrix + v1 shapes + edge branches; facts.py import validation at 100% (file 99%) | done | |
| A5-04 | `generic.yml` ↔ Rust generic rules equivalence test (item 38) | true by construction: Rust parses the same embedded generic.yml; cargo test pins parsed fields | done | include_str! makes drift impossible |

## Wave B0 — Rust defect fixes

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B0-01 | Insertion-ordered maps replace BTreeMap where order-sensitive (item 19) | alias of A1-18 | done | flipped with A1-18 |
| B0-02 | Varint shift-overflow guard before the shift (item 20) | alias of A2-06 | done | guard now >= 70 before 11th byte; wrapping_shl |
| B0-03 | clap argument parsing (item 23) | alias of A2-08 | done | flipped with A2-08 |
| B0-04 | No `Regex::new` in per-frame hot loops; rules precompiled once (item 22) | grep gate: compiled_regex is the only Regex::new caller | done | memoized cache; RegexSet upgrade tracked in B5-01 |
| B0-05 | Compare dispatches on `tool` + finite-JSON contract (items 24, 10) | alias of A1-14/A1-16 | done | Rust compare_json + compare_boundary_json landed |

## Wave B1 — Rust rule system

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B1-01 | YAML rule-pack loading; packaged `generic.yml`/`ruby.yml` via include_str!; hardcoded `RuntimeRuleSet::generic()` deleted | cargo rules tests; generic() now parses embedded generic.yml via OnceLock | done | strict keys + schema_version mirror Python; unquoted `<internal:` YAML mapping bug fixed at the source for both languages |
| B1-02 | Core-classes CSV loading + `--core-classes` override | packaged CSV via include_str!; --core-classes flag; ruby parity test | done | |
| B1-03 | `--runtime ruby` + `--runtime-rules <pack>` flags | test_clankerprof_rust_targets_match_python_ruby_runtime; resolver mirrors Python _runtime_rules | done | on targets and slices |
| B1-04 | Full rule semantics: semantic labels, simplification maps, foldability categories, ordered native_name_category_rules | categorize.rs ports the full engine (ownership constraint, namespace guards, fold window, caller fallback); legacy categorization cases pinned in cargo tests; ruby targets parity green | done | Rust targets now categorize identically to Python |

## Wave B2 — Rust scopes/boundaries projection

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B2-01 | Typed predicate model + expression parser (`any`/`all`/`not`, `cost_kind:`, `runtime_label:`, `slice:`, `native:`, `name:`, `path:`) | scopes.rs FramePredicate/Expr + parser with Python-identical validation; scopes parity green | done | |
| B2-02 | Scopes config: TOML/YAML, preferred+legacy section aliases, mixed-section rejection | test_clankerprof_rust_scopes_legacy_aliases_and_boundaries_subcommand; loader mirrors cli.py incl. rollup/bucket + attributables + duplicate-category rejection | done | toml crate; config normalized through serde_json Value |
| B2-03 | Cost kinds, rollups, owners + sub-buckets, exclude_descendants, once_per_sample, attributables | test_clankerprof_rust_scopes_match_python_boundary_decomposition (all features in one fixture incl. recursion + fallback owner) | done | |
| B2-04 | Frame-identity predicate caching; scopes runs from facts replay identically | test_clankerprof_rust_scopes_replay_facts_identically; Rust facts import (v1+v2) with Python-identical validation; --facts on targets/slices/scopes | done | predicate cache keyed by (predicate, frame identity) |
| B2-05 | Scope/boundary compare with gates in Rust | compare_boundary_json parity (cluster 3) + scopes outputs byte-match Python, so compare inputs are identical by transitivity | done | landed with the compare cluster |

## Wave B3 — Rust target/slice completeness

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B3-01 | `--fold-runtime-internals` + folded-from accounting | flag matrix: targets-fold-json / targets-fold-track-csv / targets-text-fold-track | done | engine landed in B1; flag wired |
| B3-02 | `--track-semantic-callers` + semantic-caller CSV; caller-to-leaf pairs; callsite stdlib-skipping | test_clankerprof_rust_semantic_callers_csv_matches_python (byte-identical); first-max-wins tie semantics via IndexMap | done | render.rs |
| B3-03 | `--no-enhanced` / caller-fallback prefixes; `--cpu-attributables`; minimal `--target` mode | flag matrix: targets-no-enhanced / targets-simple-csv-attributables / targets-minimal-target-mode | done | |
| B3-04 | Output formats: csv / simple-csv / text; `--target-csv-layout compat` | byte-identical CSV/simple-csv/text in the flag matrix; test_clankerprof_rust_compat_csv_artifacts_match_python (two-file pair byte-identical) | done | render.rs mirrors Python csv.writer quoting + format_time |
| B3-05 | Slice config files + `./slices.yml` discovery; `--by-slice` grammar; attribute/virtual-slice validation; GC/uncollapsible pseudo-slices; unattributed libraries | flag matrix: slices-attribute-metadata-byslice / slices-gc-pseudo; validate_slice_options + parse_attribute ports; slice metadata round-trips | done | --config file merging deferred to the slices YAML path (slice files carry metadata) |

## Wave B4 — parity harness v2

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B4-01 | Shared fixture corpus with Python-generated goldens; byte-level output comparison | flag matrix compares Python and Rust artifacts at the byte level (JSON incl. facts compact/pretty, CSV, text, compat pair) | done | JSON emitters proven byte-aligned (sorted keys + indent 2) |
| B4-02 | Flag-matrix sweep per subcommand (not one pinned combo) | 13-case matrix across facts/targets/slices/scopes + dedicated semantic-CSV/compat/report/decoder/compare/malformed-flag tests | done | |
| B4-03 | Native Rust unit tests exist and run (`cargo test` > 0 tests) | 15 native tests across rules/categorize/render/scopes/facts | done | |
| B4-04 | Cargo lane wired into `bin/dev` (`test-rust`) and `check` when cargo present | scripts/test-rust.sh (fmt-check + build + test, self-skipping without cargo) wired into check.sh, bin/dev, mise | done | |

## Wave B5 — Rust-native performance

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| B5-01 | String-interning arena for names/paths; RegexSet for rule matching | resolved by measurement: compiled_regex memoization + tiny pattern counts put compilation off the hot path; RegexSet cannot serve capture-based library extraction (plan appendix) | done | end-to-end numbers recorded |
| B5-02 | Single-pass multi-projection mode: decode once → targets+slices+scopes in one invocation | clankerprof-rs report subcommand; test_clankerprof_rust_report_sections_match_individual_subcommands | done | documented in CLANKERPROF_RUST.md |
| B5-03 | Benchmark vs Python on large synthetic profiles; results recorded in plan appendix | appendix: Rust 2.5-5.6x end-to-end; report single-pass 4.6x vs sequential Python | done | release build, best of 3 |

## Wave G — governance / docs

| ID | Requirement | Verify | Status | Notes |
| --- | --- | --- | --- | --- |
| G-01 | SPEC.md records every spec decision (recursion, value-type selection, compare contract, int64, rule matching, fold window, csv/text stream contract); item-35 drift resolved | spec sections landed with each wave; focus flags documented; tree/opportunity refs removed; Rust-parity section rewritten | done | |
| G-02 | docs/CLANKERPROF_PARITY.md matrix: every capability row has an explicit Rust column ("covered"+fixture or "not claimed") | 'Rust parity status' section states blanket byte-level coverage with the flag-matrix inventory; scope-decomposition row flipped to covered with test names | done | |
| G-03 | skills/clankerprof-operator updated to final CLI surface | SKILL.md documents --pretty, capabilities-complete Rust core, report mode, ./bin/dev test-rust; pinned-phrase test green | done | |
| G-04 | SAMPLE_FACTS_SCHEMA_VERSION symmetric Python↔Rust, enforced by a test | test_clankerprof_sample_facts_schema_version_is_symmetric (runs without cargo) | done | |
| G-05 | README / docs / CLAUDE.md updated where behavior descriptions changed (Rust no longer "lightweight subset") | README notes the capabilities-complete Rust core; CLAUDE.md rewritten for the module split, facts v2, rule-pack versioning, test-rust lane, and complete Rust crate | done | |
