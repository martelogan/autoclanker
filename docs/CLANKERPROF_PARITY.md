# clankerprof parity matrix

This matrix records the compatibility contract for `clankerprof`.
Target-function attribution is the first replacement milestone and the source
of truth for `clankerprof targets`. Slice coverage is secondary, additive
migration coverage for `clankerprof slices` and `clankerprof compare`; slice
behavior must not define or override target-attribution semantics.

## Status meanings

- `covered`: enforced by self-contained tests in `tests/test_clankerprof.py`.
- `covered with intentional extension`: compatible with the older workflow while
  preserving a useful `clankerprof` extension.
- `not claimed`: not proven by the current self-contained suite.

## Rust parity status

`crates/clankerprof-core` reaches parity with every capability below through
`tests/test_clankerprof_rust_parity.py`: byte-level artifact comparison across
a per-subcommand flag matrix (facts compact/pretty, targets
json/csv/simple-csv/text with runtime/fold/verbose/track/attributable/
no-enhanced flags and the compat CSV pair, slices with filters/collapse/
attributes/metadata/pseudo-slices, scopes with preferred and legacy configs,
facts replay, and the no-enhanced/fold/verbose runtime flags, compare gates,
malformed-input rejection). Rows marked `not claimed`
below are not claimed by either implementation (legacy prose formats). Any
capability added to Python must land in the Rust core and this parity suite
in the same change.

## Target attribution

| Capability | Status | Test coverage |
| --- | --- | --- |
| Raw and gzipped pprof profile decoding without generated protobuf runtime files | covered | `test_clankerprof_decodes_raw_and_gzipped_pprof_profiles` |
| Value-type metadata (`sample_type`, `period_type`, `period`, `default_sample_type`) selects the primary value per pprof convention | covered | `test_clankerprof_primary_value_defaults_to_last_value_type`, `test_clankerprof_primary_value_honors_default_sample_type`, `test_clankerprof_primary_value_unknown_default_falls_back_to_last` |
| Packed varint sample encoding decodes identically to unpacked encoding | covered | `test_clankerprof_decodes_packed_sample_encoding_identically` |
| Signed int64 protobuf fields (sample values, line numbers) decode as two's-complement | covered | `test_clankerprof_decodes_signed_int64_fields_as_twos_complement` |
| Inline frames from repeated `Location.line` entries participate in leaf-to-root target traversal | covered | `test_clankerprof_expands_inline_location_frames_for_target_traversal` |
| First-class sample-facts API exposes per-sample value, leaf-to-root stack, inline frames, and stable sample index | covered | `test_clankerprof_sample_facts_are_the_shared_projection_surface` |
| Versioned sample-facts JSON export/import preserves projection behavior and rejects malformed frame entries | covered | `test_clankerprof_sample_facts_export_round_trips_projection_inputs`, `test_clankerprof_sample_facts_import_rejects_malformed_frames` |
| Strict facts numeric domains: uint64 IDs round-trip (each language replays its own export), non-integral or out-of-range numbers are validation errors, and non-finite JSON tokens are rejected in both languages | covered | `test_clankerprof_facts_import_accepts_uint64_ids`, `test_clankerprof_facts_import_rejects_non_integral_numeric_fields`, `test_clankerprof_rust_facts_uint64_ids_round_trip`, `test_clankerprof_rust_facts_numeric_contract_matches_python`, `test_clankerprof_json_inputs_reject_non_finite_tokens` |
| Exact aggregate parity with a documented bound: totals beyond `i64::MAX` (up to `u64::MAX`) serialize identically, sums beyond the bound exit 2 with matching envelopes, and no input can panic either implementation | covered | `test_clankerprof_facts_aggregate_bounds`, `test_clankerprof_rust_big_aggregates_match_python_exactly`, `test_clankerprof_rust_facts_numeric_contract_matches_python` |
| Projection-neutral fact index exposes shared stack helpers without owning target or slice policy | covered | `test_clankerprof_fact_index_exposes_shared_stack_operations` |
| Target projection can consume sample facts directly with identical output to profile-based analysis | covered | `test_clankerprof_target_projection_matches_sample_fact_projection` |
| Target CLI can replay `clankerprof facts` JSON through standalone and umbrella command surfaces | covered | `test_clankerprof_cli_and_autoclanker_alias_generate_outputs` |
| Target-contained sample self-time attribution by leaf frame | covered | `test_clankerprof_preserves_target_attribution_parity` |
| Recursive parent frames attribute the sample value once per sample per parent | covered | `test_clankerprof_targets_recursive_frames_count_once_per_sample` |
| Category precedence is first-match-wins in config order; ranked arrays break ties by first-seen order (Rust parity via insertion-ordered maps) | covered | `test_clankerprof_target_category_precedence_and_tie_order` |
| Complete target accounting with `Other` catch-all | covered | `test_clankerprof_preserves_target_attribution_parity` |
| Generic request/rendering boundary attribution outside a specific application domain | covered | `test_clankerprof_supports_generic_request_rendering_attribution` |
| Packaged Ruby core class CSV is used by default for `--runtime ruby`, with explicit override still available; quoted CSV fields parse with `csv.reader` semantics in both languages | covered | `test_clankerprof_loads_packaged_ruby_core_classes_by_default`, `test_clankerprof_rust_target_renderer_semantics_match_python`, Rust `core_classes_csv_first_field_matches_python_csv_reader` |
| Ruby core/native semantic labels including OpenSSL false-positive avoidance | covered | `test_clankerprof_ruby_rule_pack_preserves_legacy_categorization_cases` |
| Semantic rules never claim application-path frames by name substring (runtime-owned-path constraint) | covered | `test_clankerprof_semantic_rules_do_not_claim_app_frames_by_substring` |
| Guarded bare module-function names (`Zlib.inflate`) resolve through native-name rules, not the core default | covered | `test_clankerprof_special_namespace_guard_covers_bare_module_names` |
| `--no-enhanced` keeps the active runtime's pack; generic never silently loads ruby rules | covered | `test_clankerprof_no_enhanced_generic_runtime_keeps_generic_rules`, `test_clankerprof_no_enhanced_runtime_selection_is_observable` |
| Four legacy Ruby flag combinations for verbose/non-verbose and folded/non-folded modes | covered | `test_clankerprof_preserves_legacy_ruby_flag_combinations` |
| Main simplified categories and “main categories never fold” behavior | covered | `test_clankerprof_preserves_main_simplified_category_totals_and_never_folds` |
| Proportional CPU attributables in CSV output | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Folded-from accounting | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Folded and semantic caller sections are available in text reports | covered with intentional extension | `test_clankerprof_text_report_includes_folded_and_semantic_caller_sections` |
| Caller-site summaries skip runtime stdlib delegators when choosing the meaningful caller, scanning the whole stack (no fixed-depth window); the immediate caller is the fallback only when no frame is eligible | covered | `test_clankerprof_target_csv_skips_runtime_stdlib_when_selecting_callsite`, `test_clankerprof_caller_selection_scans_past_deep_native_runs`, `test_clankerprof_rust_target_renderer_semantics_match_python` |
| Semantic caller tracking and compatibility top-caller CSV export shape | covered | `test_clankerprof_exports_semantic_caller_csv` |
| `--no-enhanced` native/delegated caller fallback before regex matching, configured through `caller_fallback_name_prefixes` | covered | `test_clankerprof_supports_legacy_no_enhanced_native_caller_fallback`, `test_clankerprof_caller_fallback_prefixes_are_generic_rule_config` |
| Compatibility two-file CSV artifact pair, `output/<name>` and `output/verbose/<name>`, via `--target-csv-layout compat` or the older alias flag | covered with explicit opt-in | `test_clankerprof_can_emit_legacy_target_csv_artifact_pair` |
| Legacy text report phrasing byte-for-byte | not claimed | Prefer structured JSON/CSV compatibility over exact prose output. |

| Parents emit in first-seen encounter order across csv/simple-csv/text and the compat pair | covered | `test_clankerprof_rust_target_parent_order_matches_python` |

## Scope decomposition

This section is scoped to `clankerprof scopes` and its compatibility alias
`clankerprof boundaries`. It absorbs the richer scope/cost-kind/rollup/owner
analysis surface without changing legacy `targets` JSON config semantics,
existing boundary JSON payloads, or slice ownership semantics.

| Capability | Status | Test coverage |
| --- | --- | --- |
| Declarative TOML/YAML scope config with `[cost_kind]`, optional `[owner]`, `[[scope]]`, `[scope.rollup]`, and `[scope.attributables]`; legacy `[category]`, `[domain]`, `[[boundary]]`, `[boundary.bucket]`, and `[boundary.attributables]` remain accepted | covered | `test_clankerprof_boundary_config_cli_replays_sample_facts`, `test_clankerprof_scope_config_aliases_preserve_boundary_output`, `test_clankerprof_scope_config_rejects_mixed_preferred_and_legacy_sections` |
| Cost-kind rows and scope rollups under one parent denominator | covered | `test_clankerprof_boundary_decomposition_tracks_domain_cost_kinds` |
| Owner rows preserve cost-kind sub-buckets under the same scope denominator | covered | `test_clankerprof_boundary_decomposition_tracks_domain_cost_kinds` |
| Top owner files and representative owner function -> hot leaf evidence | covered | `test_clankerprof_boundary_decomposition_tracks_domain_cost_kinds` |
| Scope reports can replay versioned sample facts through the standalone and umbrella CLI surfaces | covered | `test_clankerprof_boundary_config_cli_replays_sample_facts` |
| Residual parent scopes via `exclude_descendants` | covered | `test_clankerprof_boundary_exclusions_build_residual_scopes` |
| Frame-predicate matching is cached by unique frame identity rather than repeated per sample occurrence | covered | `test_clankerprof_boundary_predicate_matching_is_cached` |
| Predicate expressions with `any`, `all`, and `not`, configured `cost_kind:<label>`/`category:<label>` selectors, runtime `runtime_label:<label>`/`runtime_category:<label>` selectors, and recursive cost-kind guardrails | covered | `test_clankerprof_boundary_config_supports_predicate_expressions_and_category_refs`, `test_clankerprof_scope_config_aliases_preserve_boundary_output`, `test_clankerprof_boundary_config_rejects_recursive_category_predicates` |
| Nested predicate expressions remain cached by unique frame identity across repeated samples | covered | `test_clankerprof_boundary_expression_matching_stays_frame_cached` |
| Configured category selectors remain cached by unique frame identity when used as owner-domain predicates | covered | `test_clankerprof_boundary_category_predicates_stay_frame_cached` |
| Rust parity for scope decomposition | covered | `test_clankerprof_rust_scopes_match_python_boundary_decomposition`, `test_clankerprof_rust_scopes_legacy_aliases_and_boundaries_subcommand`, `test_clankerprof_rust_scopes_replay_facts_identically` |
| Scope aggregates in `(i64::MAX, u64::MAX]` render, filter, and sort identically; occurrence-weighted totals beyond the aggregate bound exit 2 with matching envelopes; signed-minimum `--top`/`--by-slice`/`--unattributed-libraries` limits truncate without panicking | covered | `test_clankerprof_rust_numeric_edge_semantics_match_python`, `test_clankerprof_scope_occurrence_aggregates_fail_closed_beyond_bound`, `test_clankerprof_slices_tail_limits_accept_i64_min` |
| Zero-aggregate rollup rows are omitted while negative rows render, keeping bucket/owner sums equal to scope totals byte-identically | covered | `test_clankerprof_rust_numeric_edge_semantics_match_python`, `test_clankerprof_scope_rollups_render_negative_costs_additively` |

| Cost-kind/owner tables evaluate in declaration order (first match wins) in YAML and TOML | covered | `test_clankerprof_scope_tables_respect_declaration_order`, `test_clankerprof_rust_scope_declaration_order_matches_python` |
| Scope `label`/`name` must be strings and `function` a string or string array; violations exit 2 with matching envelopes | covered | `test_clankerprof_scope_labels_must_be_strings`, `test_clankerprof_rust_scope_declaration_order_matches_python` |
| Slice `default` accepts only YAML booleans (absent/null read as false; truthiness coercion rejected with a shared message) | covered | `test_clankerprof_config_string_fields_fail_closed`, `test_clankerprof_rust_slice_default_boolean_matches_python` |
| Scope runtime flags (`--no-enhanced`, `--fold-runtime-internals`, `--verbose-runtime-internals`) parity | covered | `test_clankerprof_rust_runtime_flags_match_python` |

## Runtime rules

| Capability | Status | Test coverage |
| --- | --- | --- |
| Ruby support is opt-in instead of the core engine default | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Runtime labels, simplification maps, foldability, namespace exclusions, stdlib paths, native paths, dependency-library extraction, and selector-specific dependency paths are loaded from packaged or external config | covered | `test_clankerprof_ruby_rule_pack_preserves_legacy_categorization_cases`, `test_clankerprof_dependency_selectors_are_runtime_rule_driven`, `test_clankerprof_loads_external_runtime_rule_packs`, `test_clankerprof_slice_native_collapse_uses_runtime_rules` |
| Application-specific labels live in the Ruby rule pack rather than hardcoded core logic | covered | `clankerprof/runtime_rules/ruby.yml` plus categorization tests |
| Ordered native-name category rules can classify folded native constructor callers before broad fallbacks | covered | `test_clankerprof_loads_external_runtime_rule_packs` |
| Project-local runtime rule packs with custom semantic labels, library extraction, foldability, and CLI usage | covered | `test_clankerprof_loads_external_runtime_rule_packs` |

## Ownership slice attribution

This section is intentionally scoped to `clankerprof slices`. It exists so the
package can absorb useful slice/report workflows without requiring the
target-attribution mode to inherit slice-specific CLI, config, or validation
semantics.

| Capability | Status | Test coverage |
| --- | --- | --- |
| Slice path attribution and default catch-all slice | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare` |
| Slice projection can consume sample facts directly with identical output to profile-based analysis | covered | `test_clankerprof_slice_projection_matches_sample_fact_projection` |
| Slice CLI can replay `clankerprof facts` JSON through standalone, umbrella, and config-file inputs | covered | `test_clankerprof_cli_uses_sample_facts_for_slice_projection` |
| Generic declarative slice metadata in JSON output | covered | `test_clankerprof_slice_cli_supports_config_and_output_limits` |
| Collapse rules, including `library:*` and legacy `gem:*` | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare`, `test_clankerprof_slice_paths_support_legacy_regex_and_simplified_patterns` |
| Attribution overrides, including invalid/duplicate filter rejection | covered | `test_clankerprof_slice_cli_validates_attribute_contract` |
| Virtual output slices from `--attribute` | covered with explicit opt-in extension | `test_clankerprof_slice_cli_validates_attribute_contract` |
| Descendant filters such as `<name:RequestHandler#render_response`, including OR semantics across descendant filters | covered | `test_clankerprof_slice_descendant_filters_use_or_semantics` |
| Repeated `!` and `<` filter prefixes in any order | covered | `test_clankerprof_slice_filter_prefixes_can_repeat_in_any_order` |
| Bottom `slice:<name>` filters include descendant-attribute rescue behavior | covered | `test_clankerprof_slice_filter_honors_descendant_attribute_slice_matches` |
| Negated `!slice:<name>` filters exclude descendant-attributed samples (rescue applies to both polarities) | covered | `test_clankerprof_slice_filter_negation_respects_descendant_attribution` |
| Multiple `default: true` slices are rejected | covered | `test_clankerprof_duplicate_default_slices_rejected` |
| `native:false` predicates match non-native frames; invalid values rejected | covered | `test_clankerprof_native_predicate_value_honored` |
| Runtime-internal fold window spans distinct locations, invariant to leaf inline expansion | covered | `test_clankerprof_fold_heuristic_ignores_leaf_inline_expansion` |
| `slice:<name>` collapse does not use descendant-attribute rescue | covered | `test_clankerprof_slice_collapse_does_not_use_descendant_attribute_rescue` |
| Non-descendant filters apply to the selected bottom attribution frame after native/collapse handling | covered | `test_clankerprof_slice_filters_apply_to_bottom_frame_after_native_collapse` |
| Unsupported filter keys, malformed filters, and collapse prefixes are rejected | covered | `test_clankerprof_slice_cli_validates_filter_and_collapse_contract` |
| Duplicate scalar config and CLI options are rejected instead of silently merged | covered | `test_clankerprof_slice_cli_rejects_duplicate_scalar_config_options`, `test_clankerprof_rust_slices_validation_envelopes_match_python` |
| `--by-slice=N`, `--by-slice=N%`, and bare `--by-slice` | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice` |
| `--by-slice` fails closed on unparsable values and non-finite thresholds; negative limits drop slices from the tail, identically across languages | covered | `test_clankerprof_by_slice_values_validate_and_support_negative_limits`, `test_clankerprof_rust_slices_validation_envelopes_match_python`, `test_clankerprof_rust_cli_flag_matrix_matches_python` |
| YAML config for common slice options | covered | `test_clankerprof_slice_cli_supports_config_and_output_limits`, `test_clankerprof_rust_cli_flag_matrix_matches_python` |
| TOML config with profile path, `filter`, and scalar options | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice`, `test_clankerprof_rust_cli_flag_matrix_matches_python` |
| `slices --config` merge semantics (value coercions, duplicate rejection, error ordering) in the Rust CLI | covered | `test_clankerprof_rust_cli_flag_matrix_matches_python`, `test_clankerprof_rust_slices_validation_envelopes_match_python` |
| Default `./slices.yml` discovery when slice-aware options are used | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice` |
| Bracket-class globs (`[seq]`, `[!seq]`, ranges, unterminated `[` literal) attribute identically across languages | covered | `test_clankerprof_rust_cli_flag_matrix_matches_python` (targets-bracket-class-globs), Rust `glob_matches_follows_cpython_fnmatch_semantics` |
| Python-dialect regexes (lookaround included) load and match in both languages; explicit invalid regexes fail closed with a shared message prefix | covered | `test_clankerprof_rust_regex_dialect_and_failclosed_match_python`, `test_clankerprof_invalid_regex_patterns_fail_closed` |
| Configured cost-kind/category predicate errors fail closed in both languages (no silent `Other` rebucketing) | covered | `test_clankerprof_rust_slices_validation_envelopes_match_python` |
| Multi-member (concatenated) gzip profiles decode fully in both languages | covered | `test_clankerprof_rust_cli_flag_matrix_matches_python` (facts-multimember-gzip), Rust `concatenated_gzip_members_all_decode` |
| GC pseudo-output for `(marking)` and `(sweeping)` | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Uncollapsible pseudo-output when all eligible frames are collapsed, using root eligible frame reporting | covered | `test_clankerprof_uncollapsible_reports_root_eligible_frame` |
| Unattributed library summaries for default slice views, with legacy gem output retained | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Slice frame output includes decoded pprof line numbers when present | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Terminal text formatting, coloring, width wrapping, and timing output from older slice tools | not claimed | `clankerprof` prioritizes machine-readable JSON plus CSV/text target reports. |
| Domain-specific responsibility fields as first-class slice concepts | not claimed | Slice metadata is preserved generically; owner decomposition is covered separately by `clankerprof scopes`. |

## Compare

| Capability | Status | Test coverage |
| --- | --- | --- |
| Structured slice comparison with absolute/relative thresholds | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare` |
| Structured boundary comparison over boundary, bucket, category, and domain rows | covered | `test_clankerprof_compare_supports_boundary_outputs` |
| Non-zero regression gate exit code | covered | `test_clankerprof_compare_exits_nonzero_for_regression_gate` |
| Top per-function regressions and improvements in JSON | covered | `test_clankerprof_compare_exits_nonzero_for_regression_gate` |
| Strict-JSON compare artifacts: new/removed rows serialize `delta_rel` as `null`, never `Infinity` | covered | `test_clankerprof_compare_new_rows_emit_finite_json` |
| Boundary `top_improvements` ordered by magnitude, most negative first | covered | `test_clankerprof_boundary_compare_orders_top_improvements_by_magnitude` |
| Compare dispatches on the shared `tool` field and rejects non-report payloads | covered | `test_clankerprof_compare_rejects_wrong_payload_types` |
| Compare fails closed on malformed reports: missing `slices`/`boundaries` arrays, non-numeric fields, and non-finite thresholds exit 2 with matching envelopes in both languages | covered | `test_clankerprof_compare_rejects_missing_row_arrays_and_bad_numbers`, `test_clankerprof_compare_rejects_non_finite_thresholds`, `test_clankerprof_rust_compare_rejects_malformed_reports_like_python` |
| `compare --focus-scopes` alias of `--focus-boundaries` in both CLIs | covered | `test_clankerprof_rust_cli_flag_matrix_matches_python` (compare-focus-scopes-alias) |
| Frames sharing a function name aggregate (sum) in per-function deltas, byte-identically across languages | covered | `test_clankerprof_compare_aggregates_duplicate_function_frames`, `test_clankerprof_rust_compare_aggregates_duplicate_functions_like_python` |
| Summary `total_time_ns` accepted across the full aggregate range `[i64::MIN, u64::MAX]` and echoed exactly in both languages | covered | `test_clankerprof_compare_summary_totals_span_u64_range`, `test_clankerprof_rust_numeric_edge_semantics_match_python` |
| Present compare rows must carry string names and their numeric fields; field absence never coerces to zero while row-level absence keeps new/removed semantics | covered | `test_clankerprof_compare_rejects_rows_missing_required_fields`, `test_clankerprof_rust_compare_row_strictness_matches_python` |
| Compare reports require a `summary` object with an integer `total_time_ns` in both languages | covered | `test_clankerprof_compare_rejects_rows_missing_required_fields`, `test_clankerprof_rust_compare_row_strictness_matches_python` |
| Facts v2 samples require `values`, `location_ids`, and `stack`; missing keys exit 2 with matching envelopes (v1 keeps documented leniency) in both languages | covered | `test_clankerprof_facts_import_rejects_each_invalid_v2_shape`, `test_clankerprof_rust_facts_and_scope_validation_matches_python` |
| Empty predicate tables and non-string scope `count` values fail closed in both languages; owner `fallback` follows Python truthiness | covered | `test_clankerprof_scope_config_rejects_empty_tables_and_bad_count`, `test_clankerprof_rust_facts_and_scope_validation_matches_python` |
| Library regexes name libraries via participating group 1 with whole-match fallback, byte-identically across languages | covered | `test_clankerprof_library_regex_group_fallback`, `test_clankerprof_rust_library_regex_fallback_matches_python` |
| Text compare report wording from older slice tools | not claimed | JSON gate compatibility is the stable contract. |

## CLI stream contract

| Capability | Status | Test coverage |
| --- | --- | --- |
| Usage/option errors exit 2 with the JSON error envelope in both standalone CLIs | covered | `test_clankerprof_cli_rejects_malformed_flags`, `test_clankerprof_rust_cli_rejects_malformed_flags`, `test_clankerprof_rust_output_receipts_and_usage_envelopes_match_python` |
| Successful `--output` writes print the JSON receipt, byte-identically across languages | covered | `test_clankerprof_output_writes_print_json_receipts`, `test_clankerprof_rust_output_receipts_and_usage_envelopes_match_python` |
| `compare --output` (local or hoisted global placement) writes the report, prints a `has_regression` receipt, and keeps the regression exit code | covered | `test_clankerprof_compare_output_receipt_preserves_regression_exit`, `test_clankerprof_rust_output_receipts_and_usage_envelopes_match_python` |
| `facts` stdout carries the artifact bytes: compact by default, `--pretty` opt-in | covered | `test_clankerprof_facts_stdout_matches_artifact_bytes`, `test_clankerprof_rust_output_receipts_and_usage_envelopes_match_python` |
| Non-facts JSON artifacts, receipts, and envelopes use Python `json.dumps` lexical form (`\uXXXX` escapes, CPython float repr) byte-identically | covered | `test_clankerprof_rust_lexical_json_matches_python`, Rust `pyjson` unit tests |
| Duplicate YAML mapping keys rejected in both languages with the shared `duplicate entry with key` message | covered | `test_clankerprof_yaml_inputs_reject_duplicate_keys`, `test_clankerprof_rust_scope_declaration_order_matches_python` |
| Integer CLI flags (`--top`, `--unattributed-libraries`) share the strict signed-64 ASCII grammar; signed limits keep Python `list[:n]` tail semantics incl. `scopes --top` | covered | `test_clankerprof_cli_integer_flags_use_strict_int64_grammar`, `test_clankerprof_scopes_negative_top_drops_from_tail`, `test_clankerprof_rust_value_domain_grammar_matches_python` |
| Focus flags take one comma-delimited value; a repeated flag keeps the last occurrence in both CLIs | covered | `test_clankerprof_compare_focus_flags_take_one_comma_delimited_value`, `test_clankerprof_rust_value_domain_grammar_matches_python` |
| Non-string YAML mapping keys rejected on every YAML surface with the shared message; no YAML 1.1 timestamp resolution | covered | `test_clankerprof_yaml_inputs_reject_non_string_mapping_keys`, `test_clankerprof_rust_value_domain_grammar_matches_python` |
| Selector and string-or-array config fields require string entries in both languages | covered | `test_clankerprof_scope_selector_arrays_require_string_entries`, `test_clankerprof_rust_value_domain_grammar_matches_python` |
| Plain YAML scalar resolution follows serde_yaml's dialect (booleans `true`/`false` only, no YAML 1.1 int/float forms, out-of-64-bit ints fail the parse, overflowing float literals stay strings) — table pinned in both engines | covered | `test_clankerprof_strict_yaml_scalars_match_serde_yaml`, Rust `yaml_scalar_semantics` integration tests, `test_clankerprof_rust_yaml_scalars_and_attributables_match_python` |
| String-typed integer config fields share the trim + ASCII signed-decimal i64 grammar | covered | `test_clankerprof_rust_yaml_scalars_and_attributables_match_python` |
| Attributable metric values must be JSON numbers (booleans/strings rejected, shared messages) in `--cpu-attributables` and scope `attributables` | covered | `test_clankerprof_attributables_reject_non_numeric_values`, `test_clankerprof_rust_yaml_scalars_and_attributables_match_python` |
| JSON integer literals outside `[i64::MIN, u64::MAX]` behave identically (f64 fallback vs unbounded int: same f64 in float domains, same rejection in integer domains) | covered | `test_clankerprof_json_out_of_range_integers_reject_in_integer_domains`, `test_clankerprof_rust_yaml_scalars_and_attributables_match_python` |
| Signed compare gating: relative deltas divide by the baseline magnitude, zero baselines are unbounded in the delta's direction (null-serialized) | covered | `test_clankerprof_compare_gates_signed_rows`, `test_clankerprof_rust_round3_fixes_match_python` |
| Duplicate top-level compare rows (slices; boundary/bucket/category/domain) are validation errors, killing order-dependent gate bypasses | covered | `test_clankerprof_compare_rejects_duplicate_rows`, `test_clankerprof_rust_round3_fixes_match_python` |
| Duplicate JSON object member names are validation errors on every JSON input surface (compare reports, facts artifacts, target configs, attributables) — shared `duplicate entry with key` message core, never last-wins | covered | `test_clankerprof_json_inputs_reject_duplicate_member_names`, `test_clankerprof_rust_compare_fail_closed_matches_python` |
| Present-but-null nested row arrays (`frames`/`buckets`/`categories`/`domains`) are validation errors, matching absent-vs-null semantics across languages | covered | `test_clankerprof_compare_rejects_present_null_row_arrays`, `test_clankerprof_rust_compare_fail_closed_matches_python` |
| Derived compare values (frame sums, absolute deltas) fail closed on overflow with a shared typed message; only `delta_rel` has a documented null path | covered | `test_clankerprof_compare_derived_values_fail_closed_on_overflow`, `test_clankerprof_rust_compare_fail_closed_matches_python` |
| Non-finite attributable estimates (input or scaled) fail closed with the shared metric-naming message | covered | `test_clankerprof_attributable_overflow_fails_closed`, `test_clankerprof_rust_round3_fixes_match_python` |
| Percentage arithmetic uses f64-operand division in both languages; floats re-emit exactly (serde_json float_roundtrip); pinned above 2^53 incl. `--by-slice` selection | covered | `test_clankerprof_rust_round3_fixes_match_python` |
| Negative GC/uncollapsible pseudo-outputs render (zero stays omitted), keeping slice artifacts additive | covered | `test_clankerprof_pseudo_slices_render_negative_aggregates`, `test_clankerprof_rust_round3_fixes_match_python` |
| Zero-total targets render every percentage as 0 across json/csv/simple-csv/text (no division error, `inf`, or `NaN`; text TOTAL reports 0.00%) | covered | `test_clankerprof_zero_total_targets_render_without_division_errors`, `test_clankerprof_rust_round3_fixes_match_python` |
| The simplified target noise gate is magnitude-aware: material negative categories render in simple-csv and the compat simplified CSV | covered | `test_clankerprof_simple_csv_renders_negative_categories`, `test_clankerprof_rust_target_renderer_semantics_match_python` |
| Target attributable estimates fail closed on non-finite values at load and at proportional scaling, in every CSV layout | covered | `test_clankerprof_target_attributable_overflow_fails_closed`, `test_clankerprof_rust_target_renderer_semantics_match_python` |
| String strictness on slice paths/names, target config patterns, and runtime-rule fields (shared messages, no str()-coercion or silent drops) | covered | `test_clankerprof_config_string_fields_fail_closed`, `test_clankerprof_rust_round3_fixes_match_python` |
| Compare thresholds share the strict float grammar (`1_0`, ` 2 `, NaN, overflow all rejected identically) and must be non-negative (a negative threshold would gate identical reports); zero stays legal | covered | `test_clankerprof_compare_threshold_flags_use_strict_grammar`, `test_clankerprof_rust_round3_fixes_match_python`, `test_clankerprof_rust_compare_fail_closed_matches_python` |

## Remaining confidence boundary

The self-contained tests now cover the mined target-attribution surface plus
the separate boundary and slice/report surfaces listed above. They do not prove
byte-for-byte replacement of every historical profile report. Before deleting
older tools, run golden replays against real `.pb` / `.pb.gz` profiles and
compare target category/caller CSV totals, boundary JSON totals, and slice JSON
totals as separate migration checks.
