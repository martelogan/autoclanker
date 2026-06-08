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

## Target attribution

| Capability | Status | Test coverage |
| --- | --- | --- |
| Raw and gzipped pprof profile decoding without generated protobuf runtime files | covered | `test_clankerprof_decodes_raw_and_gzipped_pprof_profiles` |
| Inline frames from repeated `Location.line` entries participate in leaf-to-root target traversal | covered | `test_clankerprof_expands_inline_location_frames_for_target_traversal` |
| Target-contained sample self-time attribution by leaf frame | covered | `test_clankerprof_preserves_target_attribution_parity` |
| Complete target accounting with `Other` catch-all | covered | `test_clankerprof_preserves_target_attribution_parity` |
| Generic request/rendering boundary attribution outside a specific application domain | covered | `test_clankerprof_supports_generic_request_rendering_attribution` |
| Packaged Ruby core class CSV is used by default for `--runtime ruby`, with explicit override still available | covered | `test_clankerprof_loads_packaged_ruby_core_classes_by_default` |
| Ruby core/native semantic labels including OpenSSL false-positive avoidance | covered | `test_clankerprof_ruby_rule_pack_preserves_legacy_categorization_cases` |
| Four legacy Ruby flag combinations for verbose/non-verbose and folded/non-folded modes | covered | `test_clankerprof_preserves_legacy_ruby_flag_combinations` |
| Main simplified categories and “main categories never fold” behavior | covered | `test_clankerprof_preserves_main_simplified_category_totals_and_never_folds` |
| Proportional CPU attributables in CSV output | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Folded-from accounting | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Folded and semantic caller sections are available in text reports | covered with intentional extension | `test_clankerprof_text_report_includes_folded_and_semantic_caller_sections` |
| Caller-site summaries skip runtime stdlib delegators when choosing the meaningful caller | covered | `test_clankerprof_target_csv_skips_runtime_stdlib_when_selecting_callsite` |
| Semantic caller tracking and legacy top-caller CSV export shape | covered | `test_clankerprof_exports_semantic_caller_csv` |
| Legacy `--no-enhanced` native-caller fallback before regex matching | covered | `test_clankerprof_supports_legacy_no_enhanced_native_caller_fallback` |
| Legacy two-file CSV artifact pair, `output/<name>` and `output/verbose/<name>` | covered with explicit opt-in | `test_clankerprof_can_emit_legacy_target_csv_artifact_pair` |
| Legacy text report phrasing byte-for-byte | not claimed | Prefer structured JSON/CSV compatibility over exact prose output. |

## Runtime rules

| Capability | Status | Test coverage |
| --- | --- | --- |
| Ruby support is opt-in instead of the core engine default | covered | `test_clankerprof_ruby_rules_support_simplification_folding_and_attributables` |
| Runtime labels, simplification maps, foldability, namespace exclusions, stdlib paths, native paths, and dependency-library extraction are loaded from packaged config | covered | `test_clankerprof_ruby_rule_pack_preserves_legacy_categorization_cases`, `test_clankerprof_dependency_selectors_are_runtime_rule_driven`, `test_clankerprof_slice_native_collapse_uses_runtime_rules` |
| Application-specific labels live in the Ruby rule pack rather than hardcoded core logic | covered | `clankerprof/runtime_rules/ruby.yml` plus categorization tests |
| Fully arbitrary third-party runtime rule packs | not claimed | The loader shape supports more packs, but only generic and Ruby rule packs are packaged and tested. |

## Ownership slice attribution

This section is intentionally scoped to `clankerprof slices`. It exists so the
package can absorb useful slice/report workflows without requiring the
target-attribution mode to inherit slice-specific CLI, config, or validation
semantics.

| Capability | Status | Test coverage |
| --- | --- | --- |
| Slice path attribution and default catch-all slice | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare` |
| Generic declarative slice metadata in JSON output | covered | `test_clankerprof_slice_cli_supports_config_and_output_limits` |
| Collapse rules, including `library:*` and legacy `gem:*` | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare`, `test_clankerprof_slice_paths_support_legacy_regex_and_simplified_patterns` |
| Attribution overrides, including invalid/duplicate filter rejection | covered | `test_clankerprof_slice_cli_validates_attribute_contract` |
| Virtual output slices from `--attribute` | covered with explicit opt-in extension | `test_clankerprof_slice_cli_validates_attribute_contract` |
| Descendant filters such as `<name:RequestHandler#render_response`, including OR semantics across descendant filters | covered | `test_clankerprof_slice_descendant_filters_use_or_semantics` |
| Repeated `!` and `<` filter prefixes in any order | covered | `test_clankerprof_slice_filter_prefixes_can_repeat_in_any_order` |
| Bottom `slice:<name>` filters include descendant-attribute rescue behavior | covered | `test_clankerprof_slice_filter_honors_descendant_attribute_slice_matches` |
| `slice:<name>` collapse does not use descendant-attribute rescue | covered | `test_clankerprof_slice_collapse_does_not_use_descendant_attribute_rescue` |
| Non-descendant filters apply to the selected bottom attribution frame after native/collapse handling | covered | `test_clankerprof_slice_filters_apply_to_bottom_frame_after_native_collapse` |
| Unsupported filter keys, malformed filters, and collapse prefixes are rejected | covered | `test_clankerprof_slice_cli_validates_filter_and_collapse_contract` |
| Duplicate scalar config and CLI options are rejected instead of silently merged | covered | `test_clankerprof_slice_cli_rejects_duplicate_scalar_config_options` |
| `--by-slice=N`, `--by-slice=N%`, and bare `--by-slice` | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice` |
| YAML config for common slice options | covered | `test_clankerprof_slice_cli_supports_config_and_output_limits` |
| TOML config with profile path, `filter`, and scalar options | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice` |
| Default `./slices.yml` discovery when slice-aware options are used | covered | `test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice` |
| GC pseudo-output for `(marking)` and `(sweeping)` | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Uncollapsible pseudo-output when all eligible frames are collapsed, using root eligible frame reporting | covered | `test_clankerprof_uncollapsible_reports_root_eligible_frame` |
| Unattributed library summaries for default slice views, with legacy gem output retained | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Slice frame output includes decoded pprof line numbers when present | covered | `test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries` |
| Terminal text formatting, coloring, width wrapping, and timing output from older slice tools | not claimed | `clankerprof` prioritizes machine-readable JSON plus CSV/text target reports. |
| Domain-specific responsibility fields as first-class concepts | not claimed | Slice metadata is preserved generically instead of hardcoding any one responsibility system. |

## Compare

| Capability | Status | Test coverage |
| --- | --- | --- |
| Structured slice comparison with absolute/relative thresholds | covered | `test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare` |
| Non-zero regression gate exit code | covered | `test_clankerprof_compare_exits_nonzero_for_regression_gate` |
| Top per-function regressions and improvements in JSON | covered | `test_clankerprof_compare_exits_nonzero_for_regression_gate` |
| Text compare report wording from older slice tools | not claimed | JSON gate compatibility is the stable contract. |

## Remaining confidence boundary

The self-contained tests now cover the mined target-attribution surface plus
the separate slice/report surface listed above. They do not prove byte-for-byte
replacement of every historical profile report. Before deleting older tools,
run golden replays against real Ruby `.pb` profiles and compare category/caller
CSV totals first, then compare slice totals as a separate migration check.
