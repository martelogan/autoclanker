use crate::model::{Frame, ProfileFacts, TimeNs};
use crate::targets::{extract_library_name, match_path_pattern, RuntimeRuleSet};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const GC_PSEUDO_SLICE: &str = "(gc)";
pub const UNCOLLAPSIBLE_PSEUDO_SLICE: &str = "(uncollapsible)";
/// The implicit fallback row for unmatched samples: a configured slice under
/// this name would silently merge with the fallback, absorbing samples its
/// paths never matched under the configured metadata.
pub const ALL_PSEUDO_SLICE: &str = "(all)";
pub const RESERVED_SLICE_NAMES: [&str; 3] = [
    ALL_PSEUDO_SLICE,
    GC_PSEUDO_SLICE,
    UNCOLLAPSIBLE_PSEUDO_SLICE,
];
pub const RESERVED_SLICE_NAMES_MESSAGE: &str =
    "The names (all), (gc) and (uncollapsible) are reserved for analyzer pseudo-outputs.";

#[derive(Debug, Clone, PartialEq)]
pub struct SliceDefinition {
    pub name: String,
    pub path_patterns: Vec<String>,
    pub is_default: bool,
    pub metadata: BTreeMap<String, Value>,
}

impl SliceDefinition {
    pub fn matches_frame(&self, frame: &Frame, rules: &RuntimeRuleSet) -> bool {
        self.path_patterns
            .iter()
            .any(|pattern| match_path_pattern(pattern, &frame.filename, rules))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributionRule {
    pub key: String,
    pub value: String,
    pub target_slice: String,
    pub descendant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceAnalysisOptions {
    pub slices: Vec<SliceDefinition>,
    pub filters: Vec<String>,
    pub collapse: Vec<String>,
    pub attributes: Vec<AttributionRule>,
    /// Signed to honor Python's slice semantics: a negative limit drops
    /// entries from the tail (`list[:-n]`) instead of keeping the head.
    pub top: Option<i64>,
    pub by_slice: Option<String>,
    pub show_paths: bool,
    pub no_collapse_native: bool,
    /// Signed for the same Python slice-semantics reason as `top`.
    pub unattributed_libraries: Option<i64>,
    pub runtime_rules: RuntimeRuleSet,
}

impl Default for SliceAnalysisOptions {
    fn default() -> Self {
        Self {
            slices: Vec::new(),
            filters: Vec::new(),
            collapse: Vec::new(),
            attributes: Vec::new(),
            top: None,
            by_slice: None,
            show_paths: false,
            no_collapse_native: false,
            unattributed_libraries: None,
            runtime_rules: RuntimeRuleSet::generic().clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceFrameStats {
    pub function: String,
    pub filename: String,
    pub line: Option<i64>,
    pub time_ns: TimeNs,
}

// Ranked frame/library arrays break ties by first-seen order, matching the
// Python reference; the accumulation maps must preserve insertion order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceStats {
    pub name: String,
    pub time_ns: TimeNs,
    // Tuple identity: a "name\0filename" string key merged distinct frames
    // whose symbols contained the delimiter.
    pub frames: IndexMap<(String, String), SliceFrameStats>,
    pub unattributed_libraries: IndexMap<String, TimeNs>,
    pub is_default: bool,
}

impl SliceStats {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            time_ns: 0,
            frames: IndexMap::new(),
            unattributed_libraries: IndexMap::new(),
            is_default: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceAnalysisResult {
    pub matching_time_ns: TimeNs,
    pub total_time_ns: TimeNs,
    pub slices: Vec<SliceStats>,
    pub gc_time_ns: TimeNs,
    pub uncollapsible: Option<SliceStats>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BottomFrameSelection<'a> {
    bottom: &'a Frame,
    root_eligible: Option<&'a Frame>,
    bottom_is_collapsed: bool,
}

pub fn load_slices_file(path: impl AsRef<Path>) -> Result<Vec<SliceDefinition>, String> {
    let payload = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_yaml::Value =
        serde_yaml::from_str(&payload).map_err(|error| error.to_string())?;
    crate::rules::require_string_keys(&value)?;
    let Some(raw_slices) = value.get("slices").and_then(serde_yaml::Value::as_sequence) else {
        return Err("Slices file must contain a slices array.".to_string());
    };
    let mut slices = Vec::new();
    for item in raw_slices {
        let Some(mapping) = item.as_mapping() else {
            return Err("Each slice entry must be an object.".to_string());
        };
        // Validation order mirrors Python `_load_slices`: paths shape, name
        // presence, name type, then paths entry types.
        let raw_paths = match mapping.get(serde_yaml::Value::String("paths".to_string())) {
            None | Some(serde_yaml::Value::Null) => None,
            Some(value) => {
                let Some(items) = value.as_sequence() else {
                    return Err("Slice paths must be an array.".to_string());
                };
                Some(items)
            }
        };
        let Some(raw_name) = mapping.get(serde_yaml::Value::String("name".to_string())) else {
            return Err("Each slice entry must include a name.".to_string());
        };
        let Some(name) = raw_name.as_str() else {
            return Err("Slice name must be a string.".to_string());
        };
        let mut paths: Vec<String> = Vec::new();
        for item in raw_paths.into_iter().flatten() {
            let Some(text) = item.as_str() else {
                return Err("Slice paths values must be strings.".to_string());
            };
            paths.push(text.to_string());
        }
        // Only YAML booleans (absent/null read as false): as_bool coercion
        // silently treated `default: 1` as non-default where Python's
        // truthiness made it the default slice.
        let is_default = match mapping.get(serde_yaml::Value::String("default".to_string())) {
            None | Some(serde_yaml::Value::Null) => false,
            Some(serde_yaml::Value::Bool(value)) => *value,
            Some(_) => return Err("Slice default must be a boolean.".to_string()),
        };
        let mut metadata: BTreeMap<String, Value> = BTreeMap::new();
        for (raw_key, raw_value) in mapping {
            let Some(key) = raw_key.as_str() else {
                continue;
            };
            if matches!(key, "name" | "paths" | "default") {
                continue;
            }
            let converted = metadata_value(raw_value)?;
            if key == "metadata" {
                if let Value::Object(nested) = converted {
                    for (nested_key, nested_value) in nested {
                        metadata.insert(nested_key, nested_value);
                    }
                    continue;
                }
                metadata.insert(key.to_string(), converted);
                continue;
            }
            metadata.insert(key.to_string(), converted);
        }
        slices.push(SliceDefinition {
            name: name.to_string(),
            path_patterns: paths,
            is_default,
            metadata,
        });
    }
    Ok(slices)
}

/// Shared definition validation for every slice-loading surface.
///
/// Scope configs load slice files without ever running slice analysis, so
/// the checks must not live only inside `analyze_slice_facts`.
pub fn validate_slice_definitions(slices: &[SliceDefinition]) -> Result<(), String> {
    let default_names: Vec<&str> = slices
        .iter()
        .filter(|slice| slice.is_default)
        .map(|slice| slice.name.as_str())
        .collect();
    if default_names.len() > 1 {
        return Err(format!(
            "Slice config declares multiple default slices: {}. Exactly one slice may set default.",
            default_names.join(", ")
        ));
    }
    let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for definition in slices {
        if !seen_names.insert(definition.name.as_str()) {
            return Err(format!(
                "Slice config declares duplicate slice name: {}. Each slice name may be defined once.",
                definition.name
            ));
        }
        // A user slice under (gc)/(uncollapsible) would be attributed and
        // then unconditionally stripped at render; one under (all) would
        // silently merge with the implicit fallback row (R9-04 reversed the
        // earlier judgment that the merge was benign).
        if RESERVED_SLICE_NAMES.contains(&definition.name.as_str()) {
            return Err(format!(
                "Slice config declares reserved pseudo-slice name: {}. {}",
                definition.name, RESERVED_SLICE_NAMES_MESSAGE
            ));
        }
        // The compare focus grammar splits values on ','; a comma-bearing
        // row name would make focusing it silently gate the split parts.
        if definition.name.contains(',') {
            return Err(format!(
                "Slice config declares comma-bearing slice name: {}. Slice names must not contain ','.",
                definition.name
            ));
        }
    }
    Ok(())
}

pub fn analyze_slice_facts(
    facts: &ProfileFacts,
    options: &SliceAnalysisOptions,
) -> Result<SliceAnalysisResult, String> {
    // Definition validation lives in the analysis layer, mirroring Python
    // `analyze_slice_facts`, so file, config, and programmatic paths all
    // share it with identical error ordering.
    validate_slice_definitions(&options.slices)?;
    let default_names: Vec<&str> = options
        .slices
        .iter()
        .filter(|slice| slice.is_default)
        .map(|slice| slice.name.as_str())
        .collect();
    let mut total_time = 0;
    let mut matching_time = 0;
    let mut gc_time = 0;
    let mut stats_by_slice: IndexMap<String, SliceStats> = IndexMap::new();
    let mut uncollapsible_stats = SliceStats::new(UNCOLLAPSIBLE_PSEUDO_SLICE);
    let default_slice = default_names
        .first()
        .map(|name| name.to_string())
        .unwrap_or_else(|| "(all)".to_string());

    for fact in &facts.samples {
        let value = fact.primary_value();
        total_time += value;
        let stack = fact.stack.as_slice();
        let Some(leaf) = stack.first() else {
            continue;
        };
        let Some(selection) = select_bottom_frame(stack, options) else {
            continue;
        };
        let bottom = selection.bottom;
        if !filters_match_sample(&options.filters, stack, options, bottom) {
            continue;
        }
        matching_time += value;

        if is_gc_function(&leaf.name) {
            gc_time += value;
            let gc_stats = stats_by_slice
                .entry(GC_PSEUDO_SLICE.to_string())
                .or_insert_with(|| SliceStats::new(GC_PSEUDO_SLICE));
            add_frame_stats(gc_stats, leaf, value);
            gc_stats.time_ns += value;
            continue;
        }

        let slice_name = slice_for_frame(bottom, stack, options, true);
        let slice_stats = stats_by_slice
            .entry(slice_name.clone())
            .or_insert_with(|| SliceStats {
                is_default: slice_name == default_slice,
                ..SliceStats::new(slice_name.clone())
            });
        slice_stats.time_ns += value;
        add_frame_stats(slice_stats, bottom, value);
        if slice_name == default_slice {
            if let Some(library_name) =
                extract_library_name(&bottom.filename, &options.runtime_rules, None)
            {
                *slice_stats
                    .unattributed_libraries
                    .entry(library_name)
                    .or_default() += value;
            }
        }
        if selection.bottom_is_collapsed {
            uncollapsible_stats.time_ns += value;
            add_frame_stats(
                &mut uncollapsible_stats,
                selection.root_eligible.unwrap_or(bottom),
                value,
            );
        }
    }

    let mut slices: Vec<_> = stats_by_slice.into_values().collect();
    slices.sort_by(|left, right| right.time_ns.cmp(&left.time_ns));
    Ok(SliceAnalysisResult {
        matching_time_ns: matching_time,
        total_time_ns: total_time,
        slices,
        gc_time_ns: gc_time,
        uncollapsible: (uncollapsible_stats.time_ns != 0).then_some(uncollapsible_stats),
    })
}

pub fn render_slice_json(
    result: &SliceAnalysisResult,
    options: &SliceAnalysisOptions,
) -> Result<Value, String> {
    // Percentages are shares of the filtered matching total; when matched
    // signed samples cancel to exactly zero the zero arms below render 0 —
    // never a silent fallback to the unrelated whole-profile total.
    let total = result.matching_time_ns;
    let mut selected_slices: Vec<_> = result
        .slices
        .iter()
        .filter(|slice| slice.name != GC_PSEUDO_SLICE && slice.name != UNCOLLAPSIBLE_PSEUDO_SLICE)
        .collect();
    if let Some(by_slice) = &options.by_slice {
        if let Some(raw_threshold) = by_slice.strip_suffix('%') {
            let threshold = parse_by_slice_threshold(raw_threshold)?;
            // At a zero matching total every rendered percentage is 0, so
            // threshold selection compares against 0 rather than deleting
            // every signed row through a nonzero-total short-circuit.
            selected_slices.retain(|slice| {
                let pct = if total != 0 {
                    slice.time_ns as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                pct >= threshold
            });
        } else {
            let limit = by_slice
                .parse::<i64>()
                .map_err(|_| "--by-slice values must be integers.".to_string())?;
            apply_python_limit(&mut selected_slices, Some(limit));
        }
    }
    let payload = json!({
        "slices": selected_slices.into_iter().map(|slice| slice_payload(slice, total, options)).collect::<Vec<_>>(),
        "summary": {
            // Python's `... if total else 0` yields an int zero here, so the
            // empty-profile summary must serialize as 0, not 0.0.
            "matching_pct": if result.total_time_ns == 0 { json!(0) } else { json!(result.matching_time_ns as f64 / result.total_time_ns as f64 * 100.0) },
            "matching_time_ns": result.matching_time_ns,
            "total_time_ns": result.total_time_ns,
        },
        "tool": "clankerprof_slices",
    });
    Ok(add_optional_slice_payloads(payload, result, total, options))
}

fn parse_by_slice_threshold(raw: &str) -> Result<f64, String> {
    const MESSAGE: &str = "--by-slice percentage thresholds must be finite numbers.";
    let threshold = raw.parse::<f64>().map_err(|_| MESSAGE.to_string())?;
    if !threshold.is_finite() {
        return Err(MESSAGE.to_string());
    }
    Ok(threshold)
}

/// Python list-slice truncation: a non-negative limit keeps the head
/// (`list[:n]`), a negative limit drops that many entries from the tail
/// (`list[:-n]`).
pub(crate) fn apply_python_limit<T>(items: &mut Vec<T>, limit: Option<i64>) {
    let Some(limit) = limit else {
        return;
    };
    if limit >= 0 {
        items.truncate(usize::try_from(limit).unwrap_or(usize::MAX));
    } else {
        // `-i64::MIN` overflows; unsigned_abs is defined for the full range.
        let dropped = usize::try_from(limit.unsigned_abs()).unwrap_or(usize::MAX);
        items.truncate(items.len().saturating_sub(dropped));
    }
}

fn add_optional_slice_payloads(
    mut payload: Value,
    result: &SliceAnalysisResult,
    total: TimeNs,
    options: &SliceAnalysisOptions,
) -> Value {
    let Some(object) = payload.as_object_mut() else {
        return payload;
    };
    // Zero-aggregate pseudo-outputs stay omitted; negative aggregates are
    // signed data and must be reported (R2-07 rule, extended by R3-05).
    if result.gc_time_ns != 0 {
        object.insert(
            "gc".to_string(),
            json!({
                "pct": if total == 0 { json!(0) } else { json!(result.gc_time_ns as f64 / total as f64 * 100.0) },
                "time_ns": result.gc_time_ns,
            }),
        );
    }
    if let Some(uncollapsible) = &result.uncollapsible {
        object.insert(
            "uncollapsible".to_string(),
            json!({
                "frames": rendered_frames(uncollapsible, total, options.top),
                "name": uncollapsible.name,
                "pct": if total == 0 { json!(0) } else { json!(uncollapsible.time_ns as f64 / total as f64 * 100.0) },
                "time_ns": uncollapsible.time_ns,
                "unattributed_gems": [],
                "unattributed_libraries": [],
            }),
        );
    }
    payload
}

fn slice_payload(slice: &SliceStats, total: TimeNs, options: &SliceAnalysisOptions) -> Value {
    let library_limit = options.unattributed_libraries;
    let mut payload = json!({
        "frames": rendered_frames(slice, total, options.top),
        "is_default": slice.is_default,
        "name": slice.name,
        "pct": if total == 0 { json!(0) } else { json!(slice.time_ns as f64 / total as f64 * 100.0) },
        "time_ns": slice.time_ns,
        "unattributed_gems": rendered_libraries(slice, total, library_limit),
        "unattributed_libraries": rendered_libraries(slice, total, library_limit),
    });
    let metadata = options
        .slices
        .iter()
        .find(|definition| definition.name == slice.name && !definition.metadata.is_empty())
        .map(|definition| definition.metadata.clone());
    if let (Some(metadata), Some(object)) = (metadata, payload.as_object_mut()) {
        object.insert(
            "metadata".to_string(),
            Value::Object(metadata.into_iter().collect()),
        );
    }
    payload
}

fn rendered_frames(slice: &SliceStats, total: TimeNs, top: Option<i64>) -> Vec<Value> {
    let mut frames: Vec<_> = slice.frames.values().collect();
    frames.sort_by(|left, right| right.time_ns.cmp(&left.time_ns));
    apply_python_limit(&mut frames, top);
    frames
        .into_iter()
        .map(|frame| {
            json!({
                "filename": frame.filename,
                "function": frame.function,
                "line": frame.line,
                "pct": if total == 0 { json!(0) } else { json!(frame.time_ns as f64 / total as f64 * 100.0) },
                "time_ns": frame.time_ns,
            })
        })
        .collect()
}

fn rendered_libraries(slice: &SliceStats, total: TimeNs, limit: Option<i64>) -> Vec<Value> {
    let mut libraries: Vec<_> = slice.unattributed_libraries.iter().collect();
    libraries.sort_by(|left, right| right.1.cmp(left.1));
    apply_python_limit(&mut libraries, limit);
    libraries
        .into_iter()
        .map(|(name, time_ns)| {
            json!({
                "name": name,
                "pct": if total == 0 { json!(0) } else { json!(*time_ns as f64 / total as f64 * 100.0) },
                "time_ns": time_ns,
            })
        })
        .collect()
}

fn select_bottom_frame<'a>(
    stack: &'a [Frame],
    options: &SliceAnalysisOptions,
) -> Option<BottomFrameSelection<'a>> {
    let mut bottom = stack.first()?;
    let mut first_eligible = None;
    let mut found_uncollapsed = false;
    for frame in stack {
        if !is_eligible(frame, options) {
            continue;
        }
        if first_eligible.is_none() {
            first_eligible = Some(frame);
        }
        if is_collapsed_frame(frame, stack, options) {
            continue;
        }
        bottom = frame;
        found_uncollapsed = true;
        break;
    }
    if let Some(first_eligible) = first_eligible {
        if !is_eligible(bottom, options) {
            bottom = first_eligible;
        }
    }
    if first_eligible.is_none() || found_uncollapsed {
        return Some(BottomFrameSelection {
            bottom,
            root_eligible: None,
            bottom_is_collapsed: false,
        });
    }
    let root_eligible = stack.iter().rev().find(|frame| is_eligible(frame, options));
    Some(BottomFrameSelection {
        bottom: first_eligible?,
        root_eligible,
        bottom_is_collapsed: is_collapsed_frame(first_eligible?, stack, options),
    })
}

fn add_frame_stats(slice: &mut SliceStats, frame: &Frame, value: TimeNs) {
    let frame_key = (frame.name.clone(), frame.filename.clone());
    let frame_stats = slice
        .frames
        .entry(frame_key)
        .or_insert_with(|| SliceFrameStats {
            function: frame.name.clone(),
            filename: frame.filename.clone(),
            line: (frame.line != 0).then_some(frame.line),
            time_ns: 0,
        });
    frame_stats.time_ns += value;
}

fn is_eligible(frame: &Frame, options: &SliceAnalysisOptions) -> bool {
    options.no_collapse_native || !is_native_path(&frame.filename, &options.runtime_rules)
}

fn is_collapsed_frame(frame: &Frame, stack: &[Frame], options: &SliceAnalysisOptions) -> bool {
    options
        .collapse
        .iter()
        .any(|collapse_filter| collapse_matches_frame(collapse_filter, frame, stack, options))
}

fn collapse_matches_frame(
    raw_filter: &str,
    frame: &Frame,
    stack: &[Frame],
    options: &SliceAnalysisOptions,
) -> bool {
    let (key, value) = raw_filter.split_once(':').unwrap_or(("", ""));
    if key == "slice" {
        return slice_for_frame(frame, stack, options, false) == value;
    }
    matches_frame_filter(frame, raw_filter, &options.runtime_rules)
}

/// Slice metadata mirrors Python's `_json_compatible`: preserved as generic
/// JSON, except non-finite numbers, which serde_json would silently null —
/// neither implementation can "preserve" those, so both fail closed.
fn metadata_value(raw: &serde_yaml::Value) -> Result<Value, String> {
    match raw {
        serde_yaml::Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                if !value.is_finite() {
                    return Err(
                        "Slice metadata values must be finite JSON-compatible numbers.".to_string(),
                    );
                }
            }
            serde_json::to_value(raw).map_err(|error| error.to_string())
        }
        serde_yaml::Value::Sequence(items) => Ok(Value::Array(
            items.iter().map(metadata_value).collect::<Result<_, _>>()?,
        )),
        serde_yaml::Value::Mapping(mapping) => {
            let mut object = serde_json::Map::new();
            for (key, value) in mapping {
                let Some(key) = key.as_str() else {
                    // Strict YAML loading already rejects non-string keys.
                    return Err("YAML mapping keys must be strings.".to_string());
                };
                object.insert(key.to_string(), metadata_value(value)?);
            }
            Ok(Value::Object(object))
        }
        serde_yaml::Value::Tagged(_) => {
            // The strict walk rejects local tags before metadata conversion;
            // kept fail-closed here so this path can never silently wrap a
            // tagged value the way serde_json's serialization would.
            Err(crate::rules::YAML_LOCAL_TAG_MESSAGE.to_string())
        }
        _ => serde_json::to_value(raw).map_err(|error| error.to_string()),
    }
}

fn filters_match_sample(
    filters: &[String],
    stack: &[Frame],
    options: &SliceAnalysisOptions,
    bottom: &Frame,
) -> bool {
    let mut bottom_filters = Vec::new();
    let mut descendant_filters = Vec::new();
    for raw_filter in filters {
        let (_, descendant, _) = parse_filter_prefixes(raw_filter);
        if descendant {
            descendant_filters.push(raw_filter);
        } else {
            bottom_filters.push(raw_filter);
        }
    }
    bottom_filters
        .iter()
        .all(|raw_filter| filter_matches_stack(raw_filter, stack, options, bottom))
        && (descendant_filters.is_empty()
            || descendant_filters
                .iter()
                .any(|raw_filter| filter_matches_stack(raw_filter, stack, options, bottom)))
}

fn filter_matches_stack(
    raw_filter: &str,
    stack: &[Frame],
    options: &SliceAnalysisOptions,
    bottom: &Frame,
) -> bool {
    let (inverted, descendant, body) = parse_filter_prefixes(raw_filter);
    let (key, value) = body.split_once(':').unwrap_or(("", ""));
    if key == "slice" && !descendant {
        // Bottom slice: filters evaluate the sample's EFFECTIVE attribution
        // in both polarities: a sample rescued into a slice by a descendant
        // attribute rule matches slice:<name> and is excluded by
        // !slice:<name>. The rescue resolves the first-match WINNING rule's
        // target — the mere existence of a losing rule targeting the slice
        // never matches. (Collapse slice: rules deliberately skip the
        // rescue — the documented carve-out.)
        let mut matched = slice_for_frame(bottom, stack, options, false) == value;
        if !matched {
            matched = slice_for_frame(bottom, stack, options, true) == value;
        }
        return if inverted { !matched } else { matched };
    }
    let frames: Vec<&Frame> = if descendant {
        stack.iter().collect()
    } else {
        vec![bottom]
    };
    let matches = frames.into_iter().map(|frame| {
        if key == "slice" {
            slice_for_frame(frame, stack, options, false) == value
        } else {
            matches_frame_filter(frame, &body, &options.runtime_rules)
        }
    });
    // Negation binds to descendant EXISTENCE: a stack containing the
    // forbidden frame must not pass just because some other frame fails to
    // match. (Bottom filters are single-frame, where the formulas coincide.)
    let matched = matches.into_iter().any(|matched| matched);
    if inverted {
        !matched
    } else {
        matched
    }
}

fn parse_filter_prefixes(raw_filter: &str) -> (bool, bool, String) {
    let mut inverted = false;
    let mut descendant = false;
    let mut body = raw_filter;
    loop {
        if let Some(rest) = body.strip_prefix('!') {
            inverted = true;
            body = rest;
            continue;
        }
        if let Some(rest) = body.strip_prefix('<') {
            descendant = true;
            body = rest;
            continue;
        }
        break;
    }
    (inverted, descendant, body.to_string())
}

fn matches_frame_filter(frame: &Frame, raw_filter: &str, rules: &RuntimeRuleSet) -> bool {
    let (key, value) = raw_filter.split_once(':').unwrap_or(("", ""));
    match key {
        "name" => frame.name.contains(value),
        "path" => frame.filename.contains(value),
        "dependency" | "gem" | "library" | "package" | "vendor" => {
            let library_name = extract_library_name(&frame.filename, rules, Some(key));
            if value == "*" {
                library_name.is_some()
            } else {
                library_name.as_deref() == Some(value)
            }
        }
        _ if rules.library_selector_path_patterns.contains_key(key) => {
            let library_name = extract_library_name(&frame.filename, rules, Some(key));
            if value == "*" {
                library_name.is_some()
            } else {
                library_name.as_deref() == Some(value)
            }
        }
        _ => false,
    }
}

fn slice_for_frame(
    frame: &Frame,
    stack: &[Frame],
    options: &SliceAnalysisOptions,
    include_descendant_attributes: bool,
) -> String {
    for rule in &options.attributes {
        if rule.descendant && !include_descendant_attributes {
            continue;
        }
        let frames: Vec<&Frame> = if rule.descendant {
            stack.iter().collect()
        } else {
            vec![frame]
        };
        if frames.into_iter().any(|candidate| {
            matches_frame_filter(
                candidate,
                &format!("{}:{}", rule.key, rule.value),
                &options.runtime_rules,
            )
        }) {
            return rule.target_slice.clone();
        }
    }
    let mut default = "(all)";
    for slice in &options.slices {
        if slice.is_default {
            default = &slice.name;
            continue;
        }
        if slice
            .path_patterns
            .iter()
            .any(|pattern| match_path_pattern(pattern, &frame.filename, &options.runtime_rules))
        {
            return slice.name.clone();
        }
    }
    default.to_string()
}

use crate::categorize::is_native_path;

fn is_gc_function(name: &str) -> bool {
    name == "(marking)" || name == "(sweeping)"
}

#[cfg(test)]
mod limit_tests {
    use super::{
        analyze_slice_facts, apply_python_limit, filter_matches_stack, metadata_value,
        parse_by_slice_threshold, validate_slice_definitions, AttributionRule, Frame, ProfileFacts,
        SliceAnalysisOptions, SliceDefinition, RESERVED_SLICE_NAMES,
    };
    use std::collections::BTreeMap;

    fn test_frame(name: &str, filename: &str) -> Frame {
        Frame {
            location_id: 0,
            function_id: 0,
            name: name.to_string(),
            filename: filename.to_string(),
            line: 0,
            location_is_folded: false,
        }
    }

    #[test]
    fn bottom_slice_filters_honor_descendant_attribute_rescue() {
        // Mirrors the Python reference (R5-01): a sample rescued into a
        // slice by a descendant attribute rule matches slice:<name> and is
        // excluded by !slice:<name>, in both polarities.
        let stack = vec![
            test_frame(
                "CacheClient#get",
                "/vendor/cache-client-1.2.3/lib/client.rb",
            ),
            test_frame("TelemetryWrapper#call", "/app/lib/telemetry.rb"),
            test_frame("ComponentRenderer#render", "/app/components/card.rb"),
            test_frame("RequestHandler#render_response", "/app/http/request.rb"),
        ];
        let options = SliceAnalysisOptions {
            slices: vec![SliceDefinition {
                name: "instrumentation".to_string(),
                path_patterns: Vec::new(),
                is_default: false,
                metadata: BTreeMap::new(),
            }],
            attributes: vec![AttributionRule {
                key: "name".to_string(),
                value: "TelemetryWrapper#call".to_string(),
                target_slice: "instrumentation".to_string(),
                descendant: true,
            }],
            ..SliceAnalysisOptions::default()
        };
        let bottom = &stack[0];
        assert!(filter_matches_stack(
            "slice:instrumentation",
            &stack,
            &options,
            bottom
        ));
        assert!(!filter_matches_stack(
            "!slice:instrumentation",
            &stack,
            &options,
            bottom
        ));
        // Without the rescue rule both polarities flip back.
        let mut no_rescue = options.clone();
        no_rescue.attributes.clear();
        assert!(!filter_matches_stack(
            "slice:instrumentation",
            &stack,
            &no_rescue,
            bottom
        ));
        assert!(filter_matches_stack(
            "!slice:instrumentation",
            &stack,
            &no_rescue,
            bottom
        ));
    }

    #[test]
    fn python_limit_keeps_head_for_non_negative_and_drops_tail_for_negative() {
        let mut items = vec![1, 2, 3, 4];
        apply_python_limit(&mut items, Some(2));
        assert_eq!(items, vec![1, 2]);
        let mut items = vec![1, 2, 3, 4];
        apply_python_limit(&mut items, Some(-1));
        assert_eq!(items, vec![1, 2, 3]);
        let mut items = vec![1, 2];
        apply_python_limit(&mut items, Some(-5));
        assert!(items.is_empty());
        let mut items = vec![1, 2];
        apply_python_limit(&mut items, None);
        assert_eq!(items, vec![1, 2]);
    }

    #[test]
    fn python_limit_handles_i64_min_without_overflow() {
        let mut items = vec![1, 2, 3];
        apply_python_limit(&mut items, Some(i64::MIN));
        assert!(items.is_empty());
        let mut items: Vec<i32> = Vec::new();
        apply_python_limit(&mut items, Some(i64::MIN));
        assert!(items.is_empty());
    }

    #[test]
    fn by_slice_thresholds_must_be_finite() {
        assert_eq!(parse_by_slice_threshold("0.5"), Ok(0.5));
        for raw in ["garbage", "nan", "inf", "-inf"] {
            assert_eq!(
                parse_by_slice_threshold(raw),
                Err("--by-slice percentage thresholds must be finite numbers.".to_string()),
                "{raw}"
            );
        }
    }

    #[test]
    fn metadata_values_reject_non_finite_numbers() {
        let finite: serde_yaml::Value = serde_yaml::from_str("{score: 1.5, tags: [a, 2]}").unwrap();
        assert_eq!(
            metadata_value(&finite).unwrap(),
            serde_json::json!({"score": 1.5, "tags": ["a", 2]})
        );
        for raw in [
            "{score: .nan}",
            "{score: .inf}",
            "[ok, .nan]",
            "{a: {b: [.inf]}}",
        ] {
            let value: serde_yaml::Value = serde_yaml::from_str(raw).unwrap();
            assert_eq!(
                metadata_value(&value),
                Err("Slice metadata values must be finite JSON-compatible numbers.".to_string()),
                "{raw}"
            );
        }
    }

    #[test]
    fn analyze_rejects_duplicate_slice_names() {
        let facts = ProfileFacts {
            samples: Vec::new(),
            total_primary_value: 0,
            empty_sample_count: 0,
            value_types: Vec::new(),
            period_type: None,
            period: 0,
            default_sample_type: String::new(),
            primary_value_index: 0,
        };
        let options = SliceAnalysisOptions {
            slices: vec![
                SliceDefinition {
                    name: "App".to_string(),
                    path_patterns: vec!["/app".to_string()],
                    is_default: false,
                    metadata: BTreeMap::new(),
                },
                SliceDefinition {
                    name: "App".to_string(),
                    path_patterns: vec!["/other".to_string()],
                    is_default: false,
                    metadata: BTreeMap::new(),
                },
            ],
            ..SliceAnalysisOptions::default()
        };
        assert_eq!(
            analyze_slice_facts(&facts, &options).unwrap_err(),
            "Slice config declares duplicate slice name: App. \
             Each slice name may be defined once."
        );
    }

    #[test]
    fn analyze_rejects_reserved_pseudo_slice_names() {
        let facts = ProfileFacts {
            samples: Vec::new(),
            total_primary_value: 0,
            empty_sample_count: 0,
            value_types: Vec::new(),
            period_type: None,
            period: 0,
            default_sample_type: String::new(),
            primary_value_index: 0,
        };
        for reserved in RESERVED_SLICE_NAMES {
            let options = SliceAnalysisOptions {
                slices: vec![SliceDefinition {
                    name: reserved.to_string(),
                    path_patterns: vec!["/app".to_string()],
                    is_default: false,
                    metadata: BTreeMap::new(),
                }],
                ..SliceAnalysisOptions::default()
            };
            assert_eq!(
                analyze_slice_facts(&facts, &options).unwrap_err(),
                format!(
                    "Slice config declares reserved pseudo-slice name: {reserved}. \
                     The names (all), (gc) and (uncollapsible) are reserved for analyzer pseudo-outputs."
                )
            );
        }
    }

    #[test]
    fn validate_rejects_comma_bearing_slice_names() {
        // The compare focus grammar splits values on ','.
        let definitions = vec![SliceDefinition {
            name: "A,B".to_string(),
            path_patterns: vec!["/srv".to_string()],
            is_default: false,
            metadata: BTreeMap::new(),
        }];
        assert_eq!(
            validate_slice_definitions(&definitions).unwrap_err(),
            "Slice config declares comma-bearing slice name: A,B. \
             Slice names must not contain ','."
        );
    }
}
