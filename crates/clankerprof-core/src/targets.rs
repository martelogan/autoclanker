use crate::categorize::{categorize_stack, RuntimeCategoryCache};
use crate::facts::{sample_facts_to_json_value, SAMPLE_FACTS_SCHEMA_VERSION};
use crate::model::{CategoryStats, Frame, FunctionMetrics, ProfileFacts, TimeNs};
use indexmap::IndexMap;
use regex::Regex;
use serde_json::value::RawValue;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Mutex, OnceLock};

// Category precedence is first-match-wins in config order, and ranked arrays
// break ties by first-seen order, matching the Python reference; both need
// insertion-ordered maps, never BTreeMap.
pub type TargetConfig = IndexMap<String, IndexMap<String, String>>;
pub type TargetResults = BTreeMap<String, IndexMap<String, CategoryStats>>;

pub use crate::rules::{LibraryPattern, RuntimeRuleSet};

pub const DEFAULT_LIBRARY_SELECTORS: &[&str] =
    &["dependency", "gem", "library", "package", "vendor"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryPath {
    pub name: String,
    pub relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetAnalysisOptions {
    pub runtime_rules: RuntimeRuleSet,
    pub enhanced_runtime_categorization: bool,
    pub fold_runtime_internals: bool,
    pub track_semantic_callers: bool,
    pub caller_fallback_when_uncategorized: bool,
}

impl Default for TargetAnalysisOptions {
    fn default() -> Self {
        Self {
            runtime_rules: RuntimeRuleSet::generic().clone(),
            enhanced_runtime_categorization: true,
            fold_runtime_internals: false,
            track_semantic_callers: false,
            caller_fallback_when_uncategorized: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PatternMode {
    Auto,
    Library,
    Path,
    Regex,
}

pub fn parse_target_config_json(payload: &str) -> Result<TargetConfig, String> {
    let parents: IndexMap<String, Box<RawValue>> = serde_json::from_str(payload)
        .map_err(|_| "Target config must be a JSON object.".to_string())?;
    let mut result = TargetConfig::new();
    for (parent, raw_categories) in parents {
        let categories: IndexMap<String, Value> = serde_json::from_str(raw_categories.get())
            .map_err(|_| format!("Target config for {parent} must be an object."))?;
        let mut category_map = IndexMap::new();
        for (category, pattern) in categories {
            let Some(pattern_text) = pattern.as_str() else {
                return Err(format!(
                    "Target config pattern for {category} must be a string."
                ));
            };
            category_map.insert(category, pattern_text.to_string());
        }
        result.insert(parent, category_map);
    }
    Ok(result)
}

pub fn analyze_target_facts(facts: &ProfileFacts, config: &TargetConfig) -> TargetResults {
    analyze_target_facts_with_options(facts, config, &TargetAnalysisOptions::default())
}

pub fn analyze_target_facts_with_options(
    facts: &ProfileFacts,
    config: &TargetConfig,
    options: &TargetAnalysisOptions,
) -> TargetResults {
    let mut results = TargetResults::new();
    let mut runtime_cache = RuntimeCategoryCache::default();
    for fact in facts.samples.iter().filter(|fact| !fact.is_empty()) {
        let value = fact.primary_value();
        let stack = fact.stack.as_slice();
        let Some(leaf) = stack.first() else {
            continue;
        };
        let target_frames = stack
            .iter()
            .filter(|frame| config.contains_key(&frame.name));
        let mut seen_targets: BTreeSet<&str> = BTreeSet::new();
        for target_frame in target_frames {
            if !seen_targets.insert(target_frame.name.as_str()) {
                continue;
            }
            let parent_config = config
                .get(&target_frame.name)
                .expect("target frame from config");
            let mut configured_category_for = |frame: &Frame| {
                parent_config
                    .iter()
                    .find(|(_, pattern)| {
                        match_category_pattern(pattern, &frame.filename, &options.runtime_rules)
                    })
                    .map(|(category, _)| category.clone())
            };
            let outcome = categorize_stack(
                stack,
                &options.runtime_rules,
                options.enhanced_runtime_categorization,
                options.fold_runtime_internals,
                options.caller_fallback_when_uncategorized,
                &mut runtime_cache,
                &mut configured_category_for,
            );
            let frame_to_categorize = &stack[outcome.frame_index];
            let parent_results = results.entry(target_frame.name.clone()).or_default();
            let stats = parent_results.entry(outcome.category).or_default();
            stats.cpu_time += value;
            stats.sample_count += 1;
            stats.add_function(&leaf.name, value);
            stats.files.insert(frame_to_categorize.filename.clone());
            let caller = stack
                .iter()
                .skip(1)
                .take(9)
                .find(|frame| {
                    !frame.filename.starts_with('<')
                        && !is_runtime_stdlib_path(&frame.filename, &options.runtime_rules)
                })
                .or_else(|| stack.get(1));
            if let Some(caller) = caller {
                stats.add_caller_leaf_pair(&caller.name, &leaf.name, value);
            }
            if outcome.folded && outcome.folded_category.is_some() {
                *stats.folded_from.entry(leaf.name.clone()).or_insert(0) += value;
            }
            if options.track_semantic_callers && leaf.filename.starts_with('<') && stack.len() > 1 {
                stats.add_semantic_caller(&leaf.name, &stack[1]);
            }
        }
    }
    results
}

pub fn render_target_json(results: &TargetResults) -> Value {
    let mut parents = serde_json::Map::new();
    for (parent, categories) in results {
        let total: TimeNs = categories.values().map(|stats| stats.cpu_time).sum();
        let mut ordered_categories: Vec<_> = categories.iter().collect();
        ordered_categories.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
        parents.insert(
            parent.clone(),
            json!({
                "categories": ordered_categories.into_iter().map(|(category, stats)| {
                    json!({
                        "files": stats.files.iter().cloned().collect::<Vec<_>>(),
                        "folded_from": stats.folded_from,
                        "leaf_functions": render_function_metrics(&stats.functions),
                        "name": category,
                        "pct": if total == 0 { 0.0 } else { stats.cpu_time as f64 / total as f64 * 100.0 },
                        "samples": stats.sample_count,
                        "semantic_callers": stats.semantic_callers.iter().map(|(leaf, metrics)| {
                            (leaf.clone(), json!({
                                "caller_files": metrics.caller_files,
                                "caller_names": metrics.caller_names,
                                "count": metrics.count,
                            }))
                        }).collect::<serde_json::Map<_, _>>(),
                        "time_ns": stats.cpu_time,
                    })
                }).collect::<Vec<_>>(),
                "total_time_ns": total,
            }),
        );
    }
    json!({
        "parents": parents,
        "tool": "clankerprof_targets",
    })
}

pub fn render_facts_json(facts: &ProfileFacts) -> Value {
    sample_facts_to_json_value(facts)
}

pub fn assert_sample_facts_schema(payload: &Value) -> Result<(), String> {
    let schema = payload
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema != SAMPLE_FACTS_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported sample facts schema version: {schema:?}; expected {SAMPLE_FACTS_SCHEMA_VERSION:?}."
        ));
    }
    Ok(())
}

fn render_function_metrics(functions: &BTreeMap<String, FunctionMetrics>) -> Value {
    let mut ordered: Vec<_> = functions.iter().collect();
    ordered.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    let mut payload = serde_json::Map::new();
    for (name, metrics) in ordered {
        payload.insert(
            name.clone(),
            json!({
                "count": metrics.count,
                "cpu_time": metrics.cpu_time,
            }),
        );
    }
    Value::Object(payload)
}

pub fn match_category_pattern(pattern: &str, path: &str, rules: &RuntimeRuleSet) -> bool {
    let (mode, resolved_pattern, selector) = pattern_mode(pattern, rules);
    match mode {
        PatternMode::Path => match_path_pattern(&resolved_pattern, path, rules),
        PatternMode::Regex => match_regex(&resolved_pattern, path),
        PatternMode::Library => {
            match_library_selector(&resolved_pattern, path, rules, selector.as_deref())
        }
        PatternMode::Auto => {
            if match_regex(&resolved_pattern, path) {
                return true;
            }
            if looks_like_path_pattern(&resolved_pattern) {
                return match_path_pattern(&resolved_pattern, path, rules);
            }
            false
        }
    }
}

pub fn match_path_pattern(pattern: &str, path: &str, rules: &RuntimeRuleSet) -> bool {
    let (mode, resolved_pattern, selector) = pattern_mode(pattern, rules);
    if mode == PatternMode::Library {
        return match_library_selector(&resolved_pattern, path, rules, selector.as_deref());
    }
    if mode == PatternMode::Regex {
        return match_regex(&resolved_pattern, path);
    }

    let normalized = normalize_profile_path(path);
    let raw_normalized_pattern = normalize_profile_path(&resolved_pattern);
    let normalized_pattern = raw_normalized_pattern
        .strip_prefix("./")
        .unwrap_or(&raw_normalized_pattern)
        .to_string();
    if normalized_pattern.is_empty() {
        return false;
    }
    if has_glob_token(&normalized_pattern) {
        return glob_matches(&normalized_pattern, &normalized)
            || (!normalized_pattern.starts_with('*')
                && !normalized_pattern.starts_with('/')
                && glob_matches(&format!("**/{normalized_pattern}"), &normalized));
    }
    let path_segments = normalized.trim_matches('/');
    let pattern_segments = normalized_pattern.trim_matches('/');
    path_segments == pattern_segments
        || path_segments.starts_with(&format!("{pattern_segments}/"))
        || path_segments.ends_with(&format!("/{pattern_segments}"))
        || format!("/{path_segments}/").contains(&format!("/{pattern_segments}/"))
}

pub fn extract_library_name(
    path: &str,
    rules: &RuntimeRuleSet,
    selector: Option<&str>,
) -> Option<String> {
    extract_library_path(path, rules, selector).map(|library| library.name)
}

pub fn extract_library_path(
    path: &str,
    rules: &RuntimeRuleSet,
    selector: Option<&str>,
) -> Option<LibraryPath> {
    let normalized = normalize_profile_path(path);
    let selector_patterns = selector
        .and_then(|name| rules.library_selector_path_patterns.get(name))
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let patterns = if selector_patterns.is_empty() {
        rules.library_path_patterns.as_slice()
    } else {
        selector_patterns
    };
    for pattern in patterns {
        match pattern {
            LibraryPattern::Regex(pattern) => {
                let regex = compiled_regex(pattern)?;
                let Some(captures) = regex.captures(&normalized) else {
                    continue;
                };
                let whole_match = captures.get(0)?;
                let component_match = captures.get(1).unwrap_or(whole_match);
                let component = component_match.as_str();
                let relative_path = normalized[component_match.start()..].to_string();
                return Some(LibraryPath {
                    name: normalize_library_component(component, rules),
                    relative_path,
                });
            }
            LibraryPattern::Path(marker) => {
                let marker = normalize_profile_path(marker).trim_matches('/').to_string();
                if marker.is_empty() {
                    continue;
                }
                let marker_text = format!("/{marker}/");
                let haystack = format!("/{}/", normalized.trim_matches('/'));
                if !haystack.contains(&marker_text) {
                    continue;
                }
                let relative_path = haystack
                    .split_once(&marker_text)
                    .map(|(_, rest)| rest.trim_end_matches('/').to_string())?;
                let component = relative_path.split('/').next().unwrap_or_default();
                return Some(LibraryPath {
                    name: normalize_library_component(component, rules),
                    relative_path,
                });
            }
        }
    }
    None
}

fn match_library_selector(
    resolved_pattern: &str,
    path: &str,
    rules: &RuntimeRuleSet,
    selector: Option<&str>,
) -> bool {
    let library_name = extract_library_name(path, rules, selector);
    if resolved_pattern == "*" {
        return library_name.is_some();
    }
    library_name.as_deref() == Some(resolved_pattern)
}

fn pattern_mode(pattern: &str, rules: &RuntimeRuleSet) -> (PatternMode, String, Option<String>) {
    if let Some((mode, rest)) = pattern.split_once(':') {
        match mode {
            "path" | "glob" => return (PatternMode::Path, rest.to_string(), None),
            "regex" => return (PatternMode::Regex, rest.to_string(), None),
            "dependency" | "gem" | "library" | "package" | "vendor" => {
                return (
                    PatternMode::Library,
                    rest.to_string(),
                    Some(mode.to_string()),
                );
            }
            _ if rules.library_selector_path_patterns.contains_key(mode) => {
                return (
                    PatternMode::Library,
                    rest.to_string(),
                    Some(mode.to_string()),
                );
            }
            _ => {}
        }
    }
    (PatternMode::Auto, pattern.to_string(), None)
}

fn normalize_profile_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// Memoized regex compilation: rule-pack and config patterns are evaluated
/// once per unique frame path across hot per-frame loops, so compiling on
/// every call dominated runtime. Patterns come from configs, so the cache is
/// small and bounded per process.
pub(crate) fn compiled_regex(pattern: &str) -> Option<Regex> {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<Regex>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("regex cache poisoned");
    if let Some(entry) = guard.get(pattern) {
        return entry.clone();
    }
    let compiled = Regex::new(pattern).ok();
    guard.insert(pattern.to_string(), compiled.clone());
    compiled
}

fn has_glob_token(pattern: &str) -> bool {
    pattern.contains('*') || pattern.contains('?') || pattern.contains('[')
}

fn looks_like_path_pattern(pattern: &str) -> bool {
    pattern.contains('/')
        || pattern.contains('\\')
        || pattern.starts_with("./")
        || pattern.starts_with("../")
}

pub(crate) fn match_regex(pattern: &str, path: &str) -> bool {
    compiled_regex(pattern)
        .map(|regex| regex.is_match(&normalize_profile_path(path)))
        .unwrap_or(false)
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    let regex = glob_to_regex(pattern);
    match_regex(&regex, path)
}

fn glob_to_regex(pattern: &str) -> String {
    let mut output = String::from("^");
    for character in pattern.chars() {
        match character {
            '*' => output.push_str(".*"),
            '?' => output.push('.'),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | '[' | ']' | '\\' => {
                output.push('\\');
                output.push(character);
            }
            _ => output.push(character),
        }
    }
    output.push('$');
    output
}

fn normalize_library_component(component: &str, rules: &RuntimeRuleSet) -> String {
    let normalized = component.trim_matches('/');
    for pattern in &rules.library_name_suffix_patterns {
        let Some(regex) = compiled_regex(pattern) else {
            continue;
        };
        if let Some(found) = regex.find(normalized) {
            if found.start() > 0 {
                return normalized[..found.start()].to_string();
            }
        }
    }
    normalized.to_string()
}

pub fn is_runtime_stdlib_path(path: &str, rules: &RuntimeRuleSet) -> bool {
    !path.is_empty()
        && !path.starts_with('<')
        && rules
            .stdlib_path_markers
            .iter()
            .any(|marker| path.contains(marker))
}
