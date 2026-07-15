use crate::categorize::{categorize_stack, RuntimeCategoryCache};
use crate::facts::{sample_facts_to_json_value, SAMPLE_FACTS_SCHEMA_VERSION};
use crate::model::{CategoryStats, Frame, FunctionMetrics, ProfileFacts, TimeNs};
use fancy_regex::Regex;
use indexmap::IndexMap;
use serde_json::value::RawValue;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

// Category precedence is first-match-wins in config order, ranked arrays
// break ties by first-seen order, and parents emit in first-seen encounter
// order across every row format, matching the Python reference; all of them
// need insertion-ordered maps, never BTreeMap.
pub type TargetConfig = IndexMap<String, IndexMap<String, String>>;
pub type TargetResults = IndexMap<String, IndexMap<String, CategoryStats>>;

thread_local! {
    // Python raises on the first invalid user pattern it evaluates; the hot
    // matchers here return plain bools through the categorization engine, so
    // instead of threading Results through every closure the first pattern
    // error is parked in this slot and the CLI fails closed (exit 2) before
    // emitting or writing any artifact. Lazy like Python: a pattern that no
    // frame ever reaches raises in neither implementation.
    static PATTERN_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub(crate) fn record_pattern_error(message: String) {
    PATTERN_ERROR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(message);
        }
    });
}

/// Fail closed on any user-pattern error recorded during analysis/rendering.
/// Must be checked before emitting or writing any artifact.
pub fn take_pattern_error() -> Result<(), String> {
    PATTERN_ERROR
        .with(|slot| slot.borrow_mut().take())
        .map_or(Ok(()), Err)
}

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
                        "pct": if total == 0 { json!(0) } else { json!(stats.cpu_time as f64 / total as f64 * 100.0) },
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

fn render_function_metrics(functions: &IndexMap<String, FunctionMetrics>) -> Value {
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
        PatternMode::Auto => match try_match_regex(&resolved_pattern, path) {
            // Mirrors Python match_category_pattern: a valid-regex match wins;
            // no-match falls through to path matching for path-like patterns;
            // an invalid regex falls back to path matching iff it looks like a
            // path, otherwise the projection fails closed.
            Ok(true) => true,
            Ok(false) => {
                looks_like_path_pattern(&resolved_pattern)
                    && match_path_pattern(&resolved_pattern, path, rules)
            }
            Err(error) => {
                if looks_like_path_pattern(&resolved_pattern) {
                    match_path_pattern(&resolved_pattern, path, rules)
                } else {
                    record_pattern_error(error);
                    false
                }
            }
        },
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
                let regex = match raw_compiled_regex(pattern) {
                    Ok(regex) => regex,
                    Err(detail) => {
                        record_pattern_error(format!(
                            "Invalid library regex pattern '{pattern}': {detail}"
                        ));
                        return None;
                    }
                };
                let captures = match regex.captures(&normalized) {
                    Ok(captures) => captures,
                    Err(detail) => {
                        record_pattern_error(format!(
                            "Invalid library regex pattern '{pattern}': {detail}"
                        ));
                        return None;
                    }
                };
                let Some(captures) = captures else {
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
/// small and bounded per process. fancy-regex is used because the documented
/// pattern language is Python's (lookaround included); the Err arm carries
/// the engine's own detail so callers can build their contract messages.
pub(crate) fn raw_compiled_regex(pattern: &str) -> Result<Arc<Regex>, String> {
    static CACHE: OnceLock<Mutex<HashMap<String, Result<Arc<Regex>, String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().expect("regex cache poisoned");
    if let Some(entry) = guard.get(pattern) {
        return entry.clone();
    }
    let compiled = Regex::new(pattern)
        .map(Arc::new)
        .map_err(|error| error.to_string());
    guard.insert(pattern.to_string(), compiled.clone());
    compiled
}

/// Non-recording compile with the shared contract message; auto-mode needs
/// the error value to decide between path fallback and failing closed.
pub(crate) fn try_compiled_regex(pattern: &str) -> Result<Arc<Regex>, String> {
    raw_compiled_regex(pattern)
        .map_err(|detail| format!("Invalid regex pattern '{pattern}': {detail}"))
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

/// Regex match that fails closed: compile or match errors are recorded in the
/// pattern-error slot (first error wins) and count as no-match so the caller
/// keeps its bool shape; the CLI turns the slot into an exit-2 envelope.
pub(crate) fn match_regex(pattern: &str, path: &str) -> bool {
    match try_match_regex(pattern, path) {
        Ok(matched) => matched,
        Err(error) => {
            record_pattern_error(error);
            false
        }
    }
}

/// Non-recording variant for auto-mode, which mirrors Python's behavior of
/// falling back to path matching when an invalid pattern looks like a path.
pub(crate) fn try_match_regex(pattern: &str, path: &str) -> Result<bool, String> {
    let regex = try_compiled_regex(pattern)?;
    regex
        .is_match(&normalize_profile_path(path))
        .map_err(|detail| format!("Invalid regex pattern '{pattern}': {detail}"))
}

fn glob_matches(pattern: &str, path: &str) -> bool {
    // Glob-derived regexes are internally generated and always compile; a
    // None translation is fnmatch's never-matching class.
    glob_to_regex(pattern)
        .map(|regex| match_regex(&regex, path))
        .unwrap_or(false)
}

/// Translate a glob into a regex with CPython `fnmatch.translate` semantics —
/// the Python reference routes glob patterns through `fnmatch`, so bracket
/// classes (`[ab]`, `[!ab]`, ranges, a literal `]` in first position, an
/// unterminated `[` treated as a literal) must all behave identically here.
/// Returns `None` for patterns fnmatch translates to a never-matching class
/// (an empty range like `[z-a]`), since the regex crate has no `(?!)`.
fn glob_to_regex(pattern: &str) -> Option<String> {
    let chars: Vec<char> = pattern.chars().collect();
    let n = chars.len();
    // (?s) mirrors fnmatch's re.DOTALL wrapper so `*`/`?` cross newlines.
    let mut output = String::from("(?s)^");
    let mut i = 0;
    while i < n {
        let character = chars[i];
        i += 1;
        match character {
            '*' => output.push_str(".*"),
            '?' => output.push('.'),
            '[' => {
                let mut j = i;
                if j < n && chars[j] == '!' {
                    j += 1;
                }
                if j < n && chars[j] == ']' {
                    j += 1;
                }
                while j < n && chars[j] != ']' {
                    j += 1;
                }
                if j >= n {
                    // Unterminated class stays a literal '[' (fnmatch parity).
                    output.push_str("\\[");
                } else {
                    let stuff: Vec<char> = chars[i..j].to_vec();
                    i = j + 1;
                    output.push_str(&translate_glob_class(&stuff)?);
                }
            }
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '{' | '}' | ']' | '\\' => {
                output.push('\\');
                output.push(character);
            }
            _ => output.push(character),
        }
    }
    output.push('$');
    Some(output)
}

/// Bracket-class body translation ported from CPython fnmatch.translate,
/// emitting escapes the regex crate accepts (it rejects Python's bare `]`
/// first-position literal and treats bare `[` as nested-class syntax).
/// Returns `None` when range normalization empties the class (never match).
fn translate_glob_class(stuff: &[char]) -> Option<String> {
    let raw: String = stuff.iter().collect();
    let chunks: Vec<String> = if !raw.contains('-') {
        vec![raw.clone()]
    } else {
        // Split on range-forming hyphens exactly the way CPython does, then
        // merge away inverted (empty) ranges like `z-a`.
        let mut chunks: Vec<String> = Vec::new();
        let mut start = 0usize;
        let mut k = if stuff.first() == Some(&'!') { 2 } else { 1 };
        loop {
            let Some(offset) = stuff[k.min(stuff.len())..]
                .iter()
                .position(|&item| item == '-')
                .map(|position| position + k.min(stuff.len()))
            else {
                break;
            };
            chunks.push(stuff[start..offset].iter().collect());
            start = offset + 1;
            k = offset + 3;
        }
        let tail: String = stuff[start..].iter().collect();
        if tail.is_empty() {
            let last = chunks.last_mut()?;
            last.push('-');
        } else {
            chunks.push(tail);
        }
        let mut index = chunks.len().saturating_sub(1);
        while index > 0 {
            let left_last = chunks[index - 1].chars().last();
            let right_first = chunks[index].chars().next();
            if let (Some(left), Some(right)) = (left_last, right_first) {
                if left > right {
                    let merged: String = chunks[index - 1]
                        .chars()
                        .take(chunks[index - 1].chars().count() - 1)
                        .chain(chunks[index].chars().skip(1))
                        .collect();
                    chunks[index - 1] = merged;
                    chunks.remove(index);
                }
            }
            index -= 1;
        }
        chunks
    };
    let escaped: Vec<String> = chunks
        .iter()
        .map(|chunk| {
            let mut item = String::new();
            for character in chunk.chars() {
                // '\\' and '-' per CPython; '[', ']', '&', '~', '|' because
                // the regex crate gives them class-level meaning Python
                // doesn't (nested classes and set operations).
                if matches!(character, '\\' | '-' | '[' | ']' | '&' | '~' | '|') {
                    item.push('\\');
                }
                item.push(character);
            }
            item
        })
        .collect();
    let mut body = escaped.join("-");
    if body.is_empty() {
        // CPython emits the never-matching (?!) here; signal it upward.
        return None;
    }
    if body == "!" {
        return Some(".".to_string());
    }
    if let Some(rest) = body.strip_prefix('!') {
        body = format!("^{rest}");
    } else if body.starts_with('^') {
        body = format!("\\{body}");
    }
    Some(format!("[{body}]"))
}

fn normalize_library_component(component: &str, rules: &RuntimeRuleSet) -> String {
    let normalized = component.trim_matches('/');
    for pattern in &rules.library_name_suffix_patterns {
        let regex = match try_compiled_regex(pattern) {
            Ok(regex) => regex,
            Err(error) => {
                record_pattern_error(error);
                continue;
            }
        };
        match regex.find(normalized) {
            Ok(Some(found)) if found.start() > 0 => {
                return normalized[..found.start()].to_string();
            }
            Ok(_) => {}
            Err(detail) => {
                record_pattern_error(format!("Invalid regex pattern '{pattern}': {detail}"));
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

#[cfg(test)]
mod glob_tests {
    use super::glob_matches;

    #[test]
    fn glob_matches_follows_cpython_fnmatch_semantics() {
        // Expectations generated from CPython 3.11 fnmatch.fnmatch — the
        // Python reference routes glob patterns straight through fnmatch.
        let table: &[(&str, &str, bool)] = &[
            ("/app/[ab].rb", "/app/a.rb", true),
            ("/app/[ab].rb", "/app/b.rb", true),
            ("/app/[ab].rb", "/app/c.rb", false),
            ("/app/[!ab].rb", "/app/a.rb", false),
            ("/app/[!ab].rb", "/app/c.rb", true),
            ("/app/[a-c].rb", "/app/b.rb", true),
            ("/app/[a-c].rb", "/app/d.rb", false),
            ("/app/[z-a].rb", "/app/a.rb", false),
            ("/app/[z-a].rb", "/app/z.rb", false),
            ("/app/[ab.rb", "/app/[ab.rb", true),
            ("/app/[ab.rb", "/app/a.rb", false),
            ("/app/[]a].rb", "/app/].rb", true),
            ("/app/[]a].rb", "/app/a.rb", true),
            ("/app/[]a].rb", "/app/x.rb", false),
            ("/app/[!]a].rb", "/app/x.rb", true),
            ("/app/[!]a].rb", "/app/].rb", false),
            ("/app/[!]a].rb", "/app/a.rb", false),
            ("/app/[a-].rb", "/app/-.rb", true),
            ("/app/[a-].rb", "/app/a.rb", true),
            ("/app/[a-].rb", "/app/b.rb", false),
            ("/app/[^a].rb", "/app/^.rb", true),
            ("/app/[^a].rb", "/app/a.rb", true),
            ("/app/[^a].rb", "/app/b.rb", false),
            ("/app/[[].rb", "/app/[.rb", true),
            ("/app/[[].rb", "/app/a.rb", false),
            ("/app/[a-c-e].rb", "/app/b.rb", true),
            ("/app/[a-c-e].rb", "/app/-.rb", true),
            ("/app/[a-c-e].rb", "/app/e.rb", true),
            ("/app/[a-c-e].rb", "/app/d.rb", false),
            ("/app/*[0-9]?.rb", "/app/x42.rb", true),
            ("/app/*[0-9]?.rb", "/app/x4.rb", false),
        ];
        for (pattern, path, expected) in table {
            assert_eq!(
                glob_matches(pattern, path),
                *expected,
                "pattern {pattern:?} vs path {path:?}"
            );
        }
    }
}
