//! Runtime rule packs: the data-driven categorization vocabulary.
//!
//! Mirrors `clankerprof/rules.py`. Packs load from YAML with strict unknown-key
//! and schema-version validation; the packaged generic and ruby packs are
//! embedded via `include_str!` from the same files Python ships, so the two
//! implementations cannot drift by construction.

use serde_yaml::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

pub const RUNTIME_RULES_SCHEMA_VERSION: &str = "clankerprof.runtime_rules.v1";

const GENERIC_PACK_YAML: &str = include_str!("../../../clankerprof/runtime_rules/generic.yml");
const RUBY_PACK_YAML: &str = include_str!("../../../clankerprof/runtime_rules/ruby.yml");
const RUBY_CORE_CLASSES_CSV: &str =
    include_str!("../../../clankerprof/runtime_rules/ruby_core_classes.csv");

const KNOWN_RULE_PACK_KEYS: &[&str] = &[
    "name",
    "schema_version",
    "semantic_rules",
    "native_rules",
    "simplification_map",
    "main_simplified_categories",
    "always_foldable_categories",
    "verbose_only_foldable_categories",
    "special_namespace_prefixes",
    "stdlib_path_markers",
    "native_path_markers",
    "native_path_patterns",
    "native_path_exclude_markers",
    "native_path_exclude_patterns",
    "library_path_patterns",
    "library_selector_path_patterns",
    "library_name_suffix_patterns",
    "native_name_category_rules",
    "caller_fallback_name_prefixes",
    "legacy_caller_fallback_name_prefixes",
    "core_native_categories",
    "core_semantic_categories",
    "core_stdlib_categories",
    "core_internal_categories",
    "core_native_default_category",
    "stdlib_category",
    "internals_category",
];

const KNOWN_MATCH_RULE_KEYS: &[&str] = &[
    "category",
    "native_category",
    "name_contains",
    "name_prefixes",
    "name_patterns",
    "except_paths",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibraryPattern {
    Regex(String),
    Path(String),
}

fn library_pattern(raw: &str) -> LibraryPattern {
    match raw.strip_prefix("regex:") {
        Some(rest) => LibraryPattern::Regex(rest.to_string()),
        None => LibraryPattern::Path(raw.to_string()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuntimeMatchRule {
    pub category: String,
    pub name_contains: Vec<String>,
    pub name_prefixes: Vec<String>,
    pub name_patterns: Vec<String>,
    pub except_paths: BTreeSet<String>,
    pub native_category: Option<String>,
}

impl RuntimeMatchRule {
    pub fn matches(&self, name: &str, path: &str) -> bool {
        if self.except_paths.contains(path) {
            return false;
        }
        self.name_contains.iter().any(|token| name.contains(token))
            || self
                .name_prefixes
                .iter()
                .any(|prefix| name.starts_with(prefix))
            || self.name_patterns.iter().any(|pattern| {
                // Packs are validated at load, so errors here are a fail-closed
                // backstop mirroring Python's lazy _match_name_pattern.
                let resolved = pattern.strip_prefix("regex:").unwrap_or(pattern);
                let compiled = match crate::targets::raw_compiled_regex(resolved) {
                    Ok(regex) => regex,
                    Err(_) => {
                        crate::targets::record_pattern_error(format!(
                            "Invalid runtime rule name pattern '{pattern}'."
                        ));
                        return false;
                    }
                };
                match compiled.is_match(name) {
                    Ok(matched) => matched,
                    Err(_) => {
                        crate::targets::record_pattern_error(format!(
                            "Invalid runtime rule name pattern '{pattern}'."
                        ));
                        false
                    }
                }
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeRuleSet {
    pub name: String,
    pub core_classes: BTreeSet<String>,
    pub verbose: bool,
    pub enabled: bool,
    pub semantic_rules: Vec<RuntimeMatchRule>,
    pub native_rules: Vec<RuntimeMatchRule>,
    pub simplification_map: BTreeMap<String, String>,
    pub main_simplified_categories: BTreeSet<String>,
    pub always_foldable_categories: BTreeSet<String>,
    pub verbose_only_foldable_categories: BTreeSet<String>,
    pub special_namespace_prefixes: BTreeSet<String>,
    pub stdlib_path_markers: Vec<String>,
    pub native_path_markers: Vec<String>,
    pub native_path_patterns: Vec<String>,
    pub native_path_exclude_markers: Vec<String>,
    pub native_path_exclude_patterns: Vec<String>,
    pub library_path_patterns: Vec<LibraryPattern>,
    pub library_selector_path_patterns: BTreeMap<String, Vec<LibraryPattern>>,
    pub library_name_suffix_patterns: Vec<String>,
    pub native_name_category_rules: Vec<RuntimeMatchRule>,
    pub caller_fallback_name_prefixes: Vec<String>,
    pub core_native_categories: BTreeMap<String, String>,
    pub core_semantic_categories: BTreeMap<String, String>,
    pub core_stdlib_categories: BTreeMap<String, String>,
    pub core_internal_categories: BTreeMap<String, String>,
    pub core_native_default_category: String,
    pub stdlib_category: String,
    pub internals_category: String,
}

impl Default for RuntimeRuleSet {
    fn default() -> Self {
        Self {
            name: String::new(),
            core_classes: BTreeSet::new(),
            verbose: false,
            enabled: false,
            semantic_rules: Vec::new(),
            native_rules: Vec::new(),
            simplification_map: BTreeMap::new(),
            main_simplified_categories: BTreeSet::new(),
            always_foldable_categories: BTreeSet::new(),
            verbose_only_foldable_categories: BTreeSet::new(),
            special_namespace_prefixes: BTreeSet::new(),
            stdlib_path_markers: Vec::new(),
            native_path_markers: Vec::new(),
            native_path_patterns: Vec::new(),
            native_path_exclude_markers: Vec::new(),
            native_path_exclude_patterns: Vec::new(),
            library_path_patterns: Vec::new(),
            library_selector_path_patterns: BTreeMap::new(),
            library_name_suffix_patterns: Vec::new(),
            native_name_category_rules: Vec::new(),
            caller_fallback_name_prefixes: Vec::new(),
            core_native_categories: BTreeMap::new(),
            core_semantic_categories: BTreeMap::new(),
            core_stdlib_categories: BTreeMap::new(),
            core_internal_categories: BTreeMap::new(),
            core_native_default_category: "Runtime Core (Native)".to_string(),
            stdlib_category: "Runtime Stdlib".to_string(),
            internals_category: "Runtime Internals".to_string(),
        }
    }
}

impl RuntimeRuleSet {
    pub fn generic() -> &'static RuntimeRuleSet {
        static GENERIC: OnceLock<RuntimeRuleSet> = OnceLock::new();
        GENERIC.get_or_init(|| {
            load_runtime_rules_str(GENERIC_PACK_YAML, "generic", BTreeSet::new(), false)
                .expect("packaged generic.yml must parse")
        })
    }
}

/// Mapping keys reaching this point are pre-validated as strings
/// (`require_string_keys` at pack load; Python's strict YAML loader mirrors
/// it), so the fallback below is unreachable in practice.
fn key_str(value: &Value) -> String {
    value.as_str().unwrap_or_default().to_string()
}

fn require_rule_str(value: &Value, key: &str) -> Result<String, String> {
    value
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| format!("Runtime rule field {key} must be a string."))
}

fn require_entry_str(value: &Value, key: &str) -> Result<String, String> {
    value
        .as_str()
        .map(ToString::to_string)
        .ok_or_else(|| format!("Runtime rule field {key} entries must be strings."))
}

fn string_list(mapping: &serde_yaml::Mapping, key: &str) -> Result<Vec<String>, String> {
    match mapping.get(Value::String(key.to_string())) {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Sequence(items)) => items
            .iter()
            .map(|item| require_entry_str(item, key))
            .collect(),
        Some(_) => Err(format!("Runtime rule field {key} must be an array.")),
    }
}

fn string_map(
    mapping: &serde_yaml::Mapping,
    key: &str,
) -> Result<BTreeMap<String, String>, String> {
    match mapping.get(Value::String(key.to_string())) {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Mapping(entries)) => entries
            .iter()
            .map(|(entry_key, entry_value)| {
                Ok((key_str(entry_key), require_entry_str(entry_value, key)?))
            })
            .collect(),
        Some(_) => Err(format!("Runtime rule field {key} must be an object.")),
    }
}

fn library_pattern_list(
    mapping: &serde_yaml::Mapping,
    key: &str,
) -> Result<Vec<LibraryPattern>, String> {
    Ok(string_list(mapping, key)?
        .iter()
        .map(|raw| library_pattern(raw))
        .collect())
}

fn selector_pattern_map(
    mapping: &serde_yaml::Mapping,
    key: &str,
) -> Result<BTreeMap<String, Vec<LibraryPattern>>, String> {
    match mapping.get(Value::String(key.to_string())) {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Mapping(entries)) => {
            let mut result = BTreeMap::new();
            for (selector, patterns) in entries {
                let selector_name = key_str(selector);
                let converted = match patterns {
                    Value::Null => Vec::new(),
                    Value::Sequence(items) => items
                        .iter()
                        .map(|item| {
                            let entry_key = format!("{key}.{selector_name}");
                            Ok(library_pattern(&require_entry_str(item, &entry_key)?))
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                    _ => {
                        return Err(format!(
                            "Runtime rule field {key}.{selector_name} must be an array."
                        ));
                    }
                };
                result.insert(selector_name, converted);
            }
            Ok(result)
        }
        Some(_) => Err(format!("Runtime rule field {key} must be an object.")),
    }
}

fn match_rules(mapping: &serde_yaml::Mapping, key: &str) -> Result<Vec<RuntimeMatchRule>, String> {
    let raw = match mapping.get(Value::String(key.to_string())) {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::Sequence(items)) => items,
        Some(_) => return Err(format!("Runtime rule field {key} must be an array.")),
    };
    let mut rules = Vec::new();
    for item in raw {
        let Value::Mapping(entry) = item else {
            return Err(format!("Each {key} entry must be an object."));
        };
        let unknown: Vec<String> = entry
            .keys()
            .map(key_str)
            .filter(|entry_key| !KNOWN_MATCH_RULE_KEYS.contains(&entry_key.as_str()))
            .collect();
        if !unknown.is_empty() {
            let mut sorted = unknown;
            sorted.sort();
            return Err(format!("Unknown {key} entry keys: {}.", sorted.join(", ")));
        }
        let Some(category) = entry.get(Value::String("category".to_string())) else {
            return Err(format!("Each {key} entry must include category."));
        };
        let native_category = match entry.get(Value::String("native_category".to_string())) {
            None | Some(Value::Null) => None,
            Some(value) => Some(require_rule_str(value, "native_category")?),
        };
        rules.push(RuntimeMatchRule {
            category: require_rule_str(category, "category")?,
            native_category,
            name_contains: string_list(entry, "name_contains")?,
            name_prefixes: string_list(entry, "name_prefixes")?,
            name_patterns: string_list(entry, "name_patterns")?,
            except_paths: string_list(entry, "except_paths")?.into_iter().collect(),
        });
        if let Some(rule) = rules.last() {
            for pattern in &rule.name_patterns {
                let resolved = pattern.strip_prefix("regex:").unwrap_or(pattern);
                if crate::targets::raw_compiled_regex(resolved).is_err() {
                    return Err(format!("Invalid runtime rule name pattern '{pattern}'."));
                }
            }
        }
    }
    Ok(rules)
}

fn aliased_string_list(
    mapping: &serde_yaml::Mapping,
    key: &str,
    legacy_key: &str,
) -> Result<Vec<String>, String> {
    let value = string_list(mapping, key)?;
    let legacy_value = string_list(mapping, legacy_key)?;
    if !value.is_empty() && !legacy_value.is_empty() && value != legacy_value {
        return Err(format!(
            "Runtime rule fields {key} and {legacy_key} are aliases; use only one."
        ));
    }
    Ok(if value.is_empty() {
        legacy_value
    } else {
        value
    })
}

fn string_field(mapping: &serde_yaml::Mapping, key: &str, default: &str) -> Result<String, String> {
    mapping
        .get(Value::String(key.to_string()))
        .map(|value| require_rule_str(value, key))
        .unwrap_or_else(|| Ok(default.to_string()))
}

pub fn runtime_rules_from_mapping(
    mapping: &serde_yaml::Mapping,
    name: &str,
    core_classes: BTreeSet<String>,
    verbose: bool,
) -> Result<RuntimeRuleSet, String> {
    let mut unknown: Vec<String> = mapping
        .keys()
        .map(key_str)
        .filter(|key| !KNOWN_RULE_PACK_KEYS.contains(&key.as_str()))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!(
            "Unknown runtime rule pack keys: {}.",
            unknown.join(", ")
        ));
    }
    let schema_version = string_field(mapping, "schema_version", RUNTIME_RULES_SCHEMA_VERSION)?;
    if schema_version != RUNTIME_RULES_SCHEMA_VERSION {
        return Err(format!(
            "Unsupported runtime rules schema version: {schema_version:?}; expected {RUNTIME_RULES_SCHEMA_VERSION:?}."
        ));
    }
    Ok(RuntimeRuleSet {
        name: string_field(mapping, "name", name)?,
        core_classes,
        verbose,
        enabled: true,
        semantic_rules: match_rules(mapping, "semantic_rules")?,
        native_rules: match_rules(mapping, "native_rules")?,
        simplification_map: string_map(mapping, "simplification_map")?,
        main_simplified_categories: string_list(mapping, "main_simplified_categories")?
            .into_iter()
            .collect(),
        always_foldable_categories: string_list(mapping, "always_foldable_categories")?
            .into_iter()
            .collect(),
        verbose_only_foldable_categories: string_list(mapping, "verbose_only_foldable_categories")?
            .into_iter()
            .collect(),
        special_namespace_prefixes: string_list(mapping, "special_namespace_prefixes")?
            .into_iter()
            .collect(),
        stdlib_path_markers: string_list(mapping, "stdlib_path_markers")?,
        native_path_markers: string_list(mapping, "native_path_markers")?,
        native_path_patterns: string_list(mapping, "native_path_patterns")?,
        native_path_exclude_markers: string_list(mapping, "native_path_exclude_markers")?,
        native_path_exclude_patterns: string_list(mapping, "native_path_exclude_patterns")?,
        library_path_patterns: library_pattern_list(mapping, "library_path_patterns")?,
        library_selector_path_patterns: selector_pattern_map(
            mapping,
            "library_selector_path_patterns",
        )?,
        library_name_suffix_patterns: string_list(mapping, "library_name_suffix_patterns")?,
        native_name_category_rules: match_rules(mapping, "native_name_category_rules")?,
        caller_fallback_name_prefixes: aliased_string_list(
            mapping,
            "caller_fallback_name_prefixes",
            "legacy_caller_fallback_name_prefixes",
        )?,
        core_native_categories: string_map(mapping, "core_native_categories")?,
        core_semantic_categories: string_map(mapping, "core_semantic_categories")?,
        core_stdlib_categories: string_map(mapping, "core_stdlib_categories")?,
        core_internal_categories: string_map(mapping, "core_internal_categories")?,
        core_native_default_category: string_field(
            mapping,
            "core_native_default_category",
            "Runtime Core (Native)",
        )?,
        stdlib_category: string_field(mapping, "stdlib_category", "Runtime Stdlib")?,
        internals_category: string_field(mapping, "internals_category", "Runtime Internals")?,
    })
}

/// Shared with Python's strict YAML loader: clankerprof YAML inputs reject
/// non-string mapping keys (bool/number/null keys have no shared spelling
/// between str() and serde's Display, so neither implementation coerces).
pub const YAML_KEY_MESSAGE: &str = "YAML mapping keys must be strings.";

/// serde_yaml represents only local (`!name`) tags as `Value::Tagged` —
/// global tags are resolved or ignored at parse. Python's strict loader
/// mirrors that split and rejects local tags with this same message.
pub const YAML_LOCAL_TAG_MESSAGE: &str = "YAML local tags are not supported in clankerprof inputs.";

/// Walk a parsed YAML tree and reject any non-string mapping key or
/// local-tagged value.
pub fn require_string_keys(value: &Value) -> Result<(), String> {
    match value {
        Value::Mapping(mapping) => {
            for (key, item) in mapping {
                if !key.is_string() {
                    return Err(YAML_KEY_MESSAGE.to_string());
                }
                require_string_keys(item)?;
            }
            Ok(())
        }
        Value::Sequence(items) => {
            for item in items {
                require_string_keys(item)?;
            }
            Ok(())
        }
        Value::Tagged(_) => Err(YAML_LOCAL_TAG_MESSAGE.to_string()),
        _ => Ok(()),
    }
}

pub fn load_runtime_rules_str(
    payload: &str,
    name: &str,
    core_classes: BTreeSet<String>,
    verbose: bool,
) -> Result<RuntimeRuleSet, String> {
    let value: Value = serde_yaml::from_str(payload).map_err(|error| error.to_string())?;
    require_string_keys(&value)?;
    let Value::Mapping(mapping) = value else {
        return Err(format!("Runtime rule pack {name} must be a YAML object."));
    };
    runtime_rules_from_mapping(&mapping, name, core_classes, verbose)
}

pub fn load_runtime_rules_file(
    path: impl AsRef<Path>,
    core_classes: BTreeSet<String>,
    verbose: bool,
) -> Result<RuntimeRuleSet, String> {
    let path = path.as_ref();
    let payload = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("custom");
    load_runtime_rules_str(&payload, stem, core_classes, verbose)
}

pub fn ruby_rules(core_classes: BTreeSet<String>, verbose: bool) -> Result<RuntimeRuleSet, String> {
    load_runtime_rules_str(RUBY_PACK_YAML, "ruby", core_classes, verbose)
}

/// First CSV field of each record with Python `csv.reader` default-dialect
/// semantics, scanning the whole payload so quoted fields may span record
/// separators (line splitting tore multiline quoted fields apart). Pinned
/// empirically against csv.reader: a quote is special only at the exact
/// start of a field; `""` inside a quoted field unescapes; after a closing
/// quote another `"` re-enters the quoted field as a literal quote while
/// any other character appends and continues unquoted (non-strict); an
/// unterminated quote runs to end of input; the `#` comment check applies
/// after unquoting and trimming.
fn core_classes_from_csv(payload: &str) -> BTreeSet<String> {
    enum State {
        StartField,
        InField,
        InQuoted,
        QuoteInQuoted,
    }
    let mut values = BTreeSet::new();
    let mut chars = payload.chars().peekable();
    while chars.peek().is_some() {
        let mut first_field = String::new();
        let mut in_first_field = true;
        let mut state = State::StartField;
        while let Some(ch) = chars.next() {
            let record_break = match state {
                State::StartField => match ch {
                    '"' => {
                        state = State::InQuoted;
                        false
                    }
                    ',' => {
                        in_first_field = false;
                        false
                    }
                    '\n' | '\r' => true,
                    other => {
                        if in_first_field {
                            first_field.push(other);
                        }
                        state = State::InField;
                        false
                    }
                },
                State::InField => match ch {
                    ',' => {
                        in_first_field = false;
                        state = State::StartField;
                        false
                    }
                    '\n' | '\r' => true,
                    other => {
                        if in_first_field {
                            first_field.push(other);
                        }
                        false
                    }
                },
                State::InQuoted => {
                    if ch == '"' {
                        state = State::QuoteInQuoted;
                    } else if in_first_field {
                        first_field.push(ch);
                    }
                    false
                }
                State::QuoteInQuoted => match ch {
                    '"' => {
                        if in_first_field {
                            first_field.push('"');
                        }
                        state = State::InQuoted;
                        false
                    }
                    ',' => {
                        in_first_field = false;
                        state = State::StartField;
                        false
                    }
                    '\n' | '\r' => true,
                    other => {
                        if in_first_field {
                            first_field.push(other);
                        }
                        state = State::InField;
                        false
                    }
                },
            };
            if record_break {
                if ch == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                break;
            }
        }
        let value = first_field.trim().to_string();
        if !value.is_empty() && !value.starts_with('#') {
            values.insert(value);
        }
    }
    values
}

pub fn load_default_ruby_core_classes() -> BTreeSet<String> {
    core_classes_from_csv(RUBY_CORE_CLASSES_CSV)
}

pub fn load_ruby_core_classes(path: impl AsRef<Path>) -> Result<BTreeSet<String>, String> {
    let payload = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    Ok(core_classes_from_csv(&payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_classes_csv_first_field_matches_python_csv_reader() {
        // Quoted fields unwrap, doubled quotes unescape, quoted commas stay
        // inside the field, unquoted lines split at the first comma, and
        // the '#' comment check applies after unquoting+trim.
        let parsed = core_classes_from_csv(
            "\"Array\"\nHash,ignored\n\"Set\",ignored\n\"Em\"\"bed\"\n\"Com,ma\"\n\n\"#quoted-comment\"\n#comment\n",
        );
        let expected: BTreeSet<String> = ["Array", "Hash", "Set", "Em\"bed", "Com,ma"]
            .into_iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(parsed, expected);
    }

    #[test]
    fn core_classes_csv_record_scanning_matches_python_csv_reader() {
        // Every expectation below is pinned against CPython csv.reader
        // (default dialect, file opened newline=""), the documented contract.
        // Quoted fields span newlines; quotes are special only at exact
        // field start; after a closing quote a bare char continues the
        // field unquoted while another quote re-enters the quoted state.
        let cases: [(&str, &[&str]); 7] = [
            ("\"Weird\nClass\"\n", &["Weird\nClass"]),
            ("\"A\r\nB\"\r\nPlain\r\n", &["A\r\nB", "Plain"]),
            ("\"a\"b,c\n", &["ab"]),
            ("a\"b\"c,d\n", &["a\"b\"c"]),
            ("\"a\"\",c\n", &["a\",c"]),
            ("\"unterminated", &["unterminated"]),
            ("x,\"quoted\nsecond\"\ny\n", &["x", "y"]),
        ];
        for (payload, expected) in cases {
            let parsed = core_classes_from_csv(payload);
            let expected: BTreeSet<String> = expected.iter().map(ToString::to_string).collect();
            assert_eq!(parsed, expected, "payload {payload:?}");
        }
    }

    #[test]
    fn packaged_generic_pack_parses_with_expected_fields() {
        let rules = RuntimeRuleSet::generic();
        assert_eq!(rules.name, "generic");
        assert!(rules.enabled);
        assert!(rules.semantic_rules.is_empty());
        assert_eq!(
            rules.native_path_markers,
            vec!["<cfunc>".to_string(), "<internal:".to_string()]
        );
        assert_eq!(rules.library_path_patterns.len(), 4);
        assert!(matches!(
            rules.library_path_patterns[0],
            LibraryPattern::Regex(ref pattern) if pattern == "/vendor/([^/]+)/"
        ));
        assert_eq!(
            rules.library_selector_path_patterns["gem"],
            vec![LibraryPattern::Regex("/gems/([^/]+)/".to_string())]
        );
    }

    #[test]
    fn packaged_ruby_pack_parses_with_ordered_rules() {
        let rules = ruby_rules(load_default_ruby_core_classes(), false).expect("ruby pack");
        assert_eq!(rules.name, "ruby");
        assert_eq!(rules.semantic_rules[0].category, "StatsD Gem");
        assert_eq!(
            rules.semantic_rules[0].native_category.as_deref(),
            Some("StatsD (Native)")
        );
        assert_eq!(rules.native_rules[0].category, "Template Engine Native");
        assert!(rules.core_classes.contains("Array"));
        assert_eq!(
            rules.simplification_map["StatsD Gem"],
            "Instrumentation Overhead"
        );
        assert!(rules
            .main_simplified_categories
            .contains("Instrumentation Overhead"));
        assert!(rules.special_namespace_prefixes.contains("Zlib"));
    }

    #[test]
    fn unknown_keys_and_versions_are_rejected() {
        let error =
            load_runtime_rules_str("name: x\nsurprise_key: 1\n", "x", BTreeSet::new(), false)
                .unwrap_err();
        assert!(error.contains("Unknown runtime rule pack keys: surprise_key."));

        let error = load_runtime_rules_str(
            "schema_version: clankerprof.runtime_rules.v9\nname: x\n",
            "x",
            BTreeSet::new(),
            false,
        )
        .unwrap_err();
        assert!(error.contains("Unsupported runtime rules schema version"));

        let error = load_runtime_rules_str(
            "name: x\nsemantic_rules:\n  - category: A\n    name_glob: '*'\n",
            "x",
            BTreeSet::new(),
            false,
        )
        .unwrap_err();
        assert!(error.contains("Unknown semantic_rules entry keys: name_glob."));
    }

    #[test]
    fn match_rule_semantics_cover_contains_prefixes_and_excepts() {
        let rules = ruby_rules(load_default_ruby_core_classes(), false).expect("ruby pack");
        let statsd = &rules.semantic_rules[0];
        assert!(statsd.matches("StatsD.increment", "<cfunc>"));
        assert!(statsd.matches("MyStatsDHelper#emit", "/app/x.rb"));
        let trilogy = rules
            .semantic_rules
            .iter()
            .find(|rule| rule.category == "Trilogy Gem")
            .expect("trilogy rule");
        assert!(!trilogy.matches("Trilogy#query", "<cfunc>"));
        assert!(trilogy.matches("Trilogy#query", "/gems/trilogy/lib.rb"));
    }
}
