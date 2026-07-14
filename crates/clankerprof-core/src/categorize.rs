//! The runtime categorization engine, mirroring `clankerprof/categorize.py`
//! and the path-ownership rules from `clankerprof/patterns.py`.

use crate::model::Frame;
use crate::rules::RuntimeRuleSet;
use crate::targets::{
    extract_library_name, is_runtime_stdlib_path, match_category_pattern, match_regex,
};
use std::collections::HashMap;

pub fn is_native_path(path: &str, rules: &RuntimeRuleSet) -> bool {
    if path.is_empty() || path.starts_with('<') {
        return true;
    }
    if rules
        .native_path_exclude_markers
        .iter()
        .any(|marker| path.contains(marker))
    {
        return false;
    }
    if rules
        .native_path_exclude_patterns
        .iter()
        .any(|pattern| match_regex(pattern, path))
    {
        return false;
    }
    if rules
        .native_path_markers
        .iter()
        .any(|marker| path.contains(marker))
    {
        return true;
    }
    rules
        .native_path_patterns
        .iter()
        .any(|pattern| match_category_pattern(pattern, path, rules))
}

/// Whether the runtime, its stdlib, or a dependency owns this path.
///
/// Semantic rules never claim frames on plain application paths; a pack that
/// declares no path-ownership configuration cannot distinguish application
/// paths, so its semantic rules apply to every frame.
fn is_runtime_owned_path(path: &str, rules: &RuntimeRuleSet) -> bool {
    let has_ownership_config = !rules.native_path_markers.is_empty()
        || !rules.native_path_patterns.is_empty()
        || !rules.stdlib_path_markers.is_empty()
        || !rules.library_path_patterns.is_empty()
        || !rules.library_selector_path_patterns.is_empty();
    if !has_ownership_config {
        return true;
    }
    if is_native_path(path, rules) {
        return true;
    }
    if is_runtime_stdlib_path(path, rules) {
        return true;
    }
    extract_library_name(path, rules, None).is_some()
}

fn class_name_of(function_name: &str) -> Option<&str> {
    let clean = function_name.trim_start_matches(':');
    if let Some((class_name, _)) = clean.split_once('#') {
        return Some(class_name).filter(|name| !name.is_empty());
    }
    if let Some((class_name, _)) = clean.split_once('.') {
        return Some(class_name).filter(|name| !name.is_empty());
    }
    if clean.contains("::") {
        let (class_name, _) = clean.rsplit_once("::").expect("contains ::");
        return Some(class_name).filter(|name| !name.is_empty());
    }
    None
}

fn direct_core_class<'a>(function_name: &'a str, rules: &'a RuntimeRuleSet) -> Option<&'a str> {
    let class_name = class_name_of(function_name)?;
    if class_name.contains("::") {
        let namespace = class_name.split("::").next().unwrap_or_default();
        if rules.special_namespace_prefixes.contains(namespace) {
            return None;
        }
        return class_name
            .split("::")
            .find(|component| rules.core_classes.contains(*component));
    }
    if rules.core_classes.contains(class_name) {
        return Some(class_name);
    }
    None
}

/// Whether the name is a bare module-function on a guarded namespace
/// (`Zlib.inflate`): native-path categorization gives the pack's native-name
/// rules the first claim before the core table's default swallows them.
fn bare_guarded_namespace(function_name: &str, rules: &RuntimeRuleSet) -> bool {
    let clean = function_name.trim_start_matches(':');
    let class_name = if let Some((class_name, _)) = clean.split_once('#') {
        class_name
    } else if let Some((class_name, _)) = clean.split_once('.') {
        class_name
    } else {
        return false;
    };
    !class_name.contains("::") && rules.special_namespace_prefixes.contains(class_name)
}

pub fn categorize_runtime_frame(frame: &Frame, rules: &RuntimeRuleSet) -> Option<String> {
    let name = frame.name.as_str();
    let path = frame.filename.as_str();

    if !rules.semantic_rules.is_empty() && is_runtime_owned_path(path, rules) {
        for rule in &rules.semantic_rules {
            if rule.matches(name, path) {
                if path == "<cfunc>" {
                    if let Some(native_category) = &rule.native_category {
                        return Some(native_category.clone());
                    }
                }
                return Some(rule.category.clone());
            }
        }
    }

    if let Some(core_class) = direct_core_class(name, rules) {
        let explicit_native_path = path == "<cfunc>"
            || (!path.starts_with("<internal:")
                && (path.starts_with('<')
                    || rules
                        .native_path_markers
                        .iter()
                        .any(|marker| path.contains(marker))));
        if explicit_native_path {
            if bare_guarded_namespace(name, rules) {
                for rule in &rules.native_rules {
                    if rule.matches(name, path) {
                        return Some(rule.category.clone());
                    }
                }
            }
            return Some(
                rules
                    .core_native_categories
                    .get(core_class)
                    .cloned()
                    .unwrap_or_else(|| rules.core_native_default_category.clone()),
            );
        }
        if is_runtime_stdlib_path(path, rules) {
            return Some(
                rules
                    .core_stdlib_categories
                    .get(core_class)
                    .cloned()
                    .unwrap_or_else(|| rules.stdlib_category.clone()),
            );
        }
        if path.starts_with("<internal:") {
            return Some(
                rules
                    .core_internal_categories
                    .get(core_class)
                    .cloned()
                    .unwrap_or_else(|| rules.internals_category.clone()),
            );
        }
        return rules.core_semantic_categories.get(core_class).cloned();
    }

    if path == "<cfunc>" {
        for rule in &rules.native_rules {
            if rule.matches(name, path) {
                return Some(rule.category.clone());
            }
        }
    }
    if path.starts_with("<internal:") {
        return Some(rules.internals_category.clone());
    }
    if is_runtime_stdlib_path(path, rules) {
        return Some(rules.stdlib_category.clone());
    }
    None
}

fn has_runtime_categories(rules: &RuntimeRuleSet) -> bool {
    !rules.semantic_rules.is_empty()
        || !rules.native_rules.is_empty()
        || !rules.core_classes.is_empty()
        || !rules.stdlib_path_markers.is_empty()
        || !rules.core_native_categories.is_empty()
}

pub fn categorize_frame(frame: &Frame, rules: &RuntimeRuleSet) -> Option<String> {
    if rules.enabled && has_runtime_categories(rules) {
        return categorize_runtime_frame(frame, rules);
    }
    None
}

pub fn simplify_category(category: &str, verbose: bool, rules: &RuntimeRuleSet) -> String {
    if verbose {
        return category.to_string();
    }
    rules
        .simplification_map
        .get(category)
        .cloned()
        .unwrap_or_else(|| category.to_string())
}

pub fn is_internal_category_for_rules(category: Option<&str>, rules: &RuntimeRuleSet) -> bool {
    let Some(category) = category else {
        return false;
    };
    if rules.always_foldable_categories.contains(category) {
        return true;
    }
    rules.verbose && rules.verbose_only_foldable_categories.contains(category)
}

fn should_fold_category(
    category: Option<&str>,
    stack: &[Frame],
    rules: &RuntimeRuleSet,
    fold_runtime_internals: bool,
    cache: &mut RuntimeCategoryCache,
) -> bool {
    let Some(category) = category else {
        return false;
    };
    if !rules.verbose {
        let simplified = simplify_category(category, false, rules);
        if rules.main_simplified_categories.contains(&simplified) {
            return false;
        }
    }
    if fold_runtime_internals && rules.always_foldable_categories.contains(category) {
        return true;
    }
    if fold_runtime_internals
        && rules.verbose
        && rules.verbose_only_foldable_categories.contains(category)
    {
        return true;
    }
    if rules.always_foldable_categories.contains(category) {
        // The caller window spans the next two distinct locations so the
        // outcome is independent of inline expansion of the leaf's location.
        let leaf_location_id = stack.first().map(|frame| frame.location_id).unwrap_or(0);
        let mut window_location_ids: Vec<u64> = Vec::new();
        for caller in stack.iter().skip(1) {
            if caller.location_id == leaf_location_id && window_location_ids.is_empty() {
                continue;
            }
            if !window_location_ids.contains(&caller.location_id) {
                if window_location_ids.len() == 2 {
                    break;
                }
                window_location_ids.push(caller.location_id);
            }
            let caller_category = cache.category_for(caller, rules);
            if let Some(caller_category) = caller_category {
                if !is_internal_category_for_rules(Some(caller_category.as_str()), rules) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn first_non_runtime_file_caller<'a>(
    stack: &'a [Frame],
    rules: &RuntimeRuleSet,
) -> Option<&'a Frame> {
    stack.iter().skip(1).find(|frame| {
        !frame.filename.starts_with('<') && !is_runtime_stdlib_path(&frame.filename, rules)
    })
}

#[derive(Debug, Default)]
pub struct RuntimeCategoryCache {
    cache: HashMap<(u64, String, String), Option<String>>,
}

impl RuntimeCategoryCache {
    pub fn category_for(&mut self, frame: &Frame, rules: &RuntimeRuleSet) -> Option<String> {
        let key = (
            frame.function_id,
            frame.name.clone(),
            frame.filename.clone(),
        );
        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }
        let category = categorize_frame(frame, rules);
        self.cache.insert(key, category.clone());
        category
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackCategory {
    pub category: String,
    pub frame_index: usize,
    pub folded: bool,
    pub folded_category: Option<String>,
}

/// The shared categorization pipeline, mirroring Python's `categorize_stack`.
#[allow(clippy::too_many_arguments)]
pub fn categorize_stack(
    stack: &[Frame],
    rules: &RuntimeRuleSet,
    enhanced_runtime_categorization: bool,
    fold_runtime_internals: bool,
    caller_fallback_when_uncategorized: bool,
    cache: &mut RuntimeCategoryCache,
    configured_category_for: &mut dyn FnMut(&Frame) -> Option<String>,
) -> StackCategory {
    let leaf = &stack[0];
    let mut category = if enhanced_runtime_categorization {
        cache.category_for(leaf, rules)
    } else {
        None
    };
    let mut frame_index = 0usize;
    let mut folded = false;
    let mut folded_category: Option<String> = None;

    if should_fold_category(
        category.as_deref(),
        stack,
        rules,
        fold_runtime_internals,
        cache,
    ) {
        for (position, caller) in stack.iter().enumerate().skip(1) {
            let caller_category = cache.category_for(caller, rules);
            if is_internal_category_for_rules(caller_category.as_deref(), rules)
                || is_runtime_stdlib_path(&caller.filename, rules)
            {
                continue;
            }
            frame_index = position;
            folded = true;
            folded_category = category.clone();
            category = caller_category;
            break;
        }
    }

    if category.is_none() && caller_fallback_when_uncategorized {
        let should_walk_up = leaf.filename.starts_with('<')
            || (is_runtime_stdlib_path(&leaf.filename, rules)
                && rules
                    .caller_fallback_name_prefixes
                    .iter()
                    .any(|prefix| leaf.name.starts_with(prefix)));
        if should_walk_up {
            if let Some(caller_position) =
                stack
                    .iter()
                    .enumerate()
                    .skip(1)
                    .find_map(|(position, frame)| {
                        (!frame.filename.starts_with('<')
                            && !is_runtime_stdlib_path(&frame.filename, rules))
                        .then_some(position)
                    })
            {
                frame_index = caller_position;
            }
        }
    }

    let frame_to_categorize = &stack[frame_index];
    if category.is_none() && folded && frame_to_categorize.filename.starts_with('<') {
        for rule in &rules.native_name_category_rules {
            if rule.matches(&frame_to_categorize.name, &frame_to_categorize.filename) {
                category = Some(rule.category.clone());
                break;
            }
        }
    }

    if category.is_none() {
        category = configured_category_for(frame_to_categorize);
    }

    let mut resolved = category.unwrap_or_else(|| "Other".to_string());
    if !rules.main_simplified_categories.contains(&resolved) {
        let simplified = simplify_category(&resolved, rules.verbose, rules);
        resolved = if simplified.is_empty() {
            "Other".to_string()
        } else {
            simplified
        };
    }

    StackCategory {
        category: resolved,
        frame_index,
        folded,
        folded_category,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{load_default_ruby_core_classes, ruby_rules};

    fn frame(name: &str, filename: &str) -> Frame {
        Frame {
            location_id: 1,
            function_id: 1,
            name: name.to_string(),
            filename: filename.to_string(),
            line: 1,
            location_is_folded: false,
        }
    }

    #[test]
    fn legacy_categorization_cases_match_the_python_reference() {
        let rules = ruby_rules(load_default_ruby_core_classes(), false).expect("ruby pack");
        let cases: &[(&str, &str, Option<&str>)] = &[
            ("String#gsub", "<cfunc>", Some("Ruby Core (Native)")),
            ("foo", "<internal:kernel>", Some("Ruby Internals")),
            (
                "Enumerable#index_by",
                "/usr/local/lib/ruby/3.2.0/enumerable.rb",
                Some("Ruby Stdlib"),
            ),
            ("Digest::MD5#digest", "<cfunc>", Some("Digest (Native)")),
            (
                "Zlib::Deflate.deflate",
                "<cfunc>",
                Some("Compression (Native)"),
            ),
            ("Zlib.inflate", "<cfunc>", Some("Compression (Native)")),
            (
                "OpenSSL.fixed_length_secure_compare",
                "<cfunc>",
                Some("OpenSSL (Native)"),
            ),
            (
                "OpenSSL::Cipher#encrypt",
                "<cfunc>",
                Some("OpenSSL (Native)"),
            ),
            ("JSON::parse", "<cfunc>", Some("JSON (Native)")),
            ("StatsD.increment", "<cfunc>", Some("StatsD (Native)")),
            (
                "StatsD.distribution",
                "/usr/local/lib/ruby/3.2.0/forwardable.rb",
                Some("StatsD Gem"),
            ),
            ("I18n.translate", "/gems/i18n/lib/i18n.rb", Some("I18n Gem")),
            ("Net::HTTP#get", "<cfunc>", None),
            ("Foo::String#process", "/app/lib/foo.rb", None),
            (
                "MyStatsDHelper#emit",
                "/srv/app/lib/my_statsd_helper.rb",
                None,
            ),
            ("Trilogy#query", "<cfunc>", Some("Trilogy (Native)")),
            ("::Array#map", "<cfunc>", Some("Ruby Core (Native)")),
        ];
        for (name, filename, expected) in cases {
            let actual = categorize_ruby_case(name, filename, &rules);
            assert_eq!(actual.as_deref(), *expected, "{name} @ {filename}");
        }
    }

    fn categorize_ruby_case(name: &str, filename: &str, rules: &RuntimeRuleSet) -> Option<String> {
        categorize_runtime_frame(&frame(name, filename), rules)
    }
}
