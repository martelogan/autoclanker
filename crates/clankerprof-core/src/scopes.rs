//! Scope (boundary) decomposition, mirroring `clankerprof/scopes.py` plus the
//! config loading from the Python CLI and the rendering from
//! `clankerprof/render.py`.

use crate::categorize::{categorize_stack, simplify_category, RuntimeCategoryCache};
use crate::model::{
    CategoryStats, DomainStats, Frame, ProfileFacts, TimeNs, AGGREGATE_BOUNDS_ERROR, AGGREGATE_MAX,
    AGGREGATE_MIN,
};
use crate::rules::RuntimeRuleSet;
use crate::slices::SliceDefinition;
use crate::targets::{
    extract_library_name, is_runtime_stdlib_path, match_path_pattern, match_regex,
    DEFAULT_LIBRARY_SELECTORS,
};
use indexmap::IndexMap;
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

thread_local! {
    // Python raises on the first non-finite attributable estimate it builds;
    // the render helpers here return plain `Value`s, so instead of threading
    // Results through every nested `json!` the first offender is parked in
    // this slot and the CLI fails closed (exit 2) before emitting or writing
    // any artifact — mirroring the pattern-error slot in `targets.rs`.
    static ATTRIBUTABLE_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn record_attributable_error(message: String) {
    ATTRIBUTABLE_ERROR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(message);
        }
    });
}

/// Fail closed on any non-finite attributable estimate recorded during
/// rendering. Must be checked before emitting or writing any artifact.
pub fn take_attributable_error() -> Result<(), String> {
    ATTRIBUTABLE_ERROR
        .with(|slot| slot.borrow_mut().take())
        .map_or(Ok(()), Err)
}

/// Attributable estimates — input or scaled — must stay JSON-representable;
/// a non-finite value records a fail-closed error naming the metric (the
/// placeholder null below is never emitted because the runner errors first).
fn finite_attributable(name: &str, estimate: f64) -> Value {
    if !estimate.is_finite() {
        record_attributable_error(format!("Attributable estimate for '{name}' is not finite."));
    }
    json!(estimate)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FramePredicate {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FramePredicateExpr {
    pub predicates: Vec<FramePredicate>,
    pub any: Vec<FramePredicateExpr>,
    pub all: Vec<FramePredicateExpr>,
    pub not_: Vec<FramePredicateExpr>,
}

impl FramePredicateExpr {
    fn has_terms(&self) -> bool {
        !self.predicates.is_empty()
            || !self.any.is_empty()
            || !self.all.is_empty()
            || !self.not_.is_empty()
    }

    fn leaf_predicates(&self) -> Vec<&FramePredicate> {
        let mut result: Vec<&FramePredicate> = self.predicates.iter().collect();
        for child in self.any.iter().chain(&self.all).chain(&self.not_) {
            result.extend(child.leaf_predicates());
        }
        result
    }
}

pub fn parse_frame_predicate(raw: &str, default_key: &str) -> Result<FramePredicate, String> {
    if raw == "native" {
        return Ok(FramePredicate {
            key: "native".to_string(),
            value: String::new(),
        });
    }
    let Some((key, value)) = raw.split_once(':') else {
        return Ok(FramePredicate {
            key: default_key.to_string(),
            value: raw.to_string(),
        });
    };
    if key.is_empty() {
        return Err(format!("Predicate key cannot be empty: {raw}"));
    }
    if key == "native" {
        if !matches!(value, "" | "true" | "false") {
            return Err(format!(
                "native: predicate value must be true or false, got {value:?}."
            ));
        }
        return Ok(FramePredicate {
            key: "native".to_string(),
            value: value.to_string(),
        });
    }
    if value.is_empty() {
        return Err(format!("Predicate value cannot be empty: {raw}"));
    }
    let resolved_key = match key {
        "cost_kind" => "category",
        "runtime_label" => "runtime_category",
        other => other,
    };
    Ok(FramePredicate {
        key: resolved_key.to_string(),
        value: value.to_string(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCategoryDefinition {
    pub name: String,
    pub predicates: FramePredicateExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryDomainDefinition {
    pub name: String,
    pub predicates: FramePredicateExpr,
    pub fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryDefinition {
    pub name: String,
    pub predicates: FramePredicateExpr,
    pub buckets: IndexMap<String, Vec<String>>,
    pub attributables: IndexMap<String, f64>,
    pub exclude_descendants: FramePredicateExpr,
    pub once_per_sample: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryAnalysisOptions {
    pub boundaries: Vec<BoundaryDefinition>,
    pub categories: Vec<BoundaryCategoryDefinition>,
    pub domains: Vec<BoundaryDomainDefinition>,
    pub slices: Vec<SliceDefinition>,
    pub runtime_rules: RuntimeRuleSet,
    pub enhanced_runtime_categorization: bool,
    pub fold_runtime_internals: bool,
    pub caller_fallback_when_uncategorized: bool,
}

#[derive(Debug, Default)]
pub struct BoundaryStats {
    pub name: String,
    pub buckets: IndexMap<String, Vec<String>>,
    pub attributables: IndexMap<String, f64>,
    pub total_time: TimeNs,
    pub sample_count: usize,
    pub categories: IndexMap<String, CategoryStats>,
    pub domains: IndexMap<String, DomainStats>,
}

#[derive(Debug)]
pub struct BoundaryAnalysisResult {
    pub total_time_ns: TimeNs,
    pub boundaries: Vec<BoundaryStats>,
    pub unique_frame_count: usize,
}

struct PredicateMatcher<'a> {
    rules: &'a RuntimeRuleSet,
    slices: &'a [SliceDefinition],
    runtime_cache: RuntimeCategoryCache,
    category_matcher: Option<CategoryMatcher>,
    cache: HashMap<(FramePredicate, (u64, String, String)), bool>,
}

fn frame_cache_key(frame: &Frame) -> (u64, String, String) {
    (
        frame.function_id,
        frame.name.clone(),
        frame.filename.clone(),
    )
}

impl<'a> PredicateMatcher<'a> {
    fn new(rules: &'a RuntimeRuleSet, slices: &'a [SliceDefinition]) -> Self {
        Self {
            rules,
            slices,
            runtime_cache: RuntimeCategoryCache::default(),
            category_matcher: None,
            cache: HashMap::new(),
        }
    }

    fn unique_frame_count(&self) -> usize {
        self.cache
            .keys()
            .map(|(_, frame_key)| frame_key)
            .collect::<HashSet<_>>()
            .len()
    }

    fn matches_expr(&mut self, frame: &Frame, expr: &FramePredicateExpr) -> Result<bool, String> {
        if !expr.has_terms() {
            return Ok(false);
        }
        if !expr.predicates.is_empty() {
            let mut matched = false;
            for predicate in expr.predicates.clone() {
                if self.matches(frame, &predicate)? {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Ok(false);
            }
        }
        if !expr.any.is_empty() {
            let mut matched = false;
            for child in expr.any.clone() {
                if self.matches_expr(frame, &child)? {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return Ok(false);
            }
        }
        for child in expr.all.clone() {
            if !self.matches_expr(frame, &child)? {
                return Ok(false);
            }
        }
        for child in expr.not_.clone() {
            if self.matches_expr(frame, &child)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn matches(&mut self, frame: &Frame, predicate: &FramePredicate) -> Result<bool, String> {
        let key = (predicate.clone(), frame_cache_key(frame));
        if let Some(cached) = self.cache.get(&key) {
            return Ok(*cached);
        }
        let result = self.matches_uncached(frame, predicate)?;
        self.cache.insert(key, result);
        Ok(result)
    }

    fn matches_uncached(
        &mut self,
        frame: &Frame,
        predicate: &FramePredicate,
    ) -> Result<bool, String> {
        let key = predicate.key.as_str();
        let value = predicate.value.as_str();
        match key {
            "name" => Ok(frame.name.contains(value)),
            "name_eq" => Ok(frame.name == value),
            "path" | "glob" => Ok(match_path_pattern(value, &frame.filename, self.rules)),
            "regex" => Ok(match_regex(value, &frame.filename)),
            "native" => {
                let is_native = crate::categorize::is_native_path(&frame.filename, self.rules);
                Ok(if value == "false" {
                    !is_native
                } else {
                    is_native
                })
            }
            "category" => {
                let Some(matcher) = self.category_matcher.take() else {
                    return Err("category: predicates require configured categories.".to_string());
                };
                let mut matcher = matcher;
                let result = matcher.category_for(frame, self)?;
                self.category_matcher = Some(matcher);
                Ok(result.as_deref() == Some(value))
            }
            "runtime_category" => {
                let category = self.runtime_cache.category_for(frame, self.rules);
                if category.as_deref() == Some(value) {
                    return Ok(true);
                }
                Ok(category
                    .map(|resolved| {
                        simplify_category(&resolved, self.rules.verbose, self.rules) == value
                    })
                    .unwrap_or(false))
            }
            "slice" => Ok(self.slices.iter().any(|definition| {
                definition.name == value && definition.matches_frame(frame, self.rules)
            })),
            _ if DEFAULT_LIBRARY_SELECTORS.contains(&key)
                || self.rules.library_selector_path_patterns.contains_key(key) =>
            {
                let library_name = extract_library_name(&frame.filename, self.rules, Some(key));
                Ok(if value == "*" {
                    library_name.is_some()
                } else {
                    library_name.as_deref() == Some(value)
                })
            }
            _ => Err(format!("Unsupported predicate key: {key}")),
        }
    }
}

#[derive(Debug, Clone)]
struct CategoryMatcher {
    definitions: Vec<BoundaryCategoryDefinition>,
    cache: HashMap<(u64, String, String), Option<String>>,
}

impl CategoryMatcher {
    fn new(definitions: Vec<BoundaryCategoryDefinition>) -> Self {
        Self {
            definitions,
            cache: HashMap::new(),
        }
    }

    fn category_for(
        &mut self,
        frame: &Frame,
        matcher: &mut PredicateMatcher<'_>,
    ) -> Result<Option<String>, String> {
        let key = frame_cache_key(frame);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }
        let mut resolved: Option<String> = None;
        for definition in self.definitions.clone() {
            if matcher.matches_expr(frame, &definition.predicates)? {
                resolved = Some(definition.name.clone());
                break;
            }
        }
        self.cache.insert(key, resolved.clone());
        Ok(resolved)
    }
}

fn validate_boundary_options(options: &BoundaryAnalysisOptions) -> Result<(), String> {
    for definition in &options.categories {
        if definition
            .predicates
            .leaf_predicates()
            .iter()
            .any(|predicate| predicate.key == "category")
        {
            return Err(format!(
                "Boundary category definitions cannot reference category: predicates: {}",
                definition.name
            ));
        }
    }
    Ok(())
}

fn is_non_runtime_file(frame: &Frame, rules: &RuntimeRuleSet) -> bool {
    !frame.filename.starts_with('<') && !is_runtime_stdlib_path(&frame.filename, rules)
}

pub fn analyze_boundary_facts(
    facts: &ProfileFacts,
    options: &BoundaryAnalysisOptions,
) -> Result<BoundaryAnalysisResult, String> {
    if options.boundaries.is_empty() {
        return Err("Boundary analysis requires at least one boundary.".to_string());
    }
    validate_boundary_options(options)?;

    let mut results: Vec<BoundaryStats> = options
        .boundaries
        .iter()
        .map(|boundary| BoundaryStats {
            name: boundary.name.clone(),
            buckets: boundary.buckets.clone(),
            attributables: boundary.attributables.clone(),
            ..BoundaryStats::default()
        })
        .collect();
    // Occurrence-mode attribution counts a sample once per matching frame
    // occurrence, so scope aggregates are NOT subset sums and can escape the
    // import-time bound. Re-enforce it during accumulation: the per-boundary
    // positive/negative occurrence sums cap every subordinate accumulator
    // (category, leaf, caller-pair, owner), so checking them here keeps all
    // rendered aggregates inside [AGGREGATE_MIN, AGGREGATE_MAX].
    let mut positive_occurrence: Vec<TimeNs> = vec![0; results.len()];
    let mut negative_occurrence: Vec<TimeNs> = vec![0; results.len()];
    let mut runtime_cache = RuntimeCategoryCache::default();
    let mut predicate_matcher = PredicateMatcher::new(&options.runtime_rules, &options.slices);
    predicate_matcher.category_matcher = Some(CategoryMatcher::new(options.categories.clone()));
    let mut domain_cache: HashMap<(u64, String, String), Option<String>> = HashMap::new();
    let domain_fallbacks: HashMap<&str, bool> = options
        .domains
        .iter()
        .map(|definition| (definition.name.as_str(), definition.fallback))
        .collect();
    let exclusion_exprs: Vec<(usize, &FramePredicateExpr)> = options
        .boundaries
        .iter()
        .enumerate()
        .filter(|(_, boundary)| boundary.exclude_descendants.has_terms())
        .map(|(index, boundary)| (index, &boundary.exclude_descendants))
        .collect();

    for fact in facts.samples.iter().filter(|fact| !fact.is_empty()) {
        let value = fact.primary_value();
        let stack = fact.stack.as_slice();
        let leaf = &stack[0];
        let mut category_error: Option<String> = None;
        let outcome = {
            let matcher_ptr: *mut PredicateMatcher<'_> = &mut predicate_matcher;
            let mut configured_category_for = |frame: &Frame| -> Option<String> {
                // SAFETY: categorize_stack only calls this closure while the
                // matcher is otherwise unused; the raw pointer sidesteps the
                // borrow of predicate_matcher held by the closure itself.
                let matcher = unsafe { &mut *matcher_ptr };
                let Some(mut category_matcher) = matcher.category_matcher.take() else {
                    return None;
                };
                let result = category_matcher.category_for(frame, matcher);
                matcher.category_matcher = Some(category_matcher);
                match result {
                    Ok(value) => value,
                    Err(error) => {
                        // Propagated after categorize_stack returns: swallowing
                        // predicate errors here silently rebuckets all cost as
                        // "Other" while Python fails closed.
                        category_error.get_or_insert(error);
                        None
                    }
                }
            };
            categorize_stack(
                stack,
                &options.runtime_rules,
                options.enhanced_runtime_categorization,
                options.fold_runtime_internals,
                options.caller_fallback_when_uncategorized,
                &mut runtime_cache,
                &mut configured_category_for,
            )
        };
        if let Some(error) = category_error {
            return Err(error);
        }
        let category = outcome.category;
        let frame_to_categorize = &stack[outcome.frame_index];

        let mut domain_owner: Option<(String, usize)> = None;
        let mut fallback_owner: Option<usize> = None;
        let mut first_non_runtime_caller_below: Option<usize> = None;
        let mut excluded_boundaries: HashSet<usize> = HashSet::new();
        let mut counted_once_boundaries: HashSet<usize> = HashSet::new();

        for (position, frame) in stack.iter().enumerate() {
            let owner_before_boundary = domain_owner.clone();
            let fallback_before_boundary = fallback_owner;
            let caller_below_boundary = if position == 0 {
                None
            } else {
                Some(first_non_runtime_caller_below.map_or(leaf, |index| &stack[index]))
            };

            for (boundary_index, boundary) in options.boundaries.iter().enumerate() {
                if excluded_boundaries.contains(&boundary_index) {
                    continue;
                }
                if boundary.once_per_sample && counted_once_boundaries.contains(&boundary_index) {
                    continue;
                }
                if !predicate_matcher.matches_expr(frame, &boundary.predicates)? {
                    continue;
                }

                let boundary_stats = &mut results[boundary_index];
                boundary_stats.total_time += value;
                boundary_stats.sample_count += 1;
                if value >= 0 {
                    positive_occurrence[boundary_index] += value;
                    if positive_occurrence[boundary_index] > AGGREGATE_MAX {
                        return Err(AGGREGATE_BOUNDS_ERROR.to_string());
                    }
                } else {
                    negative_occurrence[boundary_index] += value;
                    if negative_occurrence[boundary_index] < AGGREGATE_MIN {
                        return Err(AGGREGATE_BOUNDS_ERROR.to_string());
                    }
                }
                if boundary.once_per_sample {
                    counted_once_boundaries.insert(boundary_index);
                }

                let category_stats = boundary_stats
                    .categories
                    .entry(category.clone())
                    .or_default();
                category_stats.cpu_time += value;
                category_stats.sample_count += 1;
                category_stats.add_function(&leaf.name, value);
                category_stats
                    .files
                    .insert(frame_to_categorize.filename.clone());

                let caller = caller_below_boundary.or_else(|| stack.get(1));
                if let Some(caller) = caller {
                    category_stats.add_caller_leaf_pair(&caller.name, &leaf.name, value);
                }
                if outcome.folded && outcome.folded_category.is_some() {
                    *category_stats
                        .folded_from
                        .entry(leaf.name.clone())
                        .or_insert(0) += value;
                }

                if !options.domains.is_empty() {
                    let (domain_name, owner): (String, &Frame) = match &owner_before_boundary {
                        Some((name, owner_index)) => (name.clone(), &stack[*owner_index]),
                        None => (
                            "Uncategorized".to_string(),
                            fallback_before_boundary.map_or(leaf, |index| &stack[index]),
                        ),
                    };
                    let domain_stats = boundary_stats.domains.entry(domain_name).or_default();
                    domain_stats.add(owner, leaf, &category, value);
                }
            }

            let frame_domain = {
                let key = frame_cache_key(frame);
                if let Some(cached) = domain_cache.get(&key) {
                    cached.clone()
                } else {
                    let mut resolved: Option<String> = None;
                    for definition in &options.domains {
                        if predicate_matcher.matches_expr(frame, &definition.predicates)? {
                            resolved = Some(definition.name.clone());
                            break;
                        }
                    }
                    domain_cache.insert(key, resolved.clone());
                    resolved
                }
            };
            if let Some(frame_domain) = frame_domain {
                domain_owner = match domain_owner {
                    None => Some((frame_domain, position)),
                    Some((current_domain, current_position)) => {
                        let current_is_fallback = domain_fallbacks
                            .get(current_domain.as_str())
                            .copied()
                            .unwrap_or(false);
                        let new_is_fallback = domain_fallbacks
                            .get(frame_domain.as_str())
                            .copied()
                            .unwrap_or(false);
                        if current_is_fallback && !new_is_fallback {
                            Some((frame_domain, position))
                        } else {
                            Some((current_domain, current_position))
                        }
                    }
                };
            }
            if position > 0
                && first_non_runtime_caller_below.is_none()
                && is_non_runtime_file(frame, &options.runtime_rules)
            {
                first_non_runtime_caller_below = Some(position);
            }
            if position > 0 && fallback_owner.is_none() {
                fallback_owner = Some(position);
            }
            for (boundary_index, exclude_expr) in &exclusion_exprs {
                if excluded_boundaries.contains(boundary_index) {
                    continue;
                }
                if predicate_matcher.matches_expr(frame, exclude_expr)? {
                    excluded_boundaries.insert(*boundary_index);
                }
            }
        }
    }

    Ok(BoundaryAnalysisResult {
        total_time_ns: facts.total_primary_value,
        boundaries: results,
        unique_frame_count: predicate_matcher.unique_frame_count(),
    })
}

fn scale_attributables(
    attributables: &IndexMap<String, f64>,
    total: TimeNs,
    value: TimeNs,
) -> Value {
    let mut scaled = Map::new();
    if attributables.is_empty() || total <= 0 {
        return Value::Object(scaled);
    }
    for (name, metric_value) in attributables {
        scaled.insert(
            name.clone(),
            finite_attributable(name, metric_value * (value as f64 / total as f64)),
        );
    }
    Value::Object(scaled)
}

/// Zero-total percentages serialize as integer `0`, matching Python's
/// `... if total else 0` arms byte-for-byte.
fn pct(value: TimeNs, total: TimeNs) -> Value {
    if total == 0 {
        json!(0)
    } else {
        json!(value as f64 / total as f64 * 100.0)
    }
}

pub fn render_boundary_json(result: &BoundaryAnalysisResult, top: Option<i64>) -> Value {
    json!({
        "boundaries": result
            .boundaries
            .iter()
            .map(|boundary| render_boundary(boundary, result.total_time_ns, top))
            .collect::<Vec<_>>(),
        "summary": {
            "total_time_ns": result.total_time_ns,
            "unique_frame_count": result.unique_frame_count,
        },
        "tool": "clankerprof_boundaries",
    })
}

fn truncated<T>(mut items: Vec<T>, top: Option<i64>) -> Vec<T> {
    // Python renders ranked sections with list[:top], so a negative limit
    // drops entries from the tail instead of rejecting.
    crate::slices::apply_python_limit(&mut items, top);
    items
}

/// Rendered totals may exceed i64 (aggregate bound allows up to u64::MAX),
/// so read them back through both integer views before widening.
fn time_ns_of(value: &Value) -> TimeNs {
    value
        .as_i64()
        .map(TimeNs::from)
        .or_else(|| value.as_u64().map(TimeNs::from))
        .unwrap_or(0)
}

fn render_boundary(boundary: &BoundaryStats, profile_total: TimeNs, top: Option<i64>) -> Value {
    let total = boundary.total_time;
    let bucketed: HashSet<&str> = boundary
        .buckets
        .values()
        .flatten()
        .map(String::as_str)
        .collect();
    let mut buckets: Vec<Value> = boundary
        .buckets
        .iter()
        .map(|(label, categories)| render_bucket(boundary, label, categories))
        .collect();
    let mut leftover_sorted: Vec<(&String, &CategoryStats)> = boundary.categories.iter().collect();
    leftover_sorted.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    let leftover: Vec<String> = leftover_sorted
        .into_iter()
        .filter(|(category, stats)| !bucketed.contains(category.as_str()) && stats.cpu_time != 0)
        .map(|(category, _)| category.clone())
        .collect();
    if !leftover.is_empty() {
        buckets.push(render_bucket(boundary, "Other", &leftover));
    }
    let mut buckets: Vec<Value> = buckets
        .into_iter()
        .filter(|bucket| time_ns_of(&bucket["time_ns"]) != 0)
        .collect();
    buckets.sort_by_key(|bucket| std::cmp::Reverse(time_ns_of(&bucket["time_ns"])));

    let mut domains_sorted: Vec<(&String, &DomainStats)> = boundary.domains.iter().collect();
    domains_sorted.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    let domains: Vec<Value> = truncated(
        domains_sorted
            .into_iter()
            .filter(|(_, stats)| stats.cpu_time != 0)
            .map(|(name, stats)| render_domain(boundary, name, stats, top))
            .collect(),
        top,
    );

    let attributable_estimates: Map<String, Value> = boundary
        .attributables
        .iter()
        .map(|(name, value)| (name.clone(), finite_attributable(name, *value)))
        .collect();

    json!({
        "attributable_estimates": attributable_estimates,
        "buckets": buckets,
        "domains": domains,
        "name": boundary.name,
        "pct_of_profile": pct(total, profile_total),
        "samples": boundary.sample_count,
        "total_time_ns": total,
    })
}

fn render_bucket(boundary: &BoundaryStats, label: &str, categories: &[String]) -> Value {
    let total = boundary.total_time;
    let category_rows: Vec<Value> = categories
        .iter()
        .filter_map(|category| {
            boundary
                .categories
                .get(category)
                .filter(|stats| stats.cpu_time != 0)
                .map(|stats| render_category(boundary, category, stats))
        })
        .collect();
    let cpu_time: TimeNs = category_rows
        .iter()
        .map(|row| time_ns_of(&row["time_ns"]))
        .sum();
    let samples: u64 = category_rows
        .iter()
        .map(|row| row["samples"].as_u64().unwrap_or(0))
        .sum();
    json!({
        "attributable_estimates": scale_attributables(&boundary.attributables, total, cpu_time),
        "categories": category_rows,
        "name": label,
        "pct": pct(cpu_time, total),
        "samples": samples,
        "time_ns": cpu_time,
    })
}

fn render_metrics_object<'a>(
    entries: impl Iterator<Item = (&'a String, &'a crate::model::FunctionMetrics)>,
) -> Value {
    let mut object = Map::new();
    for (name, metrics) in entries {
        object.insert(
            name.clone(),
            json!({"count": metrics.count, "cpu_time": metrics.cpu_time}),
        );
    }
    Value::Object(object)
}

fn render_category(boundary: &BoundaryStats, category: &str, stats: &CategoryStats) -> Value {
    let total = boundary.total_time;
    let mut pairs_sorted: Vec<(&String, &crate::model::CallerMetrics)> =
        stats.caller_leaf_pairs.iter().collect();
    pairs_sorted.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    json!({
        "attributable_estimates": scale_attributables(&boundary.attributables, total, stats.cpu_time),
        "caller_leaf_pairs": pairs_sorted.into_iter().map(|(pair, metrics)| json!({
            "attributable_estimates": scale_attributables(&boundary.attributables, total, metrics.cpu_time),
            "pair": pair,
            "pct": pct(metrics.cpu_time, total),
            "samples": metrics.count,
            "time_ns": metrics.cpu_time,
        })).collect::<Vec<_>>(),
        "files": stats.files.iter().cloned().collect::<Vec<_>>(),
        "folded_from": stats.folded_from,
        "leaf_functions": render_metrics_object(stats.functions.iter()),
        "name": category,
        "pct": pct(stats.cpu_time, total),
        "samples": stats.sample_count,
        "time_ns": stats.cpu_time,
    })
}

fn render_domain(
    boundary: &BoundaryStats,
    name: &str,
    stats: &DomainStats,
    top: Option<i64>,
) -> Value {
    let total = boundary.total_time;
    let mut cost_kinds_sorted: Vec<_> = stats.cost_kinds.iter().collect();
    cost_kinds_sorted.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    let mut files_sorted: Vec<_> = stats.files.values().collect();
    files_sorted.sort_by(|left, right| right.cpu_time.cmp(&left.cpu_time));
    json!({
        "attributable_estimates": scale_attributables(&boundary.attributables, total, stats.cpu_time),
        "cost_kinds": truncated(cost_kinds_sorted.into_iter().map(|(cost_kind, metrics)| json!({
            "attributable_estimates": scale_attributables(&boundary.attributables, total, metrics.cpu_time),
            "name": cost_kind,
            "pct_of_boundary": pct(metrics.cpu_time, total),
            "pct_of_domain": pct(metrics.cpu_time, stats.cpu_time),
            "samples": metrics.count,
            "time_ns": metrics.cpu_time,
        })).collect(), top),
        "files": truncated(files_sorted.into_iter().map(|file_stats| {
            let mut functions_sorted: Vec<_> = file_stats.functions.iter().collect();
            functions_sorted.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
            let mut file_cost_kinds: Vec<_> = file_stats.cost_kinds.iter().collect();
            file_cost_kinds.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
            let mut file_pairs: Vec<_> = file_stats.caller_leaf_pairs.iter().collect();
            file_pairs.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
            json!({
                "attributable_estimates": scale_attributables(&boundary.attributables, total, file_stats.cpu_time),
                "caller_leaf_pairs": truncated(file_pairs.into_iter().map(|((caller, leaf), metrics)| json!({
                    "caller": caller,
                    "leaf": leaf,
                    "pct_of_file": pct(metrics.cpu_time, file_stats.cpu_time),
                    "samples": metrics.count,
                    "time_ns": metrics.cpu_time,
                })).collect(), top),
                "cost_kinds": truncated(file_cost_kinds.into_iter().map(|(cost_kind, metrics)| json!({
                    "name": cost_kind,
                    "pct_of_file": pct(metrics.cpu_time, file_stats.cpu_time),
                    "samples": metrics.count,
                    "time_ns": metrics.cpu_time,
                })).collect(), top),
                "filename": file_stats.filename,
                "functions": render_metrics_object(truncated(functions_sorted, top).into_iter()),
                "pct_of_boundary": pct(file_stats.cpu_time, total),
                "pct_of_domain": pct(file_stats.cpu_time, stats.cpu_time),
                "samples": file_stats.sample_count,
                "time_ns": file_stats.cpu_time,
            })
        }).collect(), top),
        "name": name,
        "pct": pct(stats.cpu_time, total),
        "samples": stats.sample_count,
        "time_ns": stats.cpu_time,
    })
}

// ---------------------------------------------------------------------------
// Config loading, mirroring the Python CLI's boundary config parsing.

/// Top-level sections whose table KEY ORDER is semantic: configured cost
/// kinds and owners evaluate first-match-wins in declaration order.
const ORDERED_CONFIG_SECTIONS: [&str; 4] = ["cost_kind", "category", "owner", "domain"];

/// Parse the config into a serde_json tree plus the declaration order of the
/// order-sensitive tables. serde_json's Map re-sorts keys alphabetically, so
/// the order is captured from the order-preserving YAML/TOML values before
/// conversion (toml is built with preserve_order for exactly this reason).
fn config_to_json(path: &Path) -> Result<(Value, HashMap<String, Vec<String>>), String> {
    let payload = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut order: HashMap<String, Vec<String>> = HashMap::new();
    let value = if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
        let value: toml::Value = toml::from_str(&payload).map_err(|error| error.to_string())?;
        if let toml::Value::Table(table) = &value {
            for section in ORDERED_CONFIG_SECTIONS {
                if let Some(toml::Value::Table(inner)) = table.get(section) {
                    order.insert(section.to_string(), inner.keys().cloned().collect());
                }
            }
        }
        serde_json::to_value(value).map_err(|error| error.to_string())?
    } else {
        let value: serde_yaml::Value =
            serde_yaml::from_str(&payload).map_err(|error| error.to_string())?;
        crate::rules::require_string_keys(&value)?;
        if let serde_yaml::Value::Mapping(mapping) = &value {
            for section in ORDERED_CONFIG_SECTIONS {
                let key = serde_yaml::Value::String(section.to_string());
                if let Some(serde_yaml::Value::Mapping(inner)) = mapping.get(&key) {
                    order.insert(
                        section.to_string(),
                        inner
                            .keys()
                            .filter_map(|item| item.as_str().map(String::from))
                            .collect(),
                    );
                }
            }
        }
        serde_json::to_value(value).map_err(|error| error.to_string())?
    };
    Ok((value, order))
}

/// Iterate a config table's entries in declaration order.
fn ordered_entries<'a>(
    table: &'a Map<String, Value>,
    order: Option<&Vec<String>>,
) -> Vec<(&'a String, &'a Value)> {
    match order {
        Some(names) => names
            .iter()
            .filter_map(|name| table.get_key_value(name))
            .collect(),
        None => table.iter().collect(),
    }
}

fn string_values(value: &Value, field_name: &str) -> Result<Vec<String>, String> {
    let message = || format!("{field_name} must be a string or array of strings.");
    match value {
        Value::String(text) => Ok(vec![text.clone()]),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                // Python str() and serde's Display disagree on bool/number
                // spellings, so non-string items fail closed in both.
                item.as_str().map(String::from).ok_or_else(message)
            })
            .collect(),
        _ => Err(message()),
    }
}

fn predicate_expr(
    value: &Value,
    field_name: &str,
    default_key: &str,
) -> Result<FramePredicateExpr, String> {
    match value {
        Value::String(text) => Ok(FramePredicateExpr {
            predicates: vec![parse_frame_predicate(text, default_key)?],
            ..FramePredicateExpr::default()
        }),
        Value::Array(_) => {
            let values = string_values(value, field_name)?;
            let mut predicates = Vec::new();
            for raw in values {
                predicates.push(parse_frame_predicate(&raw, default_key)?);
            }
            Ok(FramePredicateExpr {
                predicates,
                ..FramePredicateExpr::default()
            })
        }
        Value::Object(mapping) => {
            let allowed = ["patterns", "match", "selector", "any", "all", "not"];
            let mut unknown: Vec<String> = mapping
                .keys()
                .filter(|key| !allowed.contains(&key.as_str()))
                .cloned()
                .collect();
            if !unknown.is_empty() {
                unknown.sort();
                return Err(format!(
                    "{field_name} contains unsupported predicate keys: {}.",
                    unknown.join(", ")
                ));
            }
            let selector_keys: Vec<&str> = ["patterns", "match", "selector"]
                .into_iter()
                .filter(|key| mapping.contains_key(*key))
                .collect();
            if selector_keys.len() > 1 {
                return Err(format!(
                    "{field_name} must use only one of patterns, match, or selector."
                ));
            }
            let mut predicates = Vec::new();
            for key in ["patterns", "match", "selector"] {
                if let Some(raw) = mapping.get(key) {
                    for value in string_values(raw, &format!("{field_name}.{key}"))? {
                        predicates.push(parse_frame_predicate(&value, default_key)?);
                    }
                }
            }
            let children = |key: &str| -> Result<Vec<FramePredicateExpr>, String> {
                match mapping.get(key) {
                    None => Ok(Vec::new()),
                    Some(raw) => {
                        predicate_expr_children(raw, &format!("{field_name}.{key}"), default_key)
                    }
                }
            };
            let any = children("any")?;
            let all = children("all")?;
            let not_ = children("not")?;
            if predicates.is_empty() && any.is_empty() && all.is_empty() && not_.is_empty() {
                return Err(format!("{field_name} predicate table cannot be empty."));
            }
            Ok(FramePredicateExpr {
                predicates,
                any,
                all,
                not_,
            })
        }
        _ => Err(format!(
            "{field_name} must be a string, array, or predicate table."
        )),
    }
}

fn predicate_expr_children(
    value: &Value,
    field_name: &str,
    default_key: &str,
) -> Result<Vec<FramePredicateExpr>, String> {
    match value {
        Value::String(text) => Ok(vec![FramePredicateExpr {
            predicates: vec![parse_frame_predicate(text, default_key)?],
            ..FramePredicateExpr::default()
        }]),
        Value::Array(items) => {
            let mut children = Vec::new();
            for (index, item) in items.iter().enumerate() {
                match item {
                    Value::String(text) => children.push(FramePredicateExpr {
                        predicates: vec![parse_frame_predicate(text, default_key)?],
                        ..FramePredicateExpr::default()
                    }),
                    Value::Object(_) => children.push(predicate_expr(
                        item,
                        &format!("{field_name}[{index}]"),
                        default_key,
                    )?),
                    _ => {
                        return Err(format!(
                            "{field_name}[{index}] must be a string or predicate table."
                        ));
                    }
                }
            }
            Ok(children)
        }
        Value::Object(_) => Ok(vec![predicate_expr(value, field_name, default_key)?]),
        _ => Err(format!(
            "{field_name} must be a string, array, or predicate table."
        )),
    }
}

fn aliased_config_value<'a>(
    payload: &'a Map<String, Value>,
    names: &[&str],
    description: &str,
) -> Result<(String, Option<&'a Value>), String> {
    let present: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| payload.contains_key(*name))
        .collect();
    if present.len() > 1 {
        let formatted = names
            .iter()
            .map(|name| format!("[{name}]"))
            .collect::<Vec<_>>()
            .join(" or ");
        return Err(format!("Use only one of {formatted} for {description}."));
    }
    match present.first() {
        None => Ok((names[0].to_string(), None)),
        Some(name) => Ok((name.to_string(), payload.get(*name))),
    }
}

fn load_boundary_categories(
    raw: Option<&Value>,
    section_name: &str,
    order: Option<&Vec<String>>,
) -> Result<Vec<BoundaryCategoryDefinition>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let Value::Object(table) = raw else {
        return Err(format!("[{section_name}] must be an object."));
    };
    let mut categories = Vec::new();
    for (name, raw_value) in ordered_entries(table, order) {
        categories.push(BoundaryCategoryDefinition {
            name: name.clone(),
            predicates: predicate_expr(raw_value, &format!("{section_name} {name}"), "path")?,
        });
    }
    Ok(categories)
}

fn load_boundary_domains(
    raw: Option<&Value>,
    section_name: &str,
    order: Option<&Vec<String>>,
) -> Result<Vec<BoundaryDomainDefinition>, String> {
    let Some(raw) = raw else {
        return Ok(Vec::new());
    };
    let Value::Object(table) = raw else {
        return Err(format!("[{section_name}] must be an object."));
    };
    let mut domains = Vec::new();
    for (name, raw_value) in ordered_entries(table, order) {
        let field_name = format!("{section_name} {name}");
        let (fallback, expression) = match raw_value {
            Value::Object(mapping) => {
                // Python evaluates `bool(fallback)` truthiness, so any present
                // value counts by shape — never silently coerced to false.
                let fallback = mapping.get("fallback").is_some_and(python_truthy);
                let mut stripped = mapping.clone();
                stripped.remove("fallback");
                (
                    fallback,
                    predicate_expr(&Value::Object(stripped), &field_name, "path")?,
                )
            }
            _ => (false, predicate_expr(raw_value, &field_name, "path")?),
        };
        domains.push(BoundaryDomainDefinition {
            name: name.clone(),
            predicates: expression,
            fallback,
        });
    }
    Ok(domains)
}

/// Python `bool()` truthiness over parsed JSON shapes (mirrors the reference
/// implementation's `bool(mapping.get("fallback", False))`).
fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(mapping) => !mapping.is_empty(),
    }
}

fn boundary_predicate_value(
    raw_boundary: &Map<String, Value>,
    section_name: &str,
) -> Result<Value, String> {
    for key in ["selector", "matcher", "match"] {
        if let Some(value) = raw_boundary.get(key) {
            return Ok(value.clone());
        }
    }
    if let Some(raw_function) = raw_boundary.get("function") {
        // Non-string function values fail closed: Python str() and Rust
        // Display disagree on bool/number spellings, so neither is trusted.
        let message = format!("{section_name}.function must be a string or array of strings.");
        return Ok(match raw_function {
            Value::Array(items) => {
                let mut predicates = Vec::new();
                for item in items {
                    let Some(text) = item.as_str() else {
                        return Err(message);
                    };
                    predicates.push(Value::String(format!("name_eq:{text}")));
                }
                Value::Array(predicates)
            }
            Value::String(text) => Value::String(format!("name_eq:{text}")),
            _ => return Err(message),
        });
    }
    Err("Each scope must define selector, matcher, match, or function.".to_string())
}

fn boundary_label(
    raw_boundary: &Map<String, Value>,
    raw_predicates: &Value,
    section_name: &str,
) -> Result<String, String> {
    for key in ["label", "name"] {
        match raw_boundary.get(key) {
            None | Some(Value::Null) => {}
            Some(Value::String(text)) => {
                // Python's or-chain skips falsy values, so an empty string
                // falls through to the next key.
                if !text.is_empty() {
                    return Ok(text.clone());
                }
            }
            Some(_) => {
                return Err(format!("{section_name}.{key} must be a string."));
            }
        }
    }
    match raw_boundary.get("function") {
        None | Some(Value::Null) => {}
        Some(Value::String(text)) if !text.is_empty() => return Ok(text.clone()),
        Some(Value::String(_)) => {}
        Some(Value::Array(items)) if !items.is_empty() => {
            // Python labels a multi-function scope with str(list); mirror
            // repr() for the validated array-of-strings shape.
            let rendered: Vec<String> = items
                .iter()
                .map(|item| python_str_repr(item.as_str().unwrap_or_default()))
                .collect();
            return Ok(format!("[{}]", rendered.join(", ")));
        }
        Some(Value::Array(_)) => {}
        // boundary_predicate_value already rejected other function shapes.
        Some(_) => {}
    }
    Ok(match raw_predicates {
        Value::String(text) => text.clone(),
        // Array entries are validated as strings before this fallback runs.
        Value::Array(items) => items
            .first()
            .and_then(Value::as_str)
            .map_or_else(|| "boundary".to_string(), String::from),
        _ => "boundary".to_string(),
    })
}

/// Python repr() for a plain string: single quotes unless the text contains a
/// single quote and no double quote.
fn python_str_repr(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{text}'")
    }
}

fn load_boundaries(
    raw: Option<&Value>,
    section_name: &str,
) -> Result<Vec<BoundaryDefinition>, String> {
    let Some(Value::Array(items)) = raw else {
        return Err(format!("Scope config must contain a {section_name} array."));
    };
    let mut boundaries = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for item in items {
        let Value::Object(raw_boundary) = item else {
            return Err("Each boundary entry must be an object.".to_string());
        };
        let raw_predicates = boundary_predicate_value(raw_boundary, section_name)?;
        // Validated before the label fallback, which derives the label from
        // the first entry: Python str() and serde Display disagree on
        // non-string spellings, so both implementations reject them here.
        if let Value::Array(entries) = &raw_predicates {
            if entries.iter().any(|entry| !entry.is_string()) {
                return Err(format!("{section_name} selector values must be strings."));
            }
        }
        let label = boundary_label(raw_boundary, &raw_predicates, section_name)?;
        if !seen.insert(label.clone()) {
            return Err(format!("Duplicate boundary label: {label}"));
        }
        // Non-string count values must fail like Python's str()-then-check,
        // never silently fall back to the occurrence default.
        let count = match raw_boundary.get("count") {
            None => "occurrence",
            Some(Value::String(text))
                if matches!(text.as_str(), "occurrence" | "once_per_sample") =>
            {
                text.as_str()
            }
            Some(_) => {
                return Err(format!(
                    "{section_name}.count must be occurrence or once_per_sample."
                ));
            }
        };
        let (rollup_name, raw_rollup) = aliased_config_value(
            raw_boundary,
            &["rollup", "bucket"],
            &format!("{section_name} {label} rollup"),
        )?;
        let mut buckets: IndexMap<String, Vec<String>> = IndexMap::new();
        if let Some(raw_rollup) = raw_rollup {
            let Value::Object(rollup_table) = raw_rollup else {
                return Err(format!("{section_name}.{rollup_name} must be an object."));
            };
            let mut category_to_bucket: HashMap<String, String> = HashMap::new();
            for (bucket_name, raw_categories) in rollup_table {
                let categories = string_values(
                    raw_categories,
                    &format!("{section_name}.{rollup_name} {bucket_name}"),
                )?;
                for category in &categories {
                    if let Some(existing) = category_to_bucket.get(category) {
                        return Err(format!(
                            "Category {category} appears in both {existing} and {bucket_name} buckets."
                        ));
                    }
                    category_to_bucket.insert(category.clone(), bucket_name.clone());
                }
                buckets.insert(bucket_name.clone(), categories);
            }
        }
        let mut attributables: IndexMap<String, f64> = IndexMap::new();
        if let Some(raw_attributables) = raw_boundary.get("attributables") {
            let Value::Object(table) = raw_attributables else {
                return Err(format!("{section_name}.attributables must be an object."));
            };
            for (name, raw_metric) in table {
                let metric = raw_metric
                    .as_f64()
                    .ok_or_else(|| format!("Boundary attributable {name} must be a number."))?;
                if metric < 0.0 {
                    return Err(format!("Boundary attributable {name} cannot be negative."));
                }
                attributables.insert(name.clone(), metric);
            }
        }
        let exclude = match raw_boundary.get("exclude_descendants") {
            None => FramePredicateExpr::default(),
            Some(raw_exclude) => predicate_expr(
                raw_exclude,
                &format!("{section_name} {label} exclude_descendants"),
                "name_eq",
            )?,
        };
        boundaries.push(BoundaryDefinition {
            name: label.clone(),
            predicates: predicate_expr(
                &raw_predicates,
                &format!("{section_name} {label}"),
                "name_eq",
            )?,
            buckets,
            attributables,
            exclude_descendants: exclude,
            once_per_sample: count == "once_per_sample",
        });
    }
    Ok(boundaries)
}

pub fn load_boundary_options(
    path: &Path,
    runtime_rules: RuntimeRuleSet,
) -> Result<BoundaryAnalysisOptions, String> {
    let (payload, declaration_order) = config_to_json(path)?;
    let Value::Object(payload) = payload else {
        return Err("Boundary config file must be an object.".to_string());
    };
    let mut slices: Vec<SliceDefinition> = Vec::new();
    if let Some(slices_path) = payload.get("slices").and_then(Value::as_str) {
        let mut resolved = std::path::PathBuf::from(slices_path);
        if resolved.is_relative() {
            if let Some(parent) = path.parent() {
                resolved = parent.join(resolved);
            }
        }
        slices = crate::slices::load_slices_file(resolved)?;
    }
    let (category_name, raw_categories) = aliased_config_value(
        &payload,
        &["cost_kind", "category"],
        "cost-kind definitions",
    )?;
    let (domain_name, raw_domains) =
        aliased_config_value(&payload, &["owner", "domain"], "owner definitions")?;
    let (scope_name, raw_scopes) =
        aliased_config_value(&payload, &["scope", "boundary"], "scope definitions")?;
    let options = BoundaryAnalysisOptions {
        boundaries: load_boundaries(raw_scopes, &scope_name)?,
        categories: load_boundary_categories(
            raw_categories,
            &category_name,
            declaration_order.get(&category_name),
        )?,
        domains: load_boundary_domains(
            raw_domains,
            &domain_name,
            declaration_order.get(&domain_name),
        )?,
        slices,
        runtime_rules,
        enhanced_runtime_categorization: true,
        fold_runtime_internals: false,
        caller_fallback_when_uncategorized: false,
    };
    let uses_slice_predicates = options
        .categories
        .iter()
        .map(|definition| &definition.predicates)
        .chain(
            options
                .domains
                .iter()
                .map(|definition| &definition.predicates),
        )
        .chain(
            options
                .boundaries
                .iter()
                .map(|boundary| &boundary.predicates),
        )
        .chain(
            options
                .boundaries
                .iter()
                .map(|boundary| &boundary.exclude_descendants),
        )
        .flat_map(FramePredicateExpr::leaf_predicates)
        .any(|predicate| predicate.key == "slice");
    if uses_slice_predicates && options.slices.is_empty() {
        return Err("slice: predicates in boundary config require slices.".to_string());
    }
    Ok(options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_ns_read_back_handles_the_full_aggregate_range() {
        assert_eq!(time_ns_of(&json!(-5)), -5);
        assert_eq!(time_ns_of(&json!(i64::MAX)), TimeNs::from(i64::MAX));
        // Above i64::MAX serde_json stores u64; as_i64-only reads see zero.
        assert_eq!(
            time_ns_of(&json!(18_446_744_073_709_551_614_u64)),
            18_446_744_073_709_551_614_i128
        );
    }

    #[test]
    fn predicate_parser_matches_python_validation() {
        assert_eq!(
            parse_frame_predicate("native", "path").unwrap(),
            FramePredicate {
                key: "native".to_string(),
                value: String::new(),
            }
        );
        assert_eq!(
            parse_frame_predicate("cost_kind:IO", "path").unwrap().key,
            "category"
        );
        assert_eq!(
            parse_frame_predicate("runtime_label:Ruby Stdlib", "path")
                .unwrap()
                .key,
            "runtime_category"
        );
        assert_eq!(
            parse_frame_predicate("bare-value", "name_eq").unwrap(),
            FramePredicate {
                key: "name_eq".to_string(),
                value: "bare-value".to_string(),
            }
        );
        assert!(parse_frame_predicate("native:maybe", "path")
            .unwrap_err()
            .contains("must be true or false"));
        assert!(parse_frame_predicate(":oops", "path")
            .unwrap_err()
            .contains("key cannot be empty"));
        assert!(parse_frame_predicate("name:", "path")
            .unwrap_err()
            .contains("value cannot be empty"));
    }

    #[test]
    fn config_rejects_mixed_preferred_and_legacy_sections() {
        let config = r#"
[cost_kind]
"A" = "path:app/**"

[category]
"B" = "path:app/**"

[[scope]]
label = "x"
match = "name_eq:X"
"#;
        let path = std::env::temp_dir().join("clankerprof-mixed-sections-test.toml");
        std::fs::write(&path, config).unwrap();
        let error = load_boundary_options(&path, crate::rules::RuntimeRuleSet::generic().clone())
            .unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(
            error.contains("Use only one of [cost_kind] or [category]"),
            "{error}"
        );
    }

    #[test]
    fn boundary_count_mode_and_duplicate_labels_validate() {
        let bad_count = r#"
[[scope]]
label = "x"
match = "name_eq:X"
count = "twice"
"#;
        let path = std::env::temp_dir().join("clankerprof-count-test.toml");
        std::fs::write(&path, bad_count).unwrap();
        let error = load_boundary_options(&path, crate::rules::RuntimeRuleSet::generic().clone())
            .unwrap_err();
        std::fs::remove_file(&path).ok();
        assert!(error.contains("count must be occurrence or once_per_sample"));
    }

    #[test]
    fn empty_predicate_tables_and_non_string_counts_fail_closed() {
        let empty_table = r#"
cost_kind:
  Empty: {}
scope:
  - function: T
"#;
        let path = std::env::temp_dir().join("clankerprof-empty-table-test.yml");
        std::fs::write(&path, empty_table).unwrap();
        let error = load_boundary_options(&path, crate::rules::RuntimeRuleSet::generic().clone())
            .unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(error, "cost_kind Empty predicate table cannot be empty.");

        let int_count = r#"
scope:
  - function: T
    count: 1
"#;
        let path = std::env::temp_dir().join("clankerprof-int-count-test.yml");
        std::fs::write(&path, int_count).unwrap();
        let error = load_boundary_options(&path, crate::rules::RuntimeRuleSet::generic().clone())
            .unwrap_err();
        std::fs::remove_file(&path).ok();
        assert_eq!(error, "scope.count must be occurrence or once_per_sample.");
    }

    #[test]
    fn fallback_flags_follow_python_truthiness() {
        assert!(python_truthy(&json!(true)));
        assert!(python_truthy(&json!("yes")));
        assert!(python_truthy(&json!(1)));
        assert!(python_truthy(&json!([0])));
        assert!(!python_truthy(&json!(false)));
        assert!(!python_truthy(&json!(null)));
        assert!(!python_truthy(&json!(0)));
        assert!(!python_truthy(&json!(0.0)));
        assert!(!python_truthy(&json!("")));
        assert!(!python_truthy(&json!([])));
        assert!(!python_truthy(&json!({})));
    }
}
