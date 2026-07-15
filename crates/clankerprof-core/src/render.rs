//! CSV and text target renderers, mirroring `clankerprof/render.py`
//! byte-for-byte (Python `csv.writer` quoting, `format_time`, and the exact
//! summary/format strings).

use crate::model::{CallerMetrics, CategoryStats, TimeNs};
use crate::rules::RuntimeRuleSet;
use crate::targets::{extract_library_path, TargetResults};
use indexmap::IndexMap;

pub fn format_time(nanoseconds: TimeNs) -> String {
    let milliseconds = nanoseconds as f64 / 1_000_000.0;
    if milliseconds >= 60_000.0 {
        format!("{:.2} min", milliseconds / 60_000.0)
    } else if milliseconds >= 1_000.0 {
        format!("{:.2} s", milliseconds / 1_000.0)
    } else {
        format!("{milliseconds:.2} ms")
    }
}

/// Python csv.writer QUOTE_MINIMAL semantics: quote when the field contains
/// a comma, quote, CR, or LF; escape quotes by doubling; rows join with CRLF
/// (the caller strips the trailing terminator like Python's rstrip).
fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\r') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_rows(rows: Vec<Vec<String>>) -> String {
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|field| csv_field(field))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect::<Vec<_>>()
        .join("\r\n")
}

fn quote_legacy_csv(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Python `max(items, key=...)` keeps the first maximum in iteration order.
fn first_max<'a, T: Copy + PartialOrd + Default>(
    entries: impl Iterator<Item = (&'a String, T)>,
) -> (String, T) {
    let mut best: Option<(String, T)> = None;
    for (name, value) in entries {
        let replace = match &best {
            None => true,
            Some((_, best_value)) => value > *best_value,
        };
        if replace {
            best = Some((name.clone(), value));
        }
    }
    best.unwrap_or_else(|| (String::new(), T::default()))
}

fn shorten_semantic_caller_file(
    file_path: &str,
    rules: &RuntimeRuleSet,
    dependency_prefix: &str,
) -> String {
    match extract_library_path(file_path, rules, None) {
        Some(library_path) => format!("{dependency_prefix}/{}", library_path.relative_path),
        None => file_path.to_string(),
    }
}

pub fn render_semantic_callers_csv(
    results: &TargetResults,
    rules: &RuntimeRuleSet,
    dependency_prefix: &str,
    legacy_layout: bool,
) -> String {
    if legacy_layout {
        return render_legacy_semantic_callers_csv(results, rules, dependency_prefix);
    }
    let mut rows: Vec<Vec<String>> = vec![vec![
        "Parent Function".to_string(),
        "Category".to_string(),
        "Leaf Function".to_string(),
        "Leaf Samples".to_string(),
        "Top Caller".to_string(),
        "Caller Samples".to_string(),
        "Caller File".to_string(),
    ]];
    for (parent, categories) in results {
        for (category, stats) in categories {
            let mut callers: Vec<_> = stats.semantic_callers.iter().collect();
            callers.sort_by(|left, right| right.1.count.cmp(&left.1.count));
            for (leaf, metrics) in callers {
                let (top_caller, caller_samples) = first_max(
                    metrics
                        .caller_names
                        .iter()
                        .map(|(name, count)| (name, *count)),
                );
                let (caller_file, _) = first_max(
                    metrics
                        .caller_files
                        .iter()
                        .map(|(name, count)| (name, *count)),
                );
                rows.push(vec![
                    parent.clone(),
                    category.clone(),
                    leaf.clone(),
                    metrics.count.to_string(),
                    top_caller,
                    caller_samples.to_string(),
                    shorten_semantic_caller_file(&caller_file, rules, dependency_prefix),
                ]);
            }
        }
    }
    csv_rows(rows)
}

fn render_legacy_semantic_callers_csv(
    results: &TargetResults,
    rules: &RuntimeRuleSet,
    dependency_prefix: &str,
) -> String {
    let mut lines = vec![
        "Parent Function,Category,Leaf Function,Leaf Samples,Top Caller,Caller Samples,Caller File"
            .to_string(),
    ];
    for (parent, categories) in results {
        for (category, stats) in categories {
            for (leaf, metrics) in &stats.semantic_callers {
                if metrics.count == 0 {
                    continue;
                }
                let (top_caller, caller_samples) = first_max(
                    metrics
                        .caller_names
                        .iter()
                        .map(|(name, count)| (name, *count)),
                );
                let (caller_file, _) = first_max(
                    metrics
                        .caller_files
                        .iter()
                        .map(|(name, count)| (name, *count)),
                );
                let shortened =
                    shorten_semantic_caller_file(&caller_file, rules, dependency_prefix);
                lines.push(format!(
                    "{},{},{},{},{},{},{}",
                    quote_legacy_csv(parent),
                    quote_legacy_csv(category),
                    quote_legacy_csv(leaf),
                    metrics.count,
                    quote_legacy_csv(&top_caller),
                    caller_samples,
                    quote_legacy_csv(&shortened),
                ));
            }
        }
    }
    lines.join("\n")
}

pub type Attributables = IndexMap<String, IndexMap<String, f64>>;

fn attributable_columns(attributables: Option<&Attributables>) -> Vec<String> {
    let mut columns: Vec<String> = attributables
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default();
    columns.sort();
    columns
}

fn attributable_values(
    attributables: Option<&Attributables>,
    columns: &[String],
    parent: &str,
    pct: f64,
) -> Result<Vec<String>, String> {
    columns
        .iter()
        .map(|column| {
            match attributables
                .and_then(|table| table.get(column))
                .and_then(|per_parent| per_parent.get(parent))
            {
                None => Ok("N/A".to_string()),
                Some(value) => {
                    // A finite metric scaled by a >100% category share can
                    // overflow; fail closed like the scope estimates.
                    let estimate = (pct / 100.0) * value;
                    if !estimate.is_finite() {
                        return Err(format!(
                            "Attributable estimate for '{column}' is not finite."
                        ));
                    }
                    Ok(format!("{estimate:.1}"))
                }
            }
        })
        .collect()
}

struct CategoryRow<'a> {
    stats: &'a CategoryStats,
    total: TimeNs,
}

/// Percentage with the shared zero-total arm: valid signed samples can
/// cancel a parent's total to exactly zero, and every percentage over a
/// zero total renders as 0 in both implementations — never inf or NaN.
fn pct_of(numerator: TimeNs, total: TimeNs) -> f64 {
    if total != 0 {
        numerator as f64 / total as f64 * 100.0
    } else {
        0.0
    }
}

fn sorted_categories(
    categories: &IndexMap<String, CategoryStats>,
) -> Vec<(&String, &CategoryStats)> {
    let mut ordered: Vec<_> = categories.iter().collect();
    ordered.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    ordered
}

fn function_summary(row: &CategoryRow<'_>, limit: usize, with_samples: bool) -> String {
    let mut top: Vec<_> = row.stats.functions.iter().collect();
    top.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    top.truncate(limit);
    top.iter()
        .map(|(name, metrics)| {
            let pct = pct_of(metrics.cpu_time, row.total);
            if with_samples {
                format!("{name} ({} samples, {pct:.1}%)", metrics.count)
            } else {
                format!("{name} ({pct:.1}%)")
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn callsites_summary(row: &CategoryRow<'_>, separator: &str) -> String {
    let mut caller_totals: IndexMap<&str, TimeNs> = IndexMap::new();
    for (pair, metrics) in &row.stats.caller_leaf_pairs {
        let caller = pair.split(" -> ").next().unwrap_or(pair);
        *caller_totals.entry(caller).or_insert(0) += metrics.cpu_time;
    }
    let mut top: Vec<_> = caller_totals.into_iter().collect();
    top.sort_by(|left, right| right.1.cmp(&left.1));
    top.truncate(3);
    top.iter()
        .map(|(caller, time_ns)| format!("{caller} ({:.1}%)", pct_of(*time_ns, row.total)))
        .collect::<Vec<_>>()
        .join(separator)
}

fn top_pairs<'a>(row: &'a CategoryRow<'a>) -> Vec<(&'a String, &'a CallerMetrics)> {
    let mut pairs: Vec<_> = row.stats.caller_leaf_pairs.iter().collect();
    pairs.sort_by(|left, right| right.1.cpu_time.cmp(&left.1.cpu_time));
    pairs.truncate(3);
    pairs
}

pub fn render_target_csv(
    results: &TargetResults,
    attributables: Option<&Attributables>,
    simplified: bool,
    legacy_layout: bool,
) -> Result<String, String> {
    if legacy_layout {
        return render_legacy_target_csv(results, attributables, simplified);
    }
    let columns = attributable_columns(attributables);
    let mut header: Vec<String> = if simplified {
        vec![
            "Parent Function".to_string(),
            "Category".to_string(),
            "CPU %".to_string(),
        ]
    } else {
        vec![
            "Parent Function".to_string(),
            "Category".to_string(),
            "CPU Time (ns)".to_string(),
            "CPU Time".to_string(),
            "%".to_string(),
        ]
    };
    header.extend(columns.iter().cloned());
    if simplified {
        header.extend([
            "Top 3 Callsites".to_string(),
            "Top Leaf Functions".to_string(),
        ]);
    } else {
        header.extend(
            [
                "Samples",
                "Leaf Functions",
                "Files",
                "Top 3 Callsites",
                "Top Leaf Functions",
                "Top Caller->Leaf Pair",
                "Rank-2 Caller->Leaf Pair",
                "Rank-3 Caller->Leaf Pair",
            ]
            .map(String::from),
        );
    }
    let mut rows = vec![header];
    for (parent, categories) in results {
        let total: TimeNs = categories.values().map(|stats| stats.cpu_time).sum();
        for (category, stats) in sorted_categories(categories) {
            let pct = pct_of(stats.cpu_time, total);
            if simplified && pct.abs() < 0.1 && category != "Other" {
                continue;
            }
            let row = CategoryRow { stats, total };
            let functions = function_summary(&row, if simplified { 3 } else { 5 }, false);
            let callsites = callsites_summary(&row, "; ");
            let attributable = attributable_values(attributables, &columns, parent, pct)?;
            if simplified {
                let mut record = vec![parent.clone(), category.clone(), format!("{pct:.1}")];
                record.extend(attributable);
                record.extend([callsites, functions]);
                rows.push(record);
                continue;
            }
            let mut pair_columns: Vec<String> = top_pairs(&row)
                .iter()
                .map(|(pair, metrics)| {
                    format!(
                        "{pair} ({} samples, {:.1}%)",
                        metrics.count,
                        pct_of(metrics.cpu_time, total)
                    )
                })
                .collect();
            while pair_columns.len() < 3 {
                pair_columns.push(String::new());
            }
            let mut record = vec![
                parent.clone(),
                category.clone(),
                stats.cpu_time.to_string(),
                format_time(stats.cpu_time),
                format!("{pct:.2}"),
            ];
            record.extend(attributable);
            record.extend([
                stats.sample_count.to_string(),
                stats.functions.len().to_string(),
                stats.files.len().to_string(),
                callsites,
                functions,
            ]);
            record.extend(pair_columns);
            rows.push(record);
        }
    }
    Ok(csv_rows(rows))
}

fn render_legacy_target_csv(
    results: &TargetResults,
    attributables: Option<&Attributables>,
    simplified: bool,
) -> Result<String, String> {
    let columns = attributable_columns(attributables);
    let mut header = if simplified {
        "Parent Function,Category,CPU %".to_string()
    } else {
        "Parent Function,Category,CPU Time (ns),CPU Time,%".to_string()
    };
    if !columns.is_empty() {
        header.push(',');
        header.push_str(&columns.join(","));
    }
    if simplified {
        header.push_str(",Top 3 Callsites,Top Leaf Functions");
    } else {
        header.push_str(
            ",Samples,Leaf Functions,Files,Top 3 Callsites,Top Leaf Functions,Top Caller\u{2192}Leaf Pair,Rank-2 Caller\u{2192}Leaf Pair,Rank-3 Caller\u{2192}Leaf Pair",
        );
    }
    let mut lines = vec![header];
    for (parent, categories) in results {
        let total: TimeNs = categories.values().map(|stats| stats.cpu_time).sum();
        for (category, stats) in sorted_categories(categories) {
            let pct = pct_of(stats.cpu_time, total);
            if simplified && pct.abs() < 0.1 && category != "Other" {
                continue;
            }
            let row = CategoryRow { stats, total };
            let functions = function_summary(&row, if simplified { 3 } else { 5 }, !simplified);
            let callsites = callsites_summary(&row, if simplified { "; " } else { ", " });
            let attributable = attributable_values(attributables, &columns, parent, pct)?;
            if simplified {
                let mut line = format!(
                    "{},{},{pct:.1}",
                    quote_legacy_csv(parent),
                    quote_legacy_csv(category),
                );
                if !attributable.is_empty() {
                    line.push(',');
                    line.push_str(&attributable.join(","));
                }
                line.push_str(&format!(
                    ",{},{}",
                    quote_legacy_csv(&callsites),
                    quote_legacy_csv(&functions),
                ));
                lines.push(line);
                continue;
            }
            let mut pair_columns: Vec<String> = top_pairs(&row)
                .iter()
                .map(|(pair, metrics)| {
                    quote_legacy_csv(&format!(
                        "{} ({} samples, {:.1}%)",
                        pair.replace(" -> ", " \u{2192} "),
                        metrics.count,
                        pct_of(metrics.cpu_time, total)
                    ))
                })
                .collect();
            while pair_columns.len() < 3 {
                pair_columns.push("\"\"".to_string());
            }
            let mut line = format!(
                "{},{},{},{},{pct:.2}",
                quote_legacy_csv(parent),
                quote_legacy_csv(category),
                stats.cpu_time,
                quote_legacy_csv(&format_time(stats.cpu_time)),
            );
            if !attributable.is_empty() {
                line.push(',');
                line.push_str(&attributable.join(","));
            }
            line.push_str(&format!(
                ",{},{},{},{},{},{}",
                stats.sample_count,
                stats.functions.len(),
                stats.files.len(),
                quote_legacy_csv(&callsites),
                quote_legacy_csv(&functions),
                pair_columns.join(","),
            ));
            lines.push(line);
        }
    }
    Ok(lines.join("\n"))
}

pub fn render_target_text(
    results: &TargetResults,
    show_folded: bool,
    show_semantic_callers: bool,
) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (parent, categories) in results {
        let total: TimeNs = categories.values().map(|stats| stats.cpu_time).sum();
        lines.push("=".repeat(100));
        lines.push(format!("Parent Function: {parent}"));
        lines.push("=".repeat(100));
        lines.push(format!(
            "Total CPU time under this function: {}",
            format_time(total)
        ));
        if show_folded {
            let folded_total: TimeNs = categories
                .values()
                .map(|stats| stats.folded_from.values().sum::<TimeNs>())
                .sum();
            if folded_total != 0 {
                lines.push(format!(
                    "Total runtime internals folded into categories: {}",
                    format_time(folded_total)
                ));
            }
        }
        lines.push(String::new());
        lines.push(format!(
            "{:<35} {:<15} {:<8} {:<10} {:<15} {:<8}",
            "Category", "CPU Time", "%", "Samples", "Leaf Functions", "Files"
        ));
        lines.push("-".repeat(110));
        let ordered = sorted_categories(categories);
        for (category, stats) in &ordered {
            let pct = pct_of(stats.cpu_time, total);
            lines.push(format!(
                "{:<35} {:<15} {:>6.2}% {:>9} {:>14} {:>7}",
                category,
                format_time(stats.cpu_time),
                pct,
                stats.sample_count,
                stats.functions.len(),
                stats.files.len(),
            ));
        }
        lines.push("-".repeat(110));
        lines.push(format!(
            "{:<35} {:<15} {:>8}",
            "TOTAL",
            format_time(total),
            if total != 0 { "100.00%" } else { "0.00%" }
        ));
        if show_folded {
            let folded: Vec<_> = ordered
                .iter()
                .filter(|(_, stats)| !stats.folded_from.is_empty())
                .take(5)
                .collect();
            if !folded.is_empty() {
                lines.push(String::new());
                lines.push("Runtime internals folded into categories:".to_string());
                for (category, stats) in folded {
                    lines.push(format!("  {category}:"));
                    let mut entries: Vec<_> = stats.folded_from.iter().collect();
                    entries.sort_by(|left, right| right.1.cmp(left.1));
                    for (function, time_ns) in entries.into_iter().take(10) {
                        lines.push(format!("    - {function}: {}", format_time(*time_ns)));
                    }
                }
            }
        }
        if show_semantic_callers {
            let semantic: Vec<_> = ordered
                .iter()
                .filter(|(_, stats)| !stats.semantic_callers.is_empty())
                .take(5)
                .collect();
            if !semantic.is_empty() {
                lines.push(String::new());
                lines.push("Semantic callers for runtime internals:".to_string());
                for (category, stats) in semantic {
                    lines.push(format!("  {category}:"));
                    let mut leaves: Vec<_> = stats.semantic_callers.iter().collect();
                    leaves.sort_by(|left, right| right.1.count.cmp(&left.1.count));
                    for (leaf, metrics) in leaves.into_iter().take(5) {
                        lines.push(format!("    {leaf} ({} samples):", metrics.count));
                        let mut callers: Vec<_> = metrics.caller_names.iter().collect();
                        callers.sort_by(|left, right| right.1.cmp(left.1));
                        for (caller, count) in callers.into_iter().take(3) {
                            lines.push(format!("      - {caller} ({count} samples)"));
                        }
                    }
                }
            }
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_time_matches_python_units() {
        assert_eq!(format_time(5_000_000), "5.00 ms");
        assert_eq!(format_time(1_500_000_000), "1.50 s");
        assert_eq!(format_time(120_000_000_000), "2.00 min");
        assert_eq!(format_time(0), "0.00 ms");
    }

    #[test]
    fn csv_fields_quote_like_python_quote_minimal() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("has,comma"), "\"has,comma\"");
        assert_eq!(csv_field("has \"quote\""), "\"has \"\"quote\"\"\"");
        assert_eq!(
            csv_rows(vec![
                vec!["a".to_string(), "b,c".to_string()],
                vec!["d".to_string(), "e".to_string()],
            ]),
            "a,\"b,c\"\r\nd,e"
        );
    }

    #[test]
    fn first_max_keeps_the_first_maximum() {
        let alpha = "alpha".to_string();
        let beta = "beta".to_string();
        let entries = vec![(&alpha, 3usize), (&beta, 3usize)];
        let (name, value) = first_max(entries.into_iter());
        assert_eq!((name.as_str(), value), ("alpha", 3));
    }
}
