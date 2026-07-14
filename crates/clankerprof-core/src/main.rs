use clankerprof_core::compare::{compare_json, CompareOptions};
use clankerprof_core::facts::sample_facts_from_json;
use clankerprof_core::model::ProfileFacts;
use clankerprof_core::render::{
    render_semantic_callers_csv, render_target_csv, render_target_text, Attributables,
};
use clankerprof_core::scopes::{
    analyze_boundary_facts, load_boundary_options, render_boundary_json,
};
use clankerprof_core::slices::{
    analyze_slice_facts, load_slices_file, render_slice_json, SliceAnalysisOptions,
};
use clankerprof_core::targets::TargetConfig;
use clankerprof_core::targets::{
    analyze_target_facts_with_options, parse_target_config_json, render_target_json,
    TargetAnalysisOptions,
};
use clankerprof_core::{load_profile, sample_facts_to_compact_json, sample_facts_to_pretty_json};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeSet;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "clankerprof-rs",
    version,
    about = "Rust core for clankerprof sample-facts decoding and projections"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Export decoded pprof sample facts as stable JSON.
    Facts(FactsArgs),
    /// Attribute target-function self time to configured categories.
    Targets(TargetsArgs),
    /// Attribute samples to ownership slices.
    Slices(SlicesArgs),
    /// Compare two clankerprof reports with regression gates.
    Compare(CompareArgs),
    /// Run scope/cost-kind/rollup/owner decomposition over a profile.
    #[command(alias = "boundaries")]
    Scopes(ScopesArgs),
}

#[derive(Args)]
struct FactsArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: PathBuf,
    /// Write the facts artifact to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Indent the facts artifact for humans (default is compact JSON).
    #[arg(long)]
    pretty: bool,
}

#[derive(Args)]
struct TargetsArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Read versioned sample-facts JSON instead of a pprof profile.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// JSON target config mapping parent functions to category patterns.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Minimal mode: add a parent function with no category patterns.
    #[arg(long = "target")]
    targets: Vec<String>,
    /// Runtime rule pack to apply (generic or ruby).
    #[arg(long, default_value = "generic")]
    runtime: String,
    /// External runtime rule pack YAML (overrides --runtime).
    #[arg(long)]
    runtime_rules: Option<PathBuf>,
    /// Core classes CSV override for the ruby runtime.
    #[arg(long)]
    core_classes: Option<PathBuf>,
    /// Disable enhanced runtime categorization (caller fallback engages).
    #[arg(long)]
    no_enhanced: bool,
    /// Fold runtime-internal leaves into the first meaningful caller.
    #[arg(long)]
    fold_runtime_internals: bool,
    /// Track semantic callers for native leaves.
    #[arg(long)]
    track_semantic_callers: bool,
    /// Write the semantic-caller CSV to this path.
    #[arg(long)]
    semantic_callers_csv: Option<PathBuf>,
    /// JSON file of attributable column -> parent function -> value.
    #[arg(long)]
    cpu_attributables: Option<PathBuf>,
    /// Write the report to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Output format: json, csv, simple-csv, or text.
    #[arg(long, default_value = "json")]
    format: String,
    /// CSV artifact layout (standard, or compat for the two-file pair).
    #[arg(long, default_value = "standard")]
    target_csv_layout: String,
}

#[derive(Args)]
struct SlicesArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Read versioned sample-facts JSON instead of a pprof profile.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// Slice definitions YAML file.
    #[arg(long)]
    slices: Option<PathBuf>,
    /// Attribution override rule '<key>:<value>,to:<slice>', repeatable.
    #[arg(long)]
    attribute: Vec<String>,
    /// Accept --attribute targets that are not declared slices.
    #[arg(long)]
    allow_virtual_attribute_slices: bool,
    /// Runtime rule pack to apply (generic or ruby).
    #[arg(long, default_value = "generic")]
    runtime: String,
    /// External runtime rule pack YAML (overrides --runtime).
    #[arg(long)]
    runtime_rules: Option<PathBuf>,
    /// Core classes CSV override for the ruby runtime.
    #[arg(long)]
    core_classes: Option<PathBuf>,
    /// Bottom-frame filter, repeatable.
    #[arg(long)]
    filter: Vec<String>,
    /// Collapse rule, repeatable.
    #[arg(long)]
    collapse: Vec<String>,
    /// Limit frames per slice.
    #[arg(long)]
    top: Option<usize>,
    /// Slice count limit or percentage threshold (bare flag means 0.1%).
    #[arg(long, num_args = 0..=1, default_missing_value = "0.1%")]
    by_slice: Option<String>,
    /// Include filename paths in frame output.
    #[arg(long)]
    show_paths: bool,
    /// Keep native frames eligible as bottom attribution frames.
    #[arg(long)]
    no_collapse_native: bool,
    /// Report unattributed dependency libraries for the default slice
    /// (optionally limited to N entries).
    #[arg(long, visible_alias = "unattributed-gems", num_args = 0..=1, default_missing_value = "9223372036854775807")]
    unattributed_libraries: Option<usize>,
    /// Write the report to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct ScopesArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Read versioned sample-facts JSON instead of a pprof profile.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// Scope config (TOML or YAML).
    #[arg(long)]
    config: PathBuf,
    /// Limit ranked rows per section.
    #[arg(long)]
    top: Option<usize>,
    /// Runtime rule pack to apply (generic or ruby).
    #[arg(long, default_value = "generic")]
    runtime: String,
    /// External runtime rule pack YAML (overrides --runtime).
    #[arg(long)]
    runtime_rules: Option<PathBuf>,
    /// Core classes CSV override for the ruby runtime.
    #[arg(long)]
    core_classes: Option<PathBuf>,
    /// Write the report to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct CompareArgs {
    /// Baseline report JSON path.
    #[arg(long)]
    before: PathBuf,
    /// Candidate report JSON path.
    #[arg(long)]
    after: PathBuf,
    /// Absolute percentage-point regression threshold.
    #[arg(long, default_value_t = 2.0)]
    threshold_abs: f64,
    /// Relative percentage regression threshold.
    #[arg(long, default_value_t = 15.0)]
    threshold_rel: f64,
    /// Comma-delimited slice names to gate on.
    #[arg(long)]
    focus_slices: Option<String>,
    /// Comma-delimited boundary names to gate on.
    #[arg(long)]
    focus_boundaries: Option<String>,
    /// Write the comparison to this path instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(exit_code) => std::process::exit(exit_code),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({
                    "error": error,
                    "ok": false,
                })
            );
            std::process::exit(2);
        }
    }
}

fn run(cli: Cli) -> Result<i32, String> {
    match cli.command {
        Command::Facts(args) => run_facts(args).map(|()| 0),
        Command::Targets(args) => run_targets(args).map(|()| 0),
        Command::Slices(args) => run_slices(args).map(|()| 0),
        Command::Compare(args) => run_compare(args),
        Command::Scopes(args) => run_scopes(args).map(|()| 0),
    }
}

fn load_projection_input(
    profile: Option<&PathBuf>,
    facts: Option<&PathBuf>,
) -> Result<ProfileFacts, String> {
    match (profile, facts) {
        (Some(_), Some(_)) => Err("--profile and --facts are mutually exclusive.".to_string()),
        (None, None) => Err("--profile or --facts is required.".to_string()),
        (Some(profile_path), None) => Ok(load_profile(profile_path)
            .map_err(|error| error.to_string())?
            .to_sample_facts()),
        (None, Some(facts_path)) => {
            let payload: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(facts_path).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
            sample_facts_from_json(&payload)
        }
    }
}

fn run_scopes(args: ScopesArgs) -> Result<(), String> {
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
    )?;
    let options = load_boundary_options(&args.config, runtime_rules)?;
    let facts = load_projection_input(args.profile.as_ref(), args.facts.as_ref())?;
    let payload = render_boundary_json(&analyze_boundary_facts(&facts, &options)?, args.top);
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    emit(&rendered, args.output.as_ref())
}

fn emit(rendered: &str, output: Option<&PathBuf>) -> Result<(), String> {
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_facts(args: FactsArgs) -> Result<(), String> {
    let profile = load_profile(&args.profile).map_err(|error| error.to_string())?;
    let facts = profile.to_sample_facts();
    let rendered = if args.pretty {
        sample_facts_to_pretty_json(&facts).map_err(|error| error.to_string())?
    } else {
        sample_facts_to_compact_json(&facts).map_err(|error| error.to_string())?
    };
    emit(&rendered, args.output.as_ref())
}

fn resolve_runtime_rules(
    runtime: &str,
    runtime_rules: Option<&PathBuf>,
    core_classes: Option<&PathBuf>,
) -> Result<clankerprof_core::rules::RuntimeRuleSet, String> {
    use clankerprof_core::rules;
    let classes = match core_classes {
        Some(path) => rules::load_ruby_core_classes(path)?,
        None if runtime == "ruby" => rules::load_default_ruby_core_classes(),
        None => std::collections::BTreeSet::new(),
    };
    if let Some(path) = runtime_rules {
        return rules::load_runtime_rules_file(path, classes, false);
    }
    match runtime {
        "generic" => Ok(rules::RuntimeRuleSet::generic().clone()),
        "ruby" => rules::ruby_rules(classes, false),
        other => Err(format!("Unsupported runtime: {other}")),
    }
}

fn load_attributables(path: Option<&PathBuf>) -> Result<Option<Attributables>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    let payload: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    let serde_json::Value::Object(columns) = payload else {
        return Err("Attributables must be a JSON object.".to_string());
    };
    let mut result = Attributables::new();
    for (name, values) in columns {
        let serde_json::Value::Object(entries) = values else {
            return Err(format!("Attributable column {name} must be an object."));
        };
        let mut column = indexmap::IndexMap::new();
        for (key, value) in entries {
            let number = value
                .as_f64()
                .ok_or_else(|| format!("Attributable column {name} must be an object."))?;
            column.insert(key, number);
        }
        result.insert(name, column);
    }
    Ok(Some(result))
}

fn run_targets(args: TargetsArgs) -> Result<(), String> {
    if !matches!(args.format.as_str(), "json" | "csv" | "simple-csv" | "text") {
        return Err(format!("Unsupported --format: {}.", args.format));
    }
    let compat_layout = match args.target_csv_layout.as_str() {
        "standard" => false,
        "compat" => true,
        other => return Err(format!("Unsupported --target-csv-layout: {other}.")),
    };
    if compat_layout && (args.format != "csv" || args.output.is_none()) {
        return Err("--target-csv-layout=compat requires --format csv and --output.".to_string());
    }
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
    )?;
    let mut config = match &args.config {
        Some(config_path) => {
            let config_payload =
                std::fs::read_to_string(config_path).map_err(|error| error.to_string())?;
            parse_target_config_json(&config_payload)?
        }
        None => TargetConfig::new(),
    };
    for target in &args.targets {
        config.entry(target.clone()).or_default();
    }
    if config.is_empty() {
        return Err("--config or --target is required.".to_string());
    }
    let attributables = load_attributables(args.cpu_attributables.as_ref())?;
    let facts = load_projection_input(args.profile.as_ref(), args.facts.as_ref())?;
    let options = TargetAnalysisOptions {
        enhanced_runtime_categorization: !args.no_enhanced,
        fold_runtime_internals: args.fold_runtime_internals,
        track_semantic_callers: args.track_semantic_callers,
        caller_fallback_when_uncategorized: args.no_enhanced,
        runtime_rules: runtime_rules.clone(),
    };
    let results = analyze_target_facts_with_options(&facts, &config, &options);
    if let Some(semantic_csv_path) = &args.semantic_callers_csv {
        if !args.track_semantic_callers {
            return Err("--semantic-callers-csv requires --track-semantic-callers.".to_string());
        }
        let dependency_prefix = if compat_layout { "gems" } else { "deps" };
        let csv_text =
            render_semantic_callers_csv(&results, &runtime_rules, dependency_prefix, compat_layout);
        let suffix = if compat_layout { "" } else { "\n" };
        std::fs::write(semantic_csv_path, format!("{csv_text}{suffix}"))
            .map_err(|error| error.to_string())?;
    }
    if args.format == "json" {
        let payload = render_target_json(&results);
        let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
        return emit(&rendered, args.output.as_ref());
    }
    if compat_layout {
        return write_compat_target_csv_artifacts(
            args.output.as_ref().expect("validated above"),
            &results,
            attributables.as_ref(),
        );
    }
    let rendered = if args.format == "csv" || args.format == "simple-csv" {
        render_target_csv(
            &results,
            attributables.as_ref(),
            args.format == "simple-csv",
            false,
        )
    } else {
        render_target_text(
            &results,
            args.fold_runtime_internals,
            args.track_semantic_callers,
        )
    };
    emit(&rendered, args.output.as_ref())
}

fn write_compat_target_csv_artifacts(
    output_path: &std::path::Path,
    results: &clankerprof_core::targets::TargetResults,
    attributables: Option<&Attributables>,
) -> Result<(), String> {
    let base_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("--output must name a file.")?;
    let output_dir = std::path::Path::new("output");
    let verbose_dir = output_dir.join("verbose");
    std::fs::create_dir_all(&verbose_dir).map_err(|error| error.to_string())?;
    std::fs::write(
        output_dir.join(base_name),
        render_target_csv(results, attributables, true, true),
    )
    .map_err(|error| error.to_string())?;
    std::fs::write(
        verbose_dir.join(base_name),
        render_target_csv(results, attributables, false, true),
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

fn run_slices(args: SlicesArgs) -> Result<(), String> {
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
    )?;
    let mut attributes = Vec::new();
    for raw in &args.attribute {
        attributes.push(parse_attribute(raw)?);
    }
    let mut slices_path = args.slices.clone();
    let slice_aware = slices_path.is_some()
        || args.by_slice.is_some()
        || !attributes.is_empty()
        || args
            .filter
            .iter()
            .chain(&args.collapse)
            .any(|item| item.trim_start_matches(['!', '<']).starts_with("slice:"));
    if slices_path.is_none() && slice_aware {
        let default_slices = std::path::Path::new("slices.yml");
        if default_slices.exists() {
            slices_path = Some(default_slices.to_path_buf());
        }
    }
    let mut options = SliceAnalysisOptions {
        filters: args.filter,
        collapse: args.collapse,
        attributes,
        top: args.top,
        by_slice: args.by_slice,
        show_paths: args.show_paths,
        no_collapse_native: args.no_collapse_native,
        unattributed_libraries: args.unattributed_libraries,
        runtime_rules,
        ..SliceAnalysisOptions::default()
    };
    if let Some(slices_path) = slices_path {
        options.slices = load_slices_file(slices_path)?;
    }
    validate_slice_options(&options, args.allow_virtual_attribute_slices)?;
    let facts = load_projection_input(args.profile.as_ref(), args.facts.as_ref())?;
    let payload = render_slice_json(&analyze_slice_facts(&facts, &options), &options);
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    emit(&rendered, args.output.as_ref())
}

fn parse_attribute(raw: &str) -> Result<clankerprof_core::slices::AttributionRule, String> {
    let Some((filter_part, target)) = raw.split_once(",to:") else {
        return Err(format!(
            "Attribute rule must be '<filter>,to:<slice>': {raw}"
        ));
    };
    if target.is_empty() {
        return Err(format!(
            "Attribute rule must be '<filter>,to:<slice>': {raw}"
        ));
    }
    if filter_part.starts_with('!') {
        return Err(format!("Attribute rules do not support '!': {raw}"));
    }
    let descendant = filter_part.starts_with('<');
    let body = if descendant {
        &filter_part[1..]
    } else {
        filter_part
    };
    if body.starts_with('!') {
        return Err(format!("Attribute rules do not support '!': {raw}"));
    }
    let Some((key, value)) = body.split_once(':') else {
        return Err(format!(
            "Attribute rule filter must be '<key>:<value>': {raw}"
        ));
    };
    if key == "slice" {
        return Err(format!(
            "Attribute rules do not support slice: filters: {raw}"
        ));
    }
    Ok(clankerprof_core::slices::AttributionRule {
        key: key.to_string(),
        value: value.to_string(),
        target_slice: target.to_string(),
        descendant,
    })
}

fn valid_filter_keys(rules: &clankerprof_core::rules::RuntimeRuleSet) -> Vec<String> {
    let mut keys: Vec<String> = vec!["name".to_string(), "path".to_string(), "slice".to_string()];
    keys.extend(
        clankerprof_core::targets::DEFAULT_LIBRARY_SELECTORS
            .iter()
            .map(ToString::to_string),
    );
    keys.extend(rules.library_selector_path_patterns.keys().cloned());
    keys
}

fn validate_slice_options(
    options: &SliceAnalysisOptions,
    allow_virtual_attribute_slices: bool,
) -> Result<(), String> {
    let names: std::collections::HashSet<&str> = options
        .slices
        .iter()
        .map(|item| item.name.as_str())
        .collect();
    let valid_keys = valid_filter_keys(&options.runtime_rules);
    if options.slices.is_empty() {
        if options.by_slice.is_some() {
            return Err("--by-slice requires --slices=<file>.".to_string());
        }
        if !options.attributes.is_empty() {
            return Err("--attribute requires --slices=<file>.".to_string());
        }
        if options
            .filters
            .iter()
            .chain(&options.collapse)
            .any(|item| item.trim_start_matches(['!', '<']).starts_with("slice:"))
        {
            return Err("slice:... requires --slices=<file>.".to_string());
        }
    }
    for raw_filter in options.filters.iter().chain(&options.collapse) {
        let body = raw_filter.trim_start_matches(['!', '<']);
        let Some((key, value)) = body.split_once(':') else {
            return Err(format!("Filter must be '<key>:<value>': {raw_filter}"));
        };
        if key.is_empty() || value.is_empty() {
            return Err(format!("Filter must be '<key>:<value>': {raw_filter}"));
        }
        if !valid_keys.iter().any(|valid| valid == key) {
            return Err(format!("Unsupported filter key: {key}"));
        }
        if key == "slice" && !names.contains(value) {
            return Err(format!("Unknown slice: {value}"));
        }
    }
    for raw_collapse in &options.collapse {
        if raw_collapse.starts_with('!') || raw_collapse.starts_with('<') {
            return Err(format!(
                "Collapse filters do not support prefixes: {raw_collapse}"
            ));
        }
    }
    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    for attribute in &options.attributes {
        if attribute.key == "slice" || !valid_keys.iter().any(|valid| valid == &attribute.key) {
            return Err(format!(
                "Unsupported attribute filter key: {}",
                attribute.key
            ));
        }
        if !names.contains(attribute.target_slice.as_str()) && !allow_virtual_attribute_slices {
            return Err(format!(
                "Unknown slice in --attribute: {}",
                attribute.target_slice
            ));
        }
        if !seen.insert((attribute.key.clone(), attribute.value.clone())) {
            return Err(format!(
                "Duplicate attribute rule filter: {}:{}",
                attribute.key, attribute.value
            ));
        }
    }
    Ok(())
}

fn run_compare(args: CompareArgs) -> Result<i32, String> {
    let options = CompareOptions {
        threshold_abs: args.threshold_abs,
        threshold_rel: args.threshold_rel,
        focus_slices: split_focus(args.focus_slices.as_deref()),
        focus_boundaries: split_focus(args.focus_boundaries.as_deref()),
    };
    let before_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&args.before).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let after_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&args.after).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let payload = compare_json(&before_payload, &after_payload, &options)?;
    let has_regression = payload
        .get("has_regression")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    emit(&rendered, args.output.as_ref())?;
    Ok(if has_regression { 2 } else { 0 })
}

fn split_focus(value: Option<&str>) -> BTreeSet<String> {
    value
        .map(|raw| {
            raw.split(',')
                .filter(|part| !part.is_empty())
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}
