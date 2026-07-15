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
    /// Decode once and emit multiple projections in a single run.
    Report(ReportArgs),
}

#[derive(Args)]
struct FactsArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: PathBuf,
    /// Write the facts artifact to this path instead of stdout.
    #[arg(long, overrides_with = "output")]
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
    #[arg(long, visible_alias = "ruby-core-classes")]
    core_classes: Option<PathBuf>,
    /// Disable enhanced runtime categorization (caller fallback engages).
    #[arg(long)]
    no_enhanced: bool,
    /// Fold runtime-internal leaves into the first meaningful caller.
    #[arg(long, visible_alias = "fold-ruby-internals")]
    fold_runtime_internals: bool,
    /// Keep verbose-only foldable categories visible in runtime rule packs.
    #[arg(long, visible_alias = "verbose-ruby-internals")]
    verbose_runtime_internals: bool,
    /// Track semantic callers for native leaves.
    #[arg(long)]
    track_semantic_callers: bool,
    /// Write the semantic-caller CSV to this path.
    #[arg(long)]
    semantic_callers_csv: Option<PathBuf>,
    /// JSON file of attributable column -> parent function -> value.
    #[arg(long, visible_alias = "attributables")]
    cpu_attributables: Option<PathBuf>,
    /// Write the report to this path instead of stdout.
    #[arg(long, overrides_with = "output")]
    output: Option<PathBuf>,
    /// Output format: json, csv, simple-csv, or text.
    #[arg(long, default_value = "json")]
    format: String,
    /// CSV artifact layout (standard, or compat for the two-file pair).
    #[arg(long)]
    target_csv_layout: Option<String>,
    /// Compatibility alias for --target-csv-layout=compat.
    #[arg(long)]
    legacy_target_csv_layout: bool,
}

#[derive(Args)]
struct SlicesArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Read versioned sample-facts JSON instead of a pprof profile.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// Slice options file (TOML or YAML) merged with CLI flags.
    #[arg(long)]
    config: Option<PathBuf>,
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
    #[arg(long, visible_alias = "ruby-core-classes")]
    core_classes: Option<PathBuf>,
    /// Keep verbose-only foldable categories visible in runtime rule packs.
    #[arg(long, visible_alias = "verbose-ruby-internals")]
    verbose_runtime_internals: bool,
    /// Bottom-frame filter, repeatable.
    #[arg(long)]
    filter: Vec<String>,
    /// Collapse rule, repeatable.
    #[arg(long)]
    collapse: Vec<String>,
    /// Limit frames per slice (negative drops from the tail, Python-style).
    #[arg(long, allow_negative_numbers = true)]
    top: Option<String>,
    /// Slice count limit or percentage threshold (bare flag means 0.1%).
    /// Negative-number values must parse as values, mirroring argparse.
    #[arg(long, num_args = 0..=1, default_missing_value = "0.1%", allow_negative_numbers = true)]
    by_slice: Option<String>,
    /// Include filename paths in frame output.
    #[arg(long)]
    show_paths: bool,
    /// Keep native frames eligible as bottom attribution frames.
    #[arg(long)]
    no_collapse_native: bool,
    /// Report unattributed dependency libraries for the default slice
    /// (optionally limited to N entries).
    #[arg(long, visible_alias = "unattributed-gems", num_args = 0..=1, default_missing_value = "9223372036854775807", allow_negative_numbers = true)]
    unattributed_libraries: Option<String>,
    /// Write the report to this path instead of stdout.
    #[arg(long, overrides_with = "output")]
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
    /// Limit ranked rows per section (negative drops from the tail, Python-style).
    #[arg(long, allow_negative_numbers = true)]
    top: Option<String>,
    /// Runtime rule pack to apply (generic or ruby).
    #[arg(long, default_value = "generic")]
    runtime: String,
    /// External runtime rule pack YAML (overrides --runtime).
    #[arg(long)]
    runtime_rules: Option<PathBuf>,
    /// Core classes CSV override for the ruby runtime.
    #[arg(long, visible_alias = "ruby-core-classes")]
    core_classes: Option<PathBuf>,
    /// Disable enhanced runtime categorization (caller fallback engages).
    #[arg(long)]
    no_enhanced: bool,
    /// Fold runtime-internal leaves into the first meaningful caller.
    #[arg(long, visible_alias = "fold-ruby-internals")]
    fold_runtime_internals: bool,
    /// Keep verbose-only foldable categories visible in runtime rule packs.
    #[arg(long, visible_alias = "verbose-ruby-internals")]
    verbose_runtime_internals: bool,
    /// Write the report to this path instead of stdout.
    #[arg(long, overrides_with = "output")]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct ReportArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Read versioned sample-facts JSON instead of a pprof profile.
    #[arg(long)]
    facts: Option<PathBuf>,
    /// JSON target config; enables the targets section.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Slice definitions YAML; enables the slices section.
    #[arg(long)]
    slices: Option<PathBuf>,
    /// Scope config (TOML or YAML); enables the scopes section.
    #[arg(long)]
    scopes_config: Option<PathBuf>,
    /// Include the sample-facts payload as a facts section.
    #[arg(long)]
    include_facts: bool,
    /// Limit ranked scope rows per section (negative drops from the tail, Python-style).
    #[arg(long, allow_negative_numbers = true)]
    top: Option<String>,
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
    #[arg(long, overrides_with = "output")]
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
    #[arg(long, overrides_with = "focus_slices")]
    focus_slices: Option<String>,
    /// Comma-delimited boundary names to gate on.
    #[arg(
        long,
        visible_alias = "focus-scopes",
        overrides_with = "focus_boundaries"
    )]
    focus_boundaries: Option<String>,
    /// Write the comparison to this path instead of stdout.
    #[arg(long, overrides_with = "output")]
    output: Option<PathBuf>,
}

fn main() {
    let argv = match hoist_global_output(std::env::args().collect()) {
        Ok(argv) => argv,
        Err(error) => exit_with_envelope(&error),
    };
    // try_parse keeps usage errors inside the JSON error contract; clap's
    // default parse() would print prose usage text and exit on its own.
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(error) => {
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) {
                error.exit();
            }
            exit_with_envelope(error.render().to_string().trim_end())
        }
    };
    match run(cli) {
        Ok(exit_code) => std::process::exit(exit_code),
        Err(error) => exit_with_envelope(&error),
    }
}

fn exit_with_envelope(error: &str) -> ! {
    // Byte-identical to Python's json.dumps({...}, sort_keys=True) envelope.
    eprintln!(
        "{}",
        clankerprof_core::pyjson::dumps_compact(&serde_json::json!({
            "error": error,
            "ok": false,
        }))
    );
    std::process::exit(2);
}

/// Treat a global --output before the subcommand as the subcommand's own,
/// mirroring the Python CLI's hoist (the moved token lands last, so it wins
/// over an earlier local --output).
fn hoist_global_output(argv: Vec<String>) -> Result<Vec<String>, String> {
    let mut tokens = argv;
    let mut moved: Vec<String> = Vec::new();
    let mut index = 1.min(tokens.len());
    while index < tokens.len() {
        let item = tokens[index].clone();
        if !item.starts_with('-') {
            break;
        }
        if item == "--output" {
            if index + 1 >= tokens.len() {
                return Err("--output requires a path argument.".to_string());
            }
            moved = tokens.drain(index..index + 2).collect();
            continue;
        }
        if item.starts_with("--output=") {
            moved = vec![tokens.remove(index)];
            continue;
        }
        index += 1;
    }
    tokens.extend(moved);
    Ok(tokens)
}

fn run(cli: Cli) -> Result<i32, String> {
    match cli.command {
        Command::Facts(args) => run_facts(args).map(|()| 0),
        Command::Targets(args) => run_targets(args).map(|()| 0),
        Command::Slices(args) => run_slices(args).map(|()| 0),
        Command::Compare(args) => run_compare(args),
        Command::Scopes(args) => run_scopes(args).map(|()| 0),
        Command::Report(args) => run_report(args).map(|()| 0),
    }
}

fn run_report(args: ReportArgs) -> Result<(), String> {
    if args.config.is_none() && args.slices.is_none() && args.scopes_config.is_none() {
        return Err(
            "report requires at least one of --config, --slices, or --scopes-config.".to_string(),
        );
    }
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
        false,
    )?;
    // The whole point: decode a single facts model and feed every projection.
    let facts = load_projection_input(args.profile.as_ref(), args.facts.as_ref())?;
    let mut payload = serde_json::Map::new();
    payload.insert(
        "tool".to_string(),
        serde_json::Value::String("clankerprof_report".to_string()),
    );
    if args.include_facts {
        payload.insert(
            "facts".to_string(),
            clankerprof_core::sample_facts_to_json_value(&facts),
        );
    }
    if let Some(config_path) = &args.config {
        let config_payload =
            std::fs::read_to_string(config_path).map_err(|error| error.to_string())?;
        let config = parse_target_config_json(&config_payload)?;
        let options = TargetAnalysisOptions {
            runtime_rules: runtime_rules.clone(),
            ..TargetAnalysisOptions::default()
        };
        payload.insert(
            "targets".to_string(),
            render_target_json(&analyze_target_facts_with_options(
                &facts, &config, &options,
            )),
        );
    }
    if let Some(slices_path) = &args.slices {
        let options = SliceAnalysisOptions {
            slices: load_slices_file(slices_path)?,
            runtime_rules: runtime_rules.clone(),
            ..SliceAnalysisOptions::default()
        };
        payload.insert(
            "slices".to_string(),
            render_slice_json(&analyze_slice_facts(&facts, &options), &options)?,
        );
    }
    if let Some(scopes_config_path) = &args.scopes_config {
        let options = load_boundary_options(scopes_config_path, runtime_rules)?;
        payload.insert(
            "scopes".to_string(),
            render_boundary_json(
                &analyze_boundary_facts(&facts, &options)?,
                int64_flag(args.top.as_deref(), TOP_INT_MESSAGE)?,
            ),
        );
    }
    clankerprof_core::targets::take_pattern_error()?;
    let rendered = clankerprof_core::pyjson::dumps_pretty(&serde_json::Value::Object(payload));
    emit(&rendered, args.output.as_ref(), "clankerprof_report")
}

fn load_projection_input(
    profile: Option<&PathBuf>,
    facts: Option<&PathBuf>,
) -> Result<ProfileFacts, String> {
    match (profile, facts) {
        (Some(_), Some(_)) => Err("--profile and --facts are mutually exclusive.".to_string()),
        (None, None) => Err("--profile or --facts is required.".to_string()),
        (Some(profile_path), None) => load_profile(profile_path)
            .map_err(|error| error.to_string())?
            .to_sample_facts(),
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
        args.verbose_runtime_internals,
    )?;
    let mut options = load_boundary_options(&args.config, runtime_rules)?;
    // Mirror the Python CLI's post-config option overrides (run_boundaries):
    // Python's legacy_no_enhanced_caller_fallback is OR-ed into the caller
    // fallback at categorize time, so one field carries both here.
    options.enhanced_runtime_categorization = !args.no_enhanced;
    options.fold_runtime_internals = args.fold_runtime_internals;
    options.caller_fallback_when_uncategorized = args.no_enhanced;
    let facts = load_projection_input(args.profile.as_ref(), args.facts.as_ref())?;
    let payload = render_boundary_json(
        &analyze_boundary_facts(&facts, &options)?,
        int64_flag(args.top.as_deref(), TOP_INT_MESSAGE)?,
    );
    clankerprof_core::targets::take_pattern_error()?;
    let rendered = clankerprof_core::pyjson::dumps_pretty(&payload);
    emit(&rendered, args.output.as_ref(), "clankerprof_boundaries")
}

const TOP_INT_MESSAGE: &str = "--top values must be integers.";
const UNATTRIBUTED_LIBRARIES_INT_MESSAGE: &str =
    "--unattributed-libraries values must be integers.";

/// Strict CLI integer grammar shared with Python's strict_int64: optional
/// sign, ASCII digits, i64 range. clap's typed i64 parse would reject the
/// same strings but with engine-specific wording.
fn int64_flag(raw: Option<&str>, message: &str) -> Result<Option<i64>, String> {
    match raw {
        None => Ok(None),
        Some(text) => text
            .parse::<i64>()
            .map(Some)
            .map_err(|_| message.to_string()),
    }
}

fn receipt_value(tool: &str, output: &std::path::Path) -> serde_json::Value {
    serde_json::json!({
        "ok": true,
        "output": output.to_string_lossy(),
        "tool": tool,
    })
}

/// Write the artifact and print the standard JSON receipt (byte-identical to
/// the Python CLI's), or print the rendered payload when no --output was
/// given.
fn emit(rendered: &str, output: Option<&PathBuf>, tool: &str) -> Result<(), String> {
    let receipt = output
        .map(|path| receipt_value(tool, path))
        .unwrap_or(serde_json::Value::Null);
    emit_with_receipt(rendered, output, receipt)
}

fn emit_with_receipt(
    rendered: &str,
    output: Option<&PathBuf>,
    receipt: serde_json::Value,
) -> Result<(), String> {
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
        let receipt_rendered = clankerprof_core::pyjson::dumps_pretty(&receipt);
        println!("{receipt_rendered}");
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_facts(args: FactsArgs) -> Result<(), String> {
    let profile = load_profile(&args.profile).map_err(|error| error.to_string())?;
    let facts = profile.to_sample_facts()?;
    let rendered = if args.pretty {
        sample_facts_to_pretty_json(&facts).map_err(|error| error.to_string())?
    } else {
        sample_facts_to_compact_json(&facts).map_err(|error| error.to_string())?
    };
    let receipt = match args.output.as_ref() {
        Some(output_path) => {
            // The facts receipt carries schema_version and summary like the
            // Python CLI's.
            let payload = clankerprof_core::sample_facts_to_json_value(&facts);
            serde_json::json!({
                "ok": true,
                "output": output_path.to_string_lossy(),
                "schema_version": payload.get("schema_version"),
                "summary": payload.get("summary"),
                "tool": "clankerprof_facts",
            })
        }
        None => serde_json::Value::Null,
    };
    emit_with_receipt(&rendered, args.output.as_ref(), receipt)
}

fn resolve_runtime_rules(
    runtime: &str,
    runtime_rules: Option<&PathBuf>,
    core_classes: Option<&PathBuf>,
    verbose: bool,
) -> Result<clankerprof_core::rules::RuntimeRuleSet, String> {
    use clankerprof_core::rules;
    let classes = match core_classes {
        Some(path) => rules::load_ruby_core_classes(path)?,
        None if runtime == "ruby" => rules::load_default_ruby_core_classes(),
        None => std::collections::BTreeSet::new(),
    };
    // Verbose reaches only external packs and the ruby pack, mirroring the
    // Python reference: the packaged generic default never applies it.
    if let Some(path) = runtime_rules {
        return rules::load_runtime_rules_file(path, classes, verbose);
    }
    match runtime {
        "generic" => Ok(rules::RuntimeRuleSet::generic().clone()),
        "ruby" => rules::ruby_rules(classes, verbose),
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
                .ok_or_else(|| format!("Attributable column {name} values must be numbers."))?;
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
    if let Some(layout) = args.target_csv_layout.as_deref() {
        if !matches!(layout, "standard" | "compat") {
            return Err(format!("Unsupported --target-csv-layout: {layout}."));
        }
    }
    if args.legacy_target_csv_layout && args.target_csv_layout.as_deref() == Some("standard") {
        return Err(
            "--legacy-target-csv-layout conflicts with --target-csv-layout=standard.".to_string(),
        );
    }
    let compat_layout =
        args.legacy_target_csv_layout || args.target_csv_layout.as_deref() == Some("compat");
    if compat_layout && (args.format != "csv" || args.output.is_none()) {
        return Err("--target-csv-layout=compat requires --format csv and --output.".to_string());
    }
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
        args.verbose_runtime_internals,
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
    // Fail closed on invalid user patterns before writing any artifact.
    clankerprof_core::targets::take_pattern_error()?;
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
        clankerprof_core::targets::take_pattern_error()?;
        let rendered = clankerprof_core::pyjson::dumps_pretty(&payload);
        return emit(&rendered, args.output.as_ref(), "clankerprof_targets");
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
    clankerprof_core::targets::take_pattern_error()?;
    emit(&rendered, args.output.as_ref(), "clankerprof_targets")
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
    let simplified_path = output_dir.join(base_name);
    let verbose_path = verbose_dir.join(base_name);
    std::fs::write(
        &simplified_path,
        render_target_csv(results, attributables, true, true),
    )
    .map_err(|error| error.to_string())?;
    std::fs::write(
        &verbose_path,
        render_target_csv(results, attributables, false, true),
    )
    .map_err(|error| error.to_string())?;
    // Rich receipt matching the Python CLI's compat-layout payload.
    let receipt = serde_json::json!({
        "compat_target_csv_layout": true,
        "legacy_target_csv_layout": true,
        "ok": true,
        "output": simplified_path.to_string_lossy(),
        "outputs": {
            "simplified_csv": simplified_path.to_string_lossy(),
            "verbose_csv": verbose_path.to_string_lossy(),
        },
        "tool": "clankerprof_targets",
    });
    let receipt_rendered = clankerprof_core::pyjson::dumps_pretty(&receipt);
    println!("{receipt_rendered}");
    Ok(())
}

fn run_slices(args: SlicesArgs) -> Result<(), String> {
    // The merge below mirrors the Python reference (run_slices in
    // clankerprof/cli.py) statement for statement, including the order in
    // which validation errors surface.
    let config = load_slices_config(args.config.as_ref())?;
    let raw_profile = merge_single_value(
        args.profile.as_ref().map(path_string),
        config.get("profile"),
        "profile",
    )?;
    let raw_facts = merge_single_value(
        args.facts.as_ref().map(path_string),
        config.get("facts"),
        "facts",
    )?;
    let profile_path = raw_profile.map(PathBuf::from);
    let facts_path = raw_facts.map(PathBuf::from);
    let facts = load_projection_input(profile_path.as_ref(), facts_path.as_ref())?;
    let raw_slices = merge_single_value(
        args.slices.as_ref().map(path_string),
        config.get("slices"),
        "slices",
    )?;
    let cli_top = int64_flag(args.top.as_deref(), TOP_INT_MESSAGE)?;
    let raw_top = optional_config_int(&config, "top")?;
    if cli_top.is_some() && raw_top.is_some() {
        return Err("top specified both on command line and in config file.".to_string());
    }
    let top = cli_top.or(raw_top);
    let raw_by_slice = optional_config_by_slice(&config)?;
    if args.by_slice.is_some() && raw_by_slice.is_some() {
        return Err("by_slice specified both on command line and in config file.".to_string());
    }
    let by_slice = args.by_slice.clone().or(raw_by_slice);
    let raw_show_paths = optional_config_bool(&config, "show_paths")?;
    if args.show_paths && raw_show_paths.is_some() {
        return Err("show_paths specified both on command line and in config file.".to_string());
    }
    let show_paths = args.show_paths || raw_show_paths.unwrap_or(false);
    let raw_no_collapse_native = optional_config_bool(&config, "no_collapse_native")?;
    if args.no_collapse_native && raw_no_collapse_native.is_some() {
        return Err(
            "no_collapse_native specified both on command line and in config file.".to_string(),
        );
    }
    let no_collapse_native = args.no_collapse_native || raw_no_collapse_native.unwrap_or(false);
    let cli_unattributed_libraries = int64_flag(
        args.unattributed_libraries.as_deref(),
        UNATTRIBUTED_LIBRARIES_INT_MESSAGE,
    )?;
    let raw_unattributed_libraries = optional_config_unattributed(&config)?;
    if cli_unattributed_libraries.is_some() && raw_unattributed_libraries.is_some() {
        return Err(
            "unattributed_libraries specified both on command line and in config file \
             (--unattributed-gems is a compatibility alias)."
                .to_string(),
        );
    }
    let unattributed_libraries = cli_unattributed_libraries.or(raw_unattributed_libraries);
    let mut filters = config_string_array(&config, "filters")?;
    filters.extend(config_string_array(&config, "filter")?);
    filters.extend(args.filter.iter().cloned());
    let mut collapse = config_string_array(&config, "collapse")?;
    collapse.extend(args.collapse.iter().cloned());
    let mut raw_attributes = config_string_array(&config, "attribute")?;
    raw_attributes.extend(args.attribute.iter().cloned());
    let mut attributes = Vec::new();
    for raw in &raw_attributes {
        attributes.push(parse_attribute(raw)?);
    }
    let mut slices_path = raw_slices.map(PathBuf::from);
    let slice_aware = slices_path.is_some()
        || by_slice.is_some()
        || !attributes.is_empty()
        || filters
            .iter()
            .chain(&collapse)
            .any(|item| item.trim_start_matches(['!', '<']).starts_with("slice:"));
    if slices_path.is_none() && slice_aware {
        let default_slices = std::path::Path::new("slices.yml");
        if default_slices.exists() {
            slices_path = Some(default_slices.to_path_buf());
        }
    }
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
        args.verbose_runtime_internals,
    )?;
    let mut options = SliceAnalysisOptions {
        filters,
        collapse,
        attributes,
        top,
        by_slice,
        show_paths,
        no_collapse_native,
        unattributed_libraries,
        runtime_rules,
        ..SliceAnalysisOptions::default()
    };
    if let Some(slices_path) = slices_path {
        options.slices = load_slices_file(slices_path)?;
    }
    validate_slice_options(&options, args.allow_virtual_attribute_slices)?;
    let payload = render_slice_json(&analyze_slice_facts(&facts, &options), &options)?;
    clankerprof_core::targets::take_pattern_error()?;
    let rendered = clankerprof_core::pyjson::dumps_pretty(&payload);
    emit(&rendered, args.output.as_ref(), "clankerprof_slices")
}

fn path_string(path: &PathBuf) -> String {
    path.to_string_lossy().into_owned()
}

/// Mirror of Python `_load_slices_config`: TOML by suffix, YAML otherwise,
/// root must be a mapping. Non-string YAML keys are dropped, matching how the
/// Python dict is only ever read through string keys.
fn load_slices_config(
    path: Option<&PathBuf>,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    const MESSAGE: &str = "Slice config file must be a YAML object.";
    let Some(path) = path else {
        return Ok(serde_json::Map::new());
    };
    let payload = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value = if path.extension().and_then(|extension| extension.to_str()) == Some("toml") {
        let parsed: toml::Value = toml::from_str(&payload).map_err(|error| error.to_string())?;
        toml_to_json(parsed)
    } else {
        let parsed: serde_yaml::Value =
            serde_yaml::from_str(&payload).map_err(|error| error.to_string())?;
        clankerprof_core::rules::require_string_keys(&parsed)?;
        yaml_to_json(parsed)
    };
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(MESSAGE.to_string()),
    }
}

fn json_float(value: f64) -> serde_json::Value {
    // Python floats flow through unchanged; non-finite values become their
    // Python str() forms so downstream coercion errors mirror the reference.
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .unwrap_or_else(|| {
            serde_json::Value::String(if value.is_nan() {
                "nan".to_string()
            } else if value > 0.0 {
                "inf".to_string()
            } else {
                "-inf".to_string()
            })
        })
}

fn yaml_to_json(value: serde_yaml::Value) -> serde_json::Value {
    match value {
        serde_yaml::Value::Null => serde_json::Value::Null,
        serde_yaml::Value::Bool(item) => serde_json::Value::Bool(item),
        serde_yaml::Value::Number(number) => {
            if let Some(int) = number.as_i64() {
                serde_json::Value::from(int)
            } else if let Some(unsigned) = number.as_u64() {
                serde_json::Value::from(unsigned)
            } else {
                json_float(number.as_f64().unwrap_or(f64::NAN))
            }
        }
        serde_yaml::Value::String(item) => serde_json::Value::String(item),
        serde_yaml::Value::Sequence(items) => {
            serde_json::Value::Array(items.into_iter().map(yaml_to_json).collect())
        }
        serde_yaml::Value::Mapping(mapping) => serde_json::Value::Object(
            mapping
                .into_iter()
                .filter_map(|(key, item)| match key {
                    serde_yaml::Value::String(key) => Some((key, yaml_to_json(item))),
                    _ => None,
                })
                .collect(),
        ),
        serde_yaml::Value::Tagged(tagged) => yaml_to_json(tagged.value),
    }
}

fn toml_to_json(value: toml::Value) -> serde_json::Value {
    match value {
        toml::Value::String(item) => serde_json::Value::String(item),
        toml::Value::Integer(item) => serde_json::Value::from(item),
        toml::Value::Float(item) => json_float(item),
        toml::Value::Boolean(item) => serde_json::Value::Bool(item),
        toml::Value::Datetime(item) => serde_json::Value::String(item.to_string()),
        toml::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(toml_to_json).collect())
        }
        toml::Value::Table(table) => serde_json::Value::Object(
            table
                .into_iter()
                .map(|(key, item)| (key, toml_to_json(item)))
                .collect(),
        ),
    }
}

/// str() coercion for config scalars, matching Python's `_merge_single_value`
/// and `_string_array` (which stringify whatever the config holds).
fn python_str(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(item) => item.clone(),
        serde_json::Value::Bool(true) => "True".to_string(),
        serde_json::Value::Bool(false) => "False".to_string(),
        serde_json::Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn merge_single_value(
    cli_value: Option<String>,
    config_value: Option<&serde_json::Value>,
    name: &str,
) -> Result<Option<String>, String> {
    let config_value = config_value.filter(|value| !value.is_null());
    if cli_value.is_some() && config_value.is_some() {
        return Err(format!(
            "{name} specified both on command line and in config file."
        ));
    }
    if cli_value.is_some() {
        return Ok(cli_value);
    }
    Ok(config_value.map(python_str))
}

/// Mirror of Python `_optional_int`: bools and non-integral floats are
/// rejected; magnitudes beyond i64 clamp, which is equivalent for a
/// truncation limit (Python's unbounded int slices the same way).
fn optional_config_int(
    config: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<i64>, String> {
    let message = format!("{key} in slice config must be an integer.");
    optional_int_value(config.get(key), &message)
}

fn optional_int_value(
    value: Option<&serde_json::Value>,
    message: &str,
) -> Result<Option<i64>, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(_)) => Err(message.to_string()),
        Some(serde_json::Value::Number(number)) => {
            if let Some(int) = number.as_i64() {
                Ok(Some(int))
            } else if number.as_u64().is_some() {
                Ok(Some(i64::MAX))
            } else if let Some(float) = number.as_f64() {
                if float.is_finite() && float.fract() == 0.0 {
                    Ok(Some(float.clamp(i64::MIN as f64, i64::MAX as f64) as i64))
                } else {
                    Err(message.to_string())
                }
            } else {
                Err(message.to_string())
            }
        }
        Some(serde_json::Value::String(raw)) => raw
            .trim()
            .parse::<i64>()
            .map(Some)
            .map_err(|_| message.to_string()),
        Some(_) => Err(message.to_string()),
    }
}

fn optional_config_bool(
    config: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<bool>, String> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(item)) => Ok(Some(*item)),
        Some(_) => Err(format!("{key} in slice config must be a boolean.")),
    }
}

/// Mirror of Python `_optional_by_slice`: bools toggle the 0.1% default,
/// integral numbers become count limits, non-integral numbers become
/// percentage thresholds, and anything else is stringified.
fn optional_config_by_slice(
    config: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<String>, String> {
    match config.get("by_slice") {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(true)) => Ok(Some("0.1%".to_string())),
        Some(serde_json::Value::Bool(false)) => Ok(None),
        Some(serde_json::Value::Number(number)) => {
            if let Some(int) = number.as_i64() {
                Ok(Some(int.to_string()))
            } else if let Some(unsigned) = number.as_u64() {
                Ok(Some(unsigned.to_string()))
            } else if let Some(float) = number.as_f64() {
                if float.fract() == 0.0 && float.is_finite() {
                    Ok(Some(
                        (float.clamp(i64::MIN as f64, i64::MAX as f64) as i64).to_string(),
                    ))
                } else {
                    Ok(Some(format!("{float}%")))
                }
            } else {
                Ok(Some(number.to_string()))
            }
        }
        Some(other) => Ok(Some(python_str(other))),
    }
}

fn optional_config_unattributed(
    config: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<i64>, String> {
    let present: Vec<(&str, &serde_json::Value)> = ["unattributed_gems", "unattributed_libraries"]
        .iter()
        .filter_map(|key| {
            config
                .get(*key)
                .filter(|value| !value.is_null())
                .map(|value| (*key, value))
        })
        .collect();
    let [(key, value)] = present.as_slice() else {
        if present.is_empty() {
            return Ok(None);
        }
        return Err(
            "unattributed_gems and unattributed_libraries are aliases; use only one.".to_string(),
        );
    };
    match value {
        serde_json::Value::Bool(true) => Ok(Some(i64::MAX)),
        serde_json::Value::Bool(false) => Ok(None),
        other => {
            let message = format!("{key} in slice config must be an integer.");
            optional_int_value(Some(other), &message)
        }
    }
}

fn config_string_array(
    config: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Vec<String>, String> {
    match config.get(key) {
        None | Some(serde_json::Value::Null) => Ok(Vec::new()),
        Some(serde_json::Value::Array(items)) => Ok(items.iter().map(python_str).collect()),
        Some(_) => Err(format!("{key} in slice config must be an array.")),
    }
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
    let rendered = clankerprof_core::pyjson::dumps_pretty(&payload);
    // The receipt keeps the regression gate intact: has_regression rides
    // along and the exit code is unchanged by --output.
    let receipt = match args.output.as_ref() {
        Some(output_path) => serde_json::json!({
            "has_regression": has_regression,
            "ok": true,
            "output": output_path.to_string_lossy(),
            "tool": "clankerprof_compare",
        }),
        None => serde_json::Value::Null,
    };
    emit_with_receipt(&rendered, args.output.as_ref(), receipt)?;
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

#[cfg(test)]
mod tests {
    use super::{
        hoist_global_output, merge_single_value, optional_config_by_slice, optional_config_int,
        optional_config_unattributed,
    };

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(ToString::to_string).collect()
    }

    fn config(payload: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        match payload {
            serde_json::Value::Object(map) => map,
            _ => unreachable!("test config must be an object"),
        }
    }

    #[test]
    fn config_merge_mirrors_python_optional_value_coercions() {
        let map = config(serde_json::json!({
            "top": 3.0,
            "by_slice": 0.5,
            "unattributed_libraries": true,
        }));
        assert_eq!(optional_config_int(&map, "top"), Ok(Some(3)));
        assert_eq!(optional_config_by_slice(&map), Ok(Some("0.5%".to_string())));
        assert_eq!(optional_config_unattributed(&map), Ok(Some(i64::MAX)));
        let bad = config(serde_json::json!({"top": true}));
        assert_eq!(
            optional_config_int(&bad, "top"),
            Err("top in slice config must be an integer.".to_string())
        );
        let fractional = config(serde_json::json!({"top": 2.5}));
        assert_eq!(
            optional_config_int(&fractional, "top"),
            Err("top in slice config must be an integer.".to_string())
        );
        let aliases = config(serde_json::json!({
            "unattributed_gems": 1,
            "unattributed_libraries": 2,
        }));
        assert_eq!(
            optional_config_unattributed(&aliases),
            Err(
                "unattributed_gems and unattributed_libraries are aliases; use only one."
                    .to_string()
            )
        );
    }

    #[test]
    fn config_merge_rejects_values_set_in_both_places() {
        let error = merge_single_value(
            Some("cli.pb".to_string()),
            Some(&serde_json::json!("config.pb")),
            "profile",
        )
        .expect_err("must reject");
        assert_eq!(
            error,
            "profile specified both on command line and in config file."
        );
        let merged = merge_single_value(None, Some(&serde_json::json!("config.pb")), "profile")
            .expect("merge");
        assert_eq!(merged, Some("config.pb".to_string()));
    }

    #[test]
    fn hoist_moves_leading_global_output_after_the_subcommand() {
        let hoisted = hoist_global_output(args(&[
            "bin", "--output", "out.json", "targets", "--target", "T",
        ]))
        .expect("hoist");
        assert_eq!(
            hoisted,
            args(&["bin", "targets", "--target", "T", "--output", "out.json"])
        );
    }

    #[test]
    fn hoist_supports_equals_form_and_leaves_local_flags_alone() {
        let hoisted = hoist_global_output(args(&[
            "bin",
            "--output=global.json",
            "compare",
            "--before",
            "b.json",
        ]))
        .expect("hoist");
        assert_eq!(
            hoisted,
            args(&[
                "bin",
                "compare",
                "--before",
                "b.json",
                "--output=global.json"
            ])
        );
        let untouched = hoist_global_output(args(&["bin", "targets", "--output", "local.json"]))
            .expect("hoist");
        assert_eq!(
            untouched,
            args(&["bin", "targets", "--output", "local.json"])
        );
    }

    #[test]
    fn hoist_rejects_global_output_without_a_path() {
        let error = hoist_global_output(args(&["bin", "--output"])).expect_err("must fail");
        assert_eq!(error, "--output requires a path argument.");
    }
}
