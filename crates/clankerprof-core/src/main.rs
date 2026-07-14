use clankerprof_core::compare::{compare_json, CompareOptions};
use clankerprof_core::slices::{
    analyze_slice_facts, load_slices_file, render_slice_json, SliceAnalysisOptions,
};
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
    profile: PathBuf,
    /// JSON target config mapping parent functions to category patterns.
    #[arg(long)]
    config: PathBuf,
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
    /// Output format (only json is supported by clankerprof-rs).
    #[arg(long, default_value = "json")]
    format: String,
}

#[derive(Args)]
struct SlicesArgs {
    /// Raw or gzipped pprof profile path.
    #[arg(long)]
    profile: PathBuf,
    /// Slice definitions YAML file.
    #[arg(long)]
    slices: Option<PathBuf>,
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
    /// Report unattributed dependency libraries for the default slice.
    #[arg(long, visible_alias = "unattributed-gems")]
    unattributed_libraries: bool,
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
    }
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

fn run_targets(args: TargetsArgs) -> Result<(), String> {
    if args.format != "json" {
        return Err("clankerprof-rs targets currently supports --format json.".to_string());
    }
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
    )?;
    let config_payload =
        std::fs::read_to_string(&args.config).map_err(|error| error.to_string())?;
    let config = parse_target_config_json(&config_payload)?;
    let profile = load_profile(&args.profile).map_err(|error| error.to_string())?;
    let options = TargetAnalysisOptions {
        runtime_rules,
        ..TargetAnalysisOptions::default()
    };
    let payload = render_target_json(&analyze_target_facts_with_options(
        &profile.to_sample_facts(),
        &config,
        &options,
    ));
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    emit(&rendered, args.output.as_ref())
}

fn run_slices(args: SlicesArgs) -> Result<(), String> {
    let runtime_rules = resolve_runtime_rules(
        &args.runtime,
        args.runtime_rules.as_ref(),
        args.core_classes.as_ref(),
    )?;
    let mut options = SliceAnalysisOptions {
        filters: args.filter,
        collapse: args.collapse,
        top: args.top,
        by_slice: args.by_slice,
        show_paths: args.show_paths,
        no_collapse_native: args.no_collapse_native,
        unattributed_libraries: args.unattributed_libraries.then_some(usize::MAX),
        runtime_rules,
        ..SliceAnalysisOptions::default()
    };
    if let Some(slices_path) = args.slices {
        options.slices = load_slices_file(slices_path)?;
    }
    let profile = load_profile(&args.profile).map_err(|error| error.to_string())?;
    let payload = render_slice_json(
        &analyze_slice_facts(&profile.to_sample_facts(), &options),
        &options,
    );
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    emit(&rendered, args.output.as_ref())
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
