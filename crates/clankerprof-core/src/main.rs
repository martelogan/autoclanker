use clankerprof_core::compare::{compare_slice_json, CompareOptions};
use clankerprof_core::slices::{
    analyze_slice_facts, load_slices_file, render_slice_json, SliceAnalysisOptions,
};
use clankerprof_core::targets::{
    analyze_target_facts, parse_target_config_json, render_target_json,
};
use clankerprof_core::{load_profile, sample_facts_to_pretty_json};
use std::collections::BTreeSet;
use std::env;
use std::path::PathBuf;

fn main() {
    match run() {
        Ok(()) => {}
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

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(usage());
    };
    if command == "facts" {
        return run_facts(args);
    }
    if command == "targets" {
        return run_targets(args);
    }
    if command == "slices" {
        return run_slices(args);
    }
    if command == "compare" {
        return run_compare(args);
    }
    Err(format!("Unsupported command: {command}. {}", usage()))
}

fn run_facts(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut profile: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                let Some(value) = args.next() else {
                    return Err("--profile requires a path.".to_string());
                };
                profile = Some(PathBuf::from(value));
            }
            "--output" => {
                let Some(value) = args.next() else {
                    return Err("--output requires a path.".to_string());
                };
                output = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => return Err(format!("Unexpected argument: {arg}. {}", usage())),
        }
    }

    let Some(profile_path) = profile else {
        return Err("--profile is required.".to_string());
    };
    let profile = load_profile(profile_path).map_err(|error| error.to_string())?;
    let rendered = sample_facts_to_pretty_json(&profile.to_sample_facts())
        .map_err(|error| error.to_string())?;
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_targets(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut profile: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                let Some(value) = args.next() else {
                    return Err("--profile requires a path.".to_string());
                };
                profile = Some(PathBuf::from(value));
            }
            "--config" => {
                let Some(value) = args.next() else {
                    return Err("--config requires a path.".to_string());
                };
                config = Some(PathBuf::from(value));
            }
            "--output" => {
                let Some(value) = args.next() else {
                    return Err("--output requires a path.".to_string());
                };
                output = Some(PathBuf::from(value));
            }
            "--format" => {
                let Some(value) = args.next() else {
                    return Err("--format requires a value.".to_string());
                };
                if value != "json" {
                    return Err(
                        "clankerprof-rs targets currently supports --format json.".to_string()
                    );
                }
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => return Err(format!("Unexpected argument: {arg}. {}", usage())),
        }
    }

    let Some(profile_path) = profile else {
        return Err("--profile is required.".to_string());
    };
    let Some(config_path) = config else {
        return Err("--config is required.".to_string());
    };
    let config_payload = std::fs::read_to_string(config_path).map_err(|error| error.to_string())?;
    let config = parse_target_config_json(&config_payload)?;
    let profile = load_profile(profile_path).map_err(|error| error.to_string())?;
    let payload = render_target_json(&analyze_target_facts(&profile.to_sample_facts(), &config));
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_slices(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut profile: Option<PathBuf> = None;
    let mut slices: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut options = SliceAnalysisOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                let Some(value) = args.next() else {
                    return Err("--profile requires a path.".to_string());
                };
                profile = Some(PathBuf::from(value));
            }
            "--slices" => {
                let Some(value) = args.next() else {
                    return Err("--slices requires a path.".to_string());
                };
                slices = Some(PathBuf::from(value));
            }
            "--filter" => {
                let Some(value) = args.next() else {
                    return Err("--filter requires a value.".to_string());
                };
                options.filters.push(value);
            }
            "--collapse" => {
                let Some(value) = args.next() else {
                    return Err("--collapse requires a value.".to_string());
                };
                options.collapse.push(value);
            }
            "--top" => {
                let Some(value) = args.next() else {
                    return Err("--top requires an integer.".to_string());
                };
                options.top = Some(value.parse::<usize>().map_err(|error| error.to_string())?);
            }
            "--by-slice" => {
                options.by_slice = Some(args.next().unwrap_or_else(|| "0.1%".to_string()));
            }
            "--show-paths" => options.show_paths = true,
            "--no-collapse-native" => options.no_collapse_native = true,
            "--unattributed-libraries" | "--unattributed-gems" => {
                options.unattributed_libraries = Some(usize::MAX);
            }
            "--output" => {
                let Some(value) = args.next() else {
                    return Err("--output requires a path.".to_string());
                };
                output = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => return Err(format!("Unexpected argument: {arg}. {}", usage())),
        }
    }
    if let Some(slices_path) = slices {
        options.slices = load_slices_file(slices_path)?;
    }
    let Some(profile_path) = profile else {
        return Err("--profile is required.".to_string());
    };
    let profile = load_profile(profile_path).map_err(|error| error.to_string())?;
    let payload = render_slice_json(
        &analyze_slice_facts(&profile.to_sample_facts(), &options),
        &options,
    );
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    } else {
        println!("{rendered}");
    }
    Ok(())
}

fn run_compare(mut args: impl Iterator<Item = String>) -> Result<(), String> {
    let mut before: Option<PathBuf> = None;
    let mut after: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut options = CompareOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--before" => {
                let Some(value) = args.next() else {
                    return Err("--before requires a path.".to_string());
                };
                before = Some(PathBuf::from(value));
            }
            "--after" => {
                let Some(value) = args.next() else {
                    return Err("--after requires a path.".to_string());
                };
                after = Some(PathBuf::from(value));
            }
            "--threshold-abs" => {
                let Some(value) = args.next() else {
                    return Err("--threshold-abs requires a number.".to_string());
                };
                options.threshold_abs = value.parse::<f64>().map_err(|error| error.to_string())?;
            }
            "--threshold-rel" => {
                let Some(value) = args.next() else {
                    return Err("--threshold-rel requires a number.".to_string());
                };
                options.threshold_rel = value.parse::<f64>().map_err(|error| error.to_string())?;
            }
            "--focus-slices" => {
                let Some(value) = args.next() else {
                    return Err("--focus-slices requires a comma-delimited value.".to_string());
                };
                options.focus_slices = value
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(ToString::to_string)
                    .collect::<BTreeSet<_>>();
            }
            "--output" => {
                let Some(value) = args.next() else {
                    return Err("--output requires a path.".to_string());
                };
                output = Some(PathBuf::from(value));
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            _ => return Err(format!("Unexpected argument: {arg}. {}", usage())),
        }
    }
    let Some(before_path) = before else {
        return Err("--before is required.".to_string());
    };
    let Some(after_path) = after else {
        return Err("--after is required.".to_string());
    };
    let before_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(before_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let after_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(after_path).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let payload = compare_slice_json(&before_payload, &after_payload, &options);
    let has_regression = payload
        .get("has_regression")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let rendered = serde_json::to_string_pretty(&payload).map_err(|error| error.to_string())?;
    if let Some(output_path) = output {
        std::fs::write(output_path, format!("{rendered}\n")).map_err(|error| error.to_string())?;
    } else {
        println!("{rendered}");
    }
    if has_regression {
        return Err("comparison detected a regression".to_string());
    }
    Ok(())
}

fn usage() -> String {
    "Usage: clankerprof-rs <facts|targets|slices|compare> --profile <profile.pb|profile.pb.gz> [--config <target-config.json>] [--slices <slices.yml>] [--output <out.json>]"
        .to_string()
}
