use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq)]
pub struct CompareOptions {
    pub threshold_abs: f64,
    pub threshold_rel: f64,
    pub focus_slices: BTreeSet<String>,
    pub focus_boundaries: BTreeSet<String>,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self {
            threshold_abs: 2.0,
            threshold_rel: 15.0,
            focus_slices: BTreeSet::new(),
            focus_boundaries: BTreeSet::new(),
        }
    }
}

/// Dispatch on the shared `tool` field, mirroring the Python contract:
/// both inputs must be the same projection and one of the comparable report
/// kinds.
pub fn compare_json(
    before: &Value,
    after: &Value,
    options: &CompareOptions,
) -> Result<Value, String> {
    let before_tool = before.get("tool").and_then(Value::as_str);
    let after_tool = after.get("tool").and_then(Value::as_str);
    if before_tool != after_tool {
        return Err("Compare inputs must use the same clankerprof projection.".to_string());
    }
    match before_tool {
        Some("clankerprof_boundaries") => Ok(compare_boundary_json(before, after, options)),
        Some("clankerprof_slices") => Ok(compare_slice_json(before, after, options)),
        other => Err(format!(
            "Compare inputs must be clankerprof_slices or clankerprof_boundaries reports; got tool {other:?}."
        )),
    }
}

pub fn compare_slice_json(before: &Value, after: &Value, options: &CompareOptions) -> Value {
    let before_slices = slice_map(before);
    let after_slices = slice_map(after);
    let names: BTreeSet<_> = before_slices
        .keys()
        .chain(after_slices.keys())
        .cloned()
        .collect();
    let mut slice_deltas = Vec::new();
    let mut frame_deltas_all = Vec::new();
    let mut has_regression = false;

    for name in names {
        let before_payload = before_slices.get(&name);
        let after_payload = after_slices.get(&name);
        let before_pct = before_payload
            .map(|value| number(value, "pct"))
            .unwrap_or(0.0);
        let after_pct = after_payload
            .map(|value| number(value, "pct"))
            .unwrap_or(0.0);
        let delta_abs = after_pct - before_pct;
        let delta_rel = if before_pct > 0.0 {
            delta_abs / before_pct * 100.0
        } else if after_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let in_focus = options.focus_slices.is_empty() || options.focus_slices.contains(&name);
        let is_regression =
            in_focus && delta_abs > options.threshold_abs && delta_rel > options.threshold_rel;
        let is_improvement =
            delta_abs < -options.threshold_abs && delta_rel < -options.threshold_rel;
        if is_regression {
            has_regression = true;
        }
        let status = if is_regression {
            "regression"
        } else if is_improvement {
            "improvement"
        } else {
            "stable"
        };

        let before_frames = before_payload.map(frames_by_function).unwrap_or_default();
        let after_frames = after_payload.map(frames_by_function).unwrap_or_default();
        let functions: BTreeSet<_> = before_frames
            .keys()
            .chain(after_frames.keys())
            .cloned()
            .collect();
        let mut frame_deltas = Vec::new();
        for function in functions {
            let before_frame_pct = before_frames.get(&function).copied().unwrap_or(0.0);
            let after_frame_pct = after_frames.get(&function).copied().unwrap_or(0.0);
            frame_deltas.push(json!({
                "after_pct": after_frame_pct,
                "before_pct": before_frame_pct,
                "delta_abs": after_frame_pct - before_frame_pct,
                "function": function,
                "slice": name,
            }));
        }
        frame_deltas.sort_by(|left, right| {
            abs_number(right, "delta_abs").total_cmp(&abs_number(left, "delta_abs"))
        });
        frame_deltas_all.extend(frame_deltas.clone());
        slice_deltas.push(json!({
            "after_pct": after_pct,
            "before_pct": before_pct,
            "delta_abs": delta_abs,
            "delta_rel": finite_json_number(delta_rel),
            "frame_deltas": frame_deltas,
            "name": name,
            "status": status,
        }));
    }

    slice_deltas.sort_by(|left, right| {
        abs_number(right, "delta_abs").total_cmp(&abs_number(left, "delta_abs"))
    });
    frame_deltas_all
        .sort_by(|left, right| number(right, "delta_abs").total_cmp(&number(left, "delta_abs")));
    let top_regressions = frame_deltas_all
        .iter()
        .filter(|item| number(item, "delta_abs") > 0.1)
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    let top_improvements = frame_deltas_all
        .iter()
        .rev()
        .filter(|item| number(item, "delta_abs") < -0.1)
        .take(10)
        .cloned()
        .collect::<Vec<_>>();

    json!({
        "after_total_ns": summary_total(after),
        "before_total_ns": summary_total(before),
        "has_regression": has_regression,
        "slices": slice_deltas,
        "tool": "clankerprof_compare",
        "top_improvements": top_improvements,
        "top_regressions": top_regressions,
    })
}

fn slice_map(payload: &Value) -> BTreeMap<String, Value> {
    payload
        .get("slices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.is_object())
        .map(|item| (string_field(item, "name"), item.clone()))
        .collect()
}

fn frames_by_function(slice_payload: &Value) -> BTreeMap<String, f64> {
    slice_payload
        .get("frames")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|item| item.is_object())
        .map(|item| (string_field(item, "function"), number(item, "pct")))
        .collect()
}

fn string_field(payload: &Value, key: &str) -> String {
    payload
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn summary_total(payload: &Value) -> i64 {
    payload
        .get("summary")
        .and_then(|summary| summary.get("total_time_ns"))
        .and_then(Value::as_i64)
        .unwrap_or(0)
}

fn number(payload: &Value, key: &str) -> f64 {
    payload.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn abs_number(payload: &Value, key: &str) -> f64 {
    number(payload, key).abs()
}

fn finite_json_number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else {
        Value::Null
    }
}

fn boundary_rows(payload: &Value) -> BTreeMap<(String, String, String), f64> {
    let mut rows = BTreeMap::new();
    let boundaries = payload
        .get("boundaries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    for boundary in boundaries {
        if !boundary.is_object() {
            continue;
        }
        let boundary_name = string_field(boundary, "name");
        rows.insert(
            (
                "boundary".to_string(),
                boundary_name.clone(),
                boundary_name.clone(),
            ),
            number(boundary, "pct_of_profile"),
        );
        for bucket in boundary
            .get("buckets")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if !bucket.is_object() {
                continue;
            }
            rows.insert(
                (
                    "bucket".to_string(),
                    boundary_name.clone(),
                    string_field(bucket, "name"),
                ),
                number(bucket, "pct"),
            );
            for category in bucket
                .get("categories")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if !category.is_object() {
                    continue;
                }
                rows.insert(
                    (
                        "category".to_string(),
                        boundary_name.clone(),
                        string_field(category, "name"),
                    ),
                    number(category, "pct"),
                );
            }
        }
        for domain in boundary
            .get("domains")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if !domain.is_object() {
                continue;
            }
            rows.insert(
                (
                    "domain".to_string(),
                    boundary_name.clone(),
                    string_field(domain, "name"),
                ),
                number(domain, "pct"),
            );
        }
    }
    rows
}

pub fn compare_boundary_json(before: &Value, after: &Value, options: &CompareOptions) -> Value {
    let before_rows = boundary_rows(before);
    let after_rows = boundary_rows(after);
    let keys: BTreeSet<_> = before_rows.keys().chain(after_rows.keys()).collect();
    let mut row_deltas = Vec::new();
    let mut has_regression = false;

    for key in keys {
        let (kind, boundary, name) = key;
        let before_pct = before_rows.get(key).copied().unwrap_or(0.0);
        let after_pct = after_rows.get(key).copied().unwrap_or(0.0);
        let delta_abs = after_pct - before_pct;
        let delta_rel = if before_pct > 0.0 {
            delta_abs / before_pct * 100.0
        } else if after_pct > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let in_focus =
            options.focus_boundaries.is_empty() || options.focus_boundaries.contains(boundary);
        let is_regression =
            in_focus && delta_abs > options.threshold_abs && delta_rel > options.threshold_rel;
        let is_improvement =
            delta_abs < -options.threshold_abs && delta_rel < -options.threshold_rel;
        if is_regression {
            has_regression = true;
        }
        let status = if is_regression {
            "regression"
        } else if is_improvement {
            "improvement"
        } else {
            "stable"
        };
        row_deltas.push(json!({
            "after_pct": after_pct,
            "before_pct": before_pct,
            "boundary": boundary,
            "delta_abs": delta_abs,
            "delta_rel": finite_json_number(delta_rel),
            "kind": kind,
            "name": name,
            "status": status,
        }));
    }

    row_deltas.sort_by(|left, right| {
        abs_number(right, "delta_abs").total_cmp(&abs_number(left, "delta_abs"))
    });
    let top_regressions = row_deltas
        .iter()
        .filter(|item| number(item, "delta_abs") > 0.1)
        .take(10)
        .cloned()
        .collect::<Vec<_>>();
    let mut improvements = row_deltas
        .iter()
        .filter(|item| number(item, "delta_abs") < -0.1)
        .cloned()
        .collect::<Vec<_>>();
    improvements
        .sort_by(|left, right| number(left, "delta_abs").total_cmp(&number(right, "delta_abs")));
    improvements.truncate(10);

    json!({
        "after_total_ns": summary_total(after),
        "before_total_ns": summary_total(before),
        "has_regression": has_regression,
        "projection": "boundaries",
        "rows": row_deltas,
        "tool": "clankerprof_compare",
        "top_improvements": improvements,
        "top_regressions": top_regressions,
    })
}
