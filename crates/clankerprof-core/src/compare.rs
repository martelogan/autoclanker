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
        Some("clankerprof_boundaries") => compare_boundary_json(before, after, options),
        Some("clankerprof_slices") => compare_slice_json(before, after, options),
        other => Err(format!(
            "Compare inputs must be clankerprof_slices or clankerprof_boundaries reports; got tool {other:?}."
        )),
    }
}

fn validate_options(options: &CompareOptions) -> Result<(), String> {
    // A non-finite threshold would silently disable gating; a negative one
    // would gate identical reports as regressions (0 > -1).
    if !options.threshold_abs.is_finite()
        || !options.threshold_rel.is_finite()
        || options.threshold_abs < 0.0
        || options.threshold_rel < 0.0
    {
        return Err("Compare thresholds must be finite, non-negative numbers.".to_string());
    }
    Ok(())
}

/// Derived compare values (frame-percentage sums and absolute deltas) must
/// stay finite; overflow fails closed rather than serializing an uncontracted
/// null. Only `delta_rel` has a documented null path.
fn finite_value(value: f64, name: &str) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("Compare values for '{name}' are not finite."))
    }
}

/// Relative delta against the magnitude of the baseline (mirrors Python
/// `_delta_rel`). Sample values are signed, so rows can carry negative
/// percentages; dividing by `before_pct.abs()` keeps the sign of the change
/// meaningful (-10% -> -5% is a +50% increase and must be gateable). A zero
/// baseline yields an unbounded delta in the direction of the absolute
/// change, serialized as null by `finite_json_number`. Positive baselines
/// are bit-identical to the plain `delta/before` form.
fn delta_rel(before_pct: f64, after_pct: f64) -> f64 {
    let delta_abs = after_pct - before_pct;
    if before_pct != 0.0 {
        delta_abs / before_pct.abs() * 100.0
    } else if delta_abs > 0.0 {
        f64::INFINITY
    } else if delta_abs < 0.0 {
        f64::NEG_INFINITY
    } else {
        0.0
    }
}

pub fn compare_slice_json(
    before: &Value,
    after: &Value,
    options: &CompareOptions,
) -> Result<Value, String> {
    validate_options(options)?;
    let before_slices = slice_map(before)?;
    let after_slices = slice_map(after)?;
    let names: BTreeSet<_> = before_slices
        .keys()
        .chain(after_slices.keys())
        .cloned()
        .collect();
    // A focus name matching zero rows would silently disable gating for it,
    // exactly like a non-finite threshold; the union keeps focusing a row
    // that was added or removed between reports legal. BTreeSet iteration is
    // byte-wise UTF-8 order, matching Python sorted() code points.
    let unknown_focus: Vec<String> = options
        .focus_slices
        .iter()
        .filter(|name| !names.contains(*name))
        .map(|name| format!("'{name}'"))
        .collect();
    if !unknown_focus.is_empty() {
        return Err(format!(
            "Focus slices not present in either report: {}.",
            unknown_focus.join(", ")
        ));
    }
    let mut slice_deltas = Vec::new();
    let mut frame_deltas_all = Vec::new();
    let mut has_regression = false;

    for name in names {
        let before_payload = before_slices.get(&name);
        let after_payload = after_slices.get(&name);
        let before_pct = before_payload
            .map(|value| require_number(value, "pct", "Slice"))
            .transpose()?
            .unwrap_or(0.0);
        let after_pct = after_payload
            .map(|value| require_number(value, "pct", "Slice"))
            .transpose()?
            .unwrap_or(0.0);
        let delta_abs = finite_value(after_pct - before_pct, &name)?;
        let delta_rel = delta_rel(before_pct, after_pct);
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

        let before_frames = before_payload
            .map(frames_by_function)
            .transpose()?
            .unwrap_or_default();
        let after_frames = after_payload
            .map(frames_by_function)
            .transpose()?
            .unwrap_or_default();
        let functions: BTreeSet<_> = before_frames
            .keys()
            .chain(after_frames.keys())
            .cloned()
            .collect();
        // Finiteness is checked while walking the sorted union so both
        // languages report the same first offender when several overflow.
        let mut frame_deltas = Vec::new();
        for function in functions {
            let before_frame_pct = finite_value(
                before_frames.get(&function).copied().unwrap_or(0.0),
                &function,
            )?;
            let after_frame_pct = finite_value(
                after_frames.get(&function).copied().unwrap_or(0.0),
                &function,
            )?;
            frame_deltas.push(json!({
                "after_pct": after_frame_pct,
                "before_pct": before_frame_pct,
                "delta_abs": finite_value(after_frame_pct - before_frame_pct, &function)?,
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

    Ok(json!({
        "after_total_ns": summary_total(after)?,
        "before_total_ns": summary_total(before)?,
        "has_regression": has_regression,
        "slices": slice_deltas,
        "tool": "clankerprof_compare",
        "top_improvements": top_improvements,
        "top_regressions": top_regressions,
    }))
}

fn slice_map(payload: &Value) -> Result<BTreeMap<String, Value>, String> {
    let items = payload
        .get("slices")
        .and_then(Value::as_array)
        .ok_or_else(|| "Profile comparison input must contain a slices array.".to_string())?;
    let mut result = BTreeMap::new();
    for item in items {
        if !item.is_object() {
            return Err("Slice rows must be objects.".to_string());
        }
        let name = require_name(item, "name", "Slice")?;
        if result.insert(name.clone(), item.clone()).is_some() {
            // Projections never emit duplicate top-level names; a duplicate is
            // malformed input and last-wins would make the gate order-dependent.
            return Err(format!("Duplicate Slice row '{name}' in comparison input."));
        }
    }
    Ok(result)
}

/// Frames sharing a function name are summed; iteration follows the frames
/// array in both languages, so f64 accumulation order (and byte parity) is
/// preserved. Output ordering never depends on this map's iteration order:
/// both sides walk the sorted union of function names.
fn frames_by_function(slice_payload: &Value) -> Result<BTreeMap<String, f64>, String> {
    let mut frames = BTreeMap::new();
    for item in optional_rows(slice_payload, "frames", "Slice")? {
        let function = require_name(item, "function", "Frame")?;
        let pct = require_number(item, "pct", "Frame")?;
        *frames.entry(function).or_insert(0.0) += pct;
    }
    Ok(frames)
}

fn summary_total(payload: &Value) -> Result<Value, String> {
    let summary = payload.get("summary").unwrap_or(&Value::Null);
    if !summary.is_object() {
        return Err("Report summary must be an object.".to_string());
    }
    match summary.get("total_time_ns") {
        // Valid report totals span [i64::MIN, u64::MAX] (aggregate bound), so
        // preserve the exact JSON integer instead of coercing through i64.
        Some(value) if value.is_i64() || value.is_u64() => Ok(value.clone()),
        // Absent and out-of-range values share the message because this JSON
        // parser cannot distinguish out-of-range integers from non-integers.
        _ => Err("Report summary field 'total_time_ns' must be an integer.".to_string()),
    }
}

/// Rows present in a report must carry their numeric fields; absent or
/// non-numeric is malformed input (mirrors Python `_require_number`).
/// Row-level absence (a name missing from one report entirely) is handled
/// by the callers, never by defaulting here.
fn require_number(payload: &Value, key: &str, context: &str) -> Result<f64, String> {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("{context} field '{key}' must be a number."))
}

fn require_name(payload: &Value, key: &str, context: &str) -> Result<String, String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{context} rows must carry a string '{key}'."))
}

fn row_context(key: &str) -> &'static str {
    match key {
        "slices" => "Slice",
        "frames" => "Frame",
        "boundaries" => "Boundary",
        "buckets" => "Bucket",
        "categories" => "Category",
        _ => "Domain",
    }
}

/// Structural row arrays may be absent, but a present key must be an array
/// of objects — wrong shapes are malformed input, never silently empty.
fn optional_rows<'a>(
    payload: &'a Value,
    key: &str,
    context: &str,
) -> Result<Vec<&'a Value>, String> {
    let Some(raw) = payload.get(key) else {
        return Ok(Vec::new());
    };
    let Some(items) = raw.as_array() else {
        return Err(format!("{context} field '{key}' must be an array."));
    };
    for item in items {
        if !item.is_object() {
            return Err(format!("{} rows must be objects.", row_context(key)));
        }
    }
    Ok(items.iter().collect())
}

/// Read for rows this module constructed itself: delta fields are checked
/// finite before insertion (`finite_value`), so a missing or null value here
/// is an internal invariant violation, never reachable from user input —
/// defaulting it to zero would silently corrupt ranking instead.
fn number(payload: &Value, key: &str) -> f64 {
    payload
        .get(key)
        .and_then(Value::as_f64)
        .expect("compare rows carry finite numeric fields by construction")
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

type BoundaryRowKey = (String, String, String);

fn insert_row(
    rows: &mut BTreeMap<BoundaryRowKey, f64>,
    key: BoundaryRowKey,
    value: f64,
    context: &str,
) -> Result<(), String> {
    let name = key.2.clone();
    if rows.insert(key, value).is_some() {
        // Projections never emit duplicate row names; a duplicate is malformed
        // input and last-wins would make the gate order-dependent.
        return Err(format!(
            "Duplicate {context} row '{name}' in comparison input."
        ));
    }
    Ok(())
}

fn boundary_rows(payload: &Value) -> Result<BTreeMap<BoundaryRowKey, f64>, String> {
    let mut rows = BTreeMap::new();
    let boundaries = payload
        .get("boundaries")
        .and_then(Value::as_array)
        .ok_or_else(|| "Boundary comparison input must contain a boundaries array.".to_string())?;
    for boundary in boundaries {
        if !boundary.is_object() {
            return Err("Boundary rows must be objects.".to_string());
        }
        let boundary_name = require_name(boundary, "name", "Boundary")?;
        insert_row(
            &mut rows,
            (
                "boundary".to_string(),
                boundary_name.clone(),
                boundary_name.clone(),
            ),
            require_number(boundary, "pct_of_profile", "Boundary")?,
            "Boundary",
        )?;
        for bucket in optional_rows(boundary, "buckets", "Boundary")? {
            let bucket_name = require_name(bucket, "name", "Bucket")?;
            insert_row(
                &mut rows,
                ("bucket".to_string(), boundary_name.clone(), bucket_name),
                require_number(bucket, "pct", "Bucket")?,
                "Bucket",
            )?;
            for category in optional_rows(bucket, "categories", "Bucket")? {
                let category_name = require_name(category, "name", "Category")?;
                insert_row(
                    &mut rows,
                    ("category".to_string(), boundary_name.clone(), category_name),
                    require_number(category, "pct", "Category")?,
                    "Category",
                )?;
            }
        }
        for domain in optional_rows(boundary, "domains", "Boundary")? {
            let domain_name = require_name(domain, "name", "Domain")?;
            insert_row(
                &mut rows,
                ("domain".to_string(), boundary_name.clone(), domain_name),
                require_number(domain, "pct", "Domain")?,
                "Domain",
            )?;
        }
    }
    Ok(rows)
}

pub fn compare_boundary_json(
    before: &Value,
    after: &Value,
    options: &CompareOptions,
) -> Result<Value, String> {
    validate_options(options)?;
    let before_rows = boundary_rows(before)?;
    let after_rows = boundary_rows(after)?;
    let keys: BTreeSet<_> = before_rows.keys().chain(after_rows.keys()).collect();
    let boundary_names: BTreeSet<&String> = keys.iter().map(|(_, boundary, _)| boundary).collect();
    let unknown_focus: Vec<String> = options
        .focus_boundaries
        .iter()
        .filter(|name| !boundary_names.contains(name))
        .map(|name| format!("'{name}'"))
        .collect();
    if !unknown_focus.is_empty() {
        return Err(format!(
            "Focus boundaries not present in either report: {}.",
            unknown_focus.join(", ")
        ));
    }
    let mut row_deltas = Vec::new();
    let mut has_regression = false;

    for key in keys {
        let (kind, boundary, name) = key;
        let before_pct = before_rows.get(key).copied().unwrap_or(0.0);
        let after_pct = after_rows.get(key).copied().unwrap_or(0.0);
        let delta_abs = finite_value(after_pct - before_pct, name)?;
        let delta_rel = delta_rel(before_pct, after_pct);
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

    Ok(json!({
        "after_total_ns": summary_total(after)?,
        "before_total_ns": summary_total(before)?,
        "has_regression": has_regression,
        "projection": "boundaries",
        "rows": row_deltas,
        "tool": "clankerprof_compare",
        "top_improvements": improvements,
        "top_regressions": top_regressions,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_report(slices: Value) -> Value {
        json!({
            "tool": "clankerprof_slices",
            "summary": {"total_time_ns": 100},
            "slices": slices,
        })
    }

    #[test]
    fn missing_row_arrays_are_validation_errors() {
        let report = slice_report(json!([]));
        let truncated = json!({"tool": "clankerprof_slices"});
        let error = compare_slice_json(&truncated, &report, &CompareOptions::default())
            .expect_err("missing slices array must be rejected");
        assert!(error.contains("must contain a slices array"));
        let boundary_truncated = json!({"tool": "clankerprof_boundaries"});
        let error = compare_boundary_json(
            &boundary_truncated,
            &boundary_truncated,
            &CompareOptions::default(),
        )
        .expect_err("missing boundaries array must be rejected");
        assert!(error.contains("must contain a boundaries array"));
    }

    #[test]
    fn non_numeric_fields_are_validation_errors() {
        let good = slice_report(json!([{"name": "A", "pct": 10.0, "frames": []}]));
        let bad_pct = slice_report(json!([{"name": "A", "pct": "not-a-number"}]));
        let error = compare_slice_json(&good, &bad_pct, &CompareOptions::default())
            .expect_err("non-numeric pct must be rejected");
        assert_eq!(error, "Slice field 'pct' must be a number.");
        let bad_frame = slice_report(
            json!([{"name": "A", "pct": 10.0, "frames": [{"function": "f", "pct": true}]}]),
        );
        let error = compare_slice_json(&good, &bad_frame, &CompareOptions::default())
            .expect_err("non-numeric frame pct must be rejected");
        assert_eq!(error, "Frame field 'pct' must be a number.");
        let bad_total = json!({
            "tool": "clankerprof_slices",
            "summary": {"total_time_ns": "later"},
            "slices": [],
        });
        let error = compare_slice_json(&bad_total, &good, &CompareOptions::default())
            .expect_err("non-integer summary total must be rejected");
        assert_eq!(
            error,
            "Report summary field 'total_time_ns' must be an integer."
        );
    }

    #[test]
    fn non_finite_or_negative_thresholds_are_validation_errors() {
        let report = slice_report(json!([]));
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, -0.001] {
            let options = CompareOptions {
                threshold_abs: threshold,
                ..CompareOptions::default()
            };
            let error = compare_slice_json(&report, &report, &options)
                .expect_err("bad threshold must be rejected");
            assert_eq!(
                error,
                "Compare thresholds must be finite, non-negative numbers."
            );
            let options = CompareOptions {
                threshold_rel: threshold,
                ..CompareOptions::default()
            };
            compare_boundary_json(
                &json!({"tool": "clankerprof_boundaries", "boundaries": []}),
                &json!({"tool": "clankerprof_boundaries", "boundaries": []}),
                &options,
            )
            .expect_err("bad threshold must be rejected");
        }
        // Zero thresholds stay legal: gating uses strict `>`, so identical
        // reports never regress at zero.
        let identical = slice_report(json!([{"name": "A", "pct": 10.0}]));
        let zero = CompareOptions {
            threshold_abs: 0.0,
            threshold_rel: 0.0,
            ..CompareOptions::default()
        };
        let compared =
            compare_slice_json(&identical, &identical, &zero).expect("zero thresholds are legal");
        assert_eq!(compared["has_regression"], json!(false));
    }

    #[test]
    fn present_null_row_arrays_are_validation_errors() {
        let good = slice_report(json!([{"name": "hot", "pct": 10.0, "frames": []}]));
        let nulled = slice_report(json!([{"name": "hot", "pct": 10.0, "frames": null}]));
        let error = compare_slice_json(&good, &nulled, &CompareOptions::default())
            .expect_err("present null frames must fail");
        assert_eq!(error, "Slice field 'frames' must be an array.");
        let boundary_nulled = json!({
            "tool": "clankerprof_boundaries",
            "summary": {"total_time_ns": 100},
            "boundaries": [{"name": "B", "pct_of_profile": 40.0, "buckets": null}],
        });
        let error = compare_boundary_json(
            &boundary_nulled,
            &boundary_nulled,
            &CompareOptions::default(),
        )
        .expect_err("present null buckets must fail");
        assert_eq!(error, "Boundary field 'buckets' must be an array.");
    }

    #[test]
    fn non_finite_derived_values_fail_closed() {
        // Duplicate-frame summation overflow.
        let before = slice_report(json!([{"name": "A", "pct": 10.0, "frames": []}]));
        let after = slice_report(json!([{"name": "A", "pct": 10.0, "frames": [
            {"function": "f", "pct": 1e308},
            {"function": "f", "pct": 1e308},
        ]}]));
        let error = compare_slice_json(&before, &after, &CompareOptions::default())
            .expect_err("overflowing frame sum must fail");
        assert_eq!(error, "Compare values for 'f' are not finite.");
        // Slice-level delta subtraction overflow.
        let low = slice_report(json!([{"name": "A", "pct": -1e308}]));
        let high = slice_report(json!([{"name": "A", "pct": 1e308}]));
        let error = compare_slice_json(&low, &high, &CompareOptions::default())
            .expect_err("overflowing slice delta must fail");
        assert_eq!(error, "Compare values for 'A' are not finite.");
    }

    #[test]
    fn duplicate_function_frames_aggregate_instead_of_overwrite() {
        let before = slice_report(json!([{"name": "A", "pct": 30.0, "frames": [
            {"function": "f", "filename": "/one", "pct": 10.0},
            {"function": "f", "filename": "/two", "pct": 15.0},
            {"function": "g", "filename": "/g", "pct": 5.0},
        ]}]));
        let after = slice_report(json!([{"name": "A", "pct": 30.0, "frames": [
            {"function": "f", "filename": "/one", "pct": 15.0},
            {"function": "f", "filename": "/two", "pct": 15.0},
            {"function": "g", "filename": "/g", "pct": 0.0},
        ]}]));
        let compared = compare_slice_json(&before, &after, &CompareOptions::default())
            .expect("valid reports compare");
        let deltas = compared["slices"][0]["frame_deltas"]
            .as_array()
            .expect("frame deltas");
        let f_row = deltas
            .iter()
            .find(|row| row["function"] == "f")
            .expect("aggregated f row");
        assert_eq!(f_row["before_pct"], json!(25.0));
        assert_eq!(f_row["after_pct"], json!(30.0));
        assert_eq!(f_row["delta_abs"], json!(5.0));
        let regressions = compared["top_regressions"]
            .as_array()
            .expect("top regressions");
        assert!(regressions.iter().any(|row| row["function"] == "f"));
    }

    #[test]
    fn present_rows_missing_fields_fail_closed() {
        let good = slice_report(json!([{"name": "hot", "pct": 10.0, "frames": []}]));
        let cases = [
            (
                slice_report(json!([{"name": "hot"}])),
                "Slice field 'pct' must be a number.",
            ),
            (
                slice_report(json!([{"pct": 10.0}])),
                "Slice rows must carry a string 'name'.",
            ),
            (slice_report(json!(["hot"])), "Slice rows must be objects."),
            (
                slice_report(json!([{"name": "hot", "pct": 10.0, "frames": "junk"}])),
                "Slice field 'frames' must be an array.",
            ),
            (
                slice_report(json!([{"name": "hot", "pct": 10.0, "frames": [{"pct": 1.0}]}])),
                "Frame rows must carry a string 'function'.",
            ),
        ];
        for (after, message) in cases {
            let error = compare_slice_json(&good, &after, &CompareOptions::default())
                .expect_err("malformed row must fail");
            assert_eq!(error, message);
        }
    }

    #[test]
    fn reports_require_summary_totals() {
        let good = slice_report(json!([{"name": "hot", "pct": 10.0}]));
        let mut missing_summary = good.clone();
        missing_summary
            .as_object_mut()
            .expect("report object")
            .remove("summary");
        let error = compare_slice_json(&missing_summary, &good, &CompareOptions::default())
            .expect_err("missing summary must fail");
        assert_eq!(error, "Report summary must be an object.");
        let mut empty_summary = good.clone();
        empty_summary["summary"] = json!({});
        let error = compare_slice_json(&empty_summary, &good, &CompareOptions::default())
            .expect_err("missing total must fail");
        assert_eq!(
            error,
            "Report summary field 'total_time_ns' must be an integer."
        );
    }

    #[test]
    fn signed_rows_gate_on_relative_increase() {
        let before = slice_report(json!([{"name": "negative", "pct": -10.0, "frames": []}]));
        let after = slice_report(json!([{"name": "negative", "pct": -5.0, "frames": []}]));
        let compared = compare_slice_json(&before, &after, &CompareOptions::default())
            .expect("signed rows compare");
        assert_eq!(compared["slices"][0]["delta_rel"], json!(50.0));
        assert_eq!(compared["slices"][0]["status"], json!("regression"));
        assert_eq!(compared["has_regression"], json!(true));
        // Zero baseline: unbounded improvement serializes delta_rel as null.
        let empty = slice_report(json!([]));
        let credit = slice_report(json!([{"name": "credit", "pct": -5.0, "frames": []}]));
        let improved = compare_slice_json(&empty, &credit, &CompareOptions::default())
            .expect("new negative rows compare");
        assert_eq!(improved["slices"][0]["delta_rel"], Value::Null);
        assert_eq!(improved["slices"][0]["status"], json!("improvement"));
    }

    #[test]
    fn duplicate_rows_are_validation_errors() {
        let clean = slice_report(json!([{"name": "hot", "pct": 10.0, "frames": []}]));
        let duplicated = slice_report(json!([
            {"name": "hot", "pct": 30.0, "frames": []},
            {"name": "hot", "pct": 10.0, "frames": []},
        ]));
        let error = compare_slice_json(&clean, &duplicated, &CompareOptions::default())
            .expect_err("duplicate slice rows must be rejected");
        assert_eq!(error, "Duplicate Slice row 'hot' in comparison input.");
        let boundary = json!({
            "tool": "clankerprof_boundaries",
            "summary": {"total_time_ns": 100},
            "boundaries": [
                {"name": "web", "pct_of_profile": 10.0},
                {"name": "web", "pct_of_profile": 30.0},
            ],
        });
        let error = compare_boundary_json(&boundary, &boundary, &CompareOptions::default())
            .expect_err("duplicate boundary rows must be rejected");
        assert_eq!(error, "Duplicate Boundary row 'web' in comparison input.");
    }

    #[test]
    fn row_absence_keeps_new_and_removed_semantics() {
        let good = slice_report(json!([{"name": "hot", "pct": 10.0, "frames": []}]));
        let empty = slice_report(json!([]));
        let compared = compare_slice_json(&good, &empty, &CompareOptions::default())
            .expect("row absence stays legal");
        assert_eq!(compared["slices"][0]["before_pct"], json!(10.0));
        assert_eq!(compared["slices"][0]["after_pct"], json!(0.0));
    }
}
