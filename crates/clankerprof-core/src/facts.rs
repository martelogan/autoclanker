use crate::model::{
    check_aggregate_bounds, Frame, ProfileFacts, Sample, SampleFact, TimeNs, ValueType,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;

pub const SAMPLE_FACTS_SCHEMA_VERSION: &str = "clankerprof.sample_facts.v2";
pub const SAMPLE_FACTS_SCHEMA_VERSION_V1: &str = "clankerprof.sample_facts.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SampleFactsSummary {
    pub empty_sample_count: usize,
    pub non_empty_sample_count: usize,
    pub sample_count: usize,
    pub total_primary_value: TimeNs,
}

/// Export sample facts in the compact interned v2 layout.
///
/// Interning order is normative (shared with the Python reference): samples
/// in order, stack frames in order; each new frame interns its name, then its
/// filename, then appends its own row.
pub fn sample_facts_to_json_value(facts: &ProfileFacts) -> Value {
    let mut strings: Vec<String> = Vec::new();
    let mut string_indexes: HashMap<String, usize> = HashMap::new();
    let mut frames: Vec<Value> = Vec::new();
    let mut frame_indexes: HashMap<(u64, u64, String, String, i64, bool), usize> = HashMap::new();

    let intern_string =
        |strings: &mut Vec<String>, string_indexes: &mut HashMap<String, usize>, value: &str| {
            if let Some(index) = string_indexes.get(value) {
                return *index;
            }
            let index = strings.len();
            strings.push(value.to_string());
            string_indexes.insert(value.to_string(), index);
            index
        };

    let mut samples_payload: Vec<Value> = Vec::with_capacity(facts.samples.len());
    for fact in &facts.samples {
        let mut stack_indexes: Vec<usize> = Vec::with_capacity(fact.stack.len());
        for frame in &fact.stack {
            let key = (
                frame.location_id,
                frame.function_id,
                frame.name.clone(),
                frame.filename.clone(),
                frame.line,
                frame.location_is_folded,
            );
            let index = match frame_indexes.get(&key) {
                Some(index) => *index,
                None => {
                    let name_index = intern_string(&mut strings, &mut string_indexes, &frame.name);
                    let filename_index =
                        intern_string(&mut strings, &mut string_indexes, &frame.filename);
                    let index = frames.len();
                    frames.push(json!([
                        frame.location_id,
                        frame.function_id,
                        name_index,
                        filename_index,
                        frame.line,
                        frame.location_is_folded,
                    ]));
                    frame_indexes.insert(key, index);
                    index
                }
            };
            stack_indexes.push(index);
        }
        samples_payload.push(json!({
            "location_ids": fact.sample.location_ids,
            "sample_index": fact.sample_index,
            "stack": stack_indexes,
            "values": fact.sample.values,
        }));
    }

    json!({
        "frames": frames,
        "profile": {
            "default_sample_type": facts.default_sample_type,
            "period": facts.period,
            "period_type": facts.period_type.as_ref().map(|value_type| json!({
                "type": value_type.type_name,
                "unit": value_type.unit,
            })),
            "primary_value_index": facts.primary_value_index,
            "value_types": facts
                .value_types
                .iter()
                .map(|value_type| json!({
                    "type": value_type.type_name,
                    "unit": value_type.unit,
                }))
                .collect::<Vec<_>>(),
        },
        "samples": samples_payload,
        "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
        "strings": strings,
        "summary": {
            "empty_sample_count": facts.empty_sample_count,
            "non_empty_sample_count": facts.non_empty_sample_count(),
            "sample_count": facts.samples.len(),
            "total_primary_value": facts.total_primary_value,
        },
        "tool": "clankerprof_facts",
    })
}

pub fn sample_facts_to_pretty_json(facts: &ProfileFacts) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&sample_facts_to_json_value(facts))
}

pub fn sample_facts_to_compact_json(facts: &ProfileFacts) -> Result<String, serde_json::Error> {
    serde_json::to_string(&sample_facts_to_json_value(facts))
}

/// Import a facts payload (v2 or legacy v1), mirroring Python's
/// `sample_facts_from_jsonable` including its strict validation.
pub fn sample_facts_from_json(payload: &Value) -> Result<ProfileFacts, String> {
    let Some(schema_version) = payload.get("schema_version").and_then(Value::as_str) else {
        return Err(unsupported_schema(None));
    };
    match schema_version {
        SAMPLE_FACTS_SCHEMA_VERSION => sample_facts_from_v2(payload),
        SAMPLE_FACTS_SCHEMA_VERSION_V1 => sample_facts_from_v1(payload),
        other => Err(unsupported_schema(Some(other))),
    }
}

fn unsupported_schema(found: Option<&str>) -> String {
    let found = found.map_or("None".to_string(), |value| format!("'{value}'"));
    format!(
        "Unsupported sample facts schema version: {found}; expected '{SAMPLE_FACTS_SCHEMA_VERSION}' or '{SAMPLE_FACTS_SCHEMA_VERSION_V1}'."
    )
}

fn sample_index_field(payload: &Value) -> Result<usize, String> {
    let Some(raw) = payload.get("sample_index") else {
        return Err("Sample facts payload missing required key: 'sample_index'.".to_string());
    };
    raw.as_u64()
        .map(|value| value as usize)
        .ok_or_else(|| "Sample fact sample_index must be a non-negative integer.".to_string())
}

fn required_sample_key<'a>(entry: &'a Value, key: &str) -> Result<&'a Value, String> {
    entry
        .get(key)
        .ok_or_else(|| format!("Sample facts payload missing required key: '{key}'."))
}

/// Signed 64-bit entries (pprof sample values).
fn i64_items(raw: &Value, field_name: &str) -> Result<Vec<i64>, String> {
    match raw {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_i64().ok_or_else(|| {
                    format!("Sample fact {field_name} entries must be signed 64-bit integers.")
                })
            })
            .collect(),
        _ => Err(format!("Sample fact {field_name} must be an array.")),
    }
}

/// v1 compatibility read: missing key means empty (v2 requires presence).
fn i64_list(payload: &Value, key: &str, field_name: &str) -> Result<Vec<i64>, String> {
    match payload.get(key) {
        None => Ok(Vec::new()),
        Some(raw) => i64_items(raw, field_name),
    }
}

/// Unsigned 64-bit entries (pprof location IDs).
fn u64_items(raw: &Value, field_name: &str) -> Result<Vec<u64>, String> {
    match raw {
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_u64().ok_or_else(|| {
                    format!("Sample fact {field_name} entries must be unsigned 64-bit integers.")
                })
            })
            .collect(),
        _ => Err(format!("Sample fact {field_name} must be an array.")),
    }
}

/// v1 compatibility read: missing key means empty (v2 requires presence).
fn u64_list(payload: &Value, key: &str, field_name: &str) -> Result<Vec<u64>, String> {
    match payload.get(key) {
        None => Ok(Vec::new()),
        Some(raw) => u64_items(raw, field_name),
    }
}

fn sample_facts_from_v2(payload: &Value) -> Result<ProfileFacts, String> {
    let profile = payload
        .get("profile")
        .and_then(Value::as_object)
        .ok_or("Sample facts payload must contain a profile object.")?;
    let value_types = profile
        .get("value_types")
        .map(|raw| -> Result<Vec<ValueType>, String> {
            let Value::Array(items) = raw else {
                return Err("Sample facts profile value_types must be an array.".to_string());
            };
            items
                .iter()
                .map(|item| {
                    let Value::Object(entry) = item else {
                        return Err("Sample facts value type must be an object.".to_string());
                    };
                    Ok(ValueType {
                        type_name: entry
                            .get("type")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        unit: entry
                            .get("unit")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                    })
                })
                .collect()
        })
        .transpose()?
        .unwrap_or_default();
    let period_type = match profile.get("period_type") {
        None | Some(Value::Null) => None,
        Some(Value::Object(entry)) => Some(ValueType {
            type_name: entry
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            unit: entry
                .get("unit")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }),
        Some(_) => return Err("Sample facts value type must be an object.".to_string()),
    };
    let primary_value_index = match profile.get("primary_value_index") {
        None => 0i64,
        Some(Value::Number(number)) if number.is_i64() => number.as_i64().unwrap_or(0),
        Some(_) => {
            return Err("Sample facts primary_value_index must be an integer.".to_string());
        }
    };
    if primary_value_index < 0 {
        return Err("Sample facts primary_value_index must be non-negative.".to_string());
    }
    let period = match profile.get("period") {
        None => 0,
        Some(value) => value
            .as_i64()
            .ok_or("Sample facts profile period must be an integer.")?,
    };
    let default_sample_type = profile
        .get("default_sample_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let strings: Vec<String> = match payload.get("strings") {
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(String::from)
                    .ok_or_else(|| "Sample facts strings entries must be strings.".to_string())
            })
            .collect::<Result<_, _>>()?,
        _ => return Err("Sample facts payload must contain a strings array.".to_string()),
    };

    let frames: Vec<Frame> = match payload.get("frames") {
        Some(Value::Array(rows)) => rows
            .iter()
            .map(|row| frame_from_row(row, &strings))
            .collect::<Result<_, _>>()?,
        _ => return Err("Sample facts payload must contain a frames array.".to_string()),
    };

    let Some(Value::Array(raw_samples)) = payload.get("samples") else {
        return Err("Sample facts payload must contain a samples array.".to_string());
    };
    let mut samples = Vec::with_capacity(raw_samples.len());
    for entry in raw_samples {
        if !entry.is_object() {
            return Err("Each sample facts entry must be an object.".to_string());
        }
        let raw_stack = match entry.get("stack") {
            None => {
                return Err("Sample facts payload missing required key: 'stack'.".to_string());
            }
            Some(Value::Array(items)) => items,
            Some(_) => return Err("Sample fact stack must be an array.".to_string()),
        };
        let mut stack = Vec::with_capacity(raw_stack.len());
        for frame_index in raw_stack {
            let index_number = match frame_index {
                Value::Number(number) if number.is_i64() || number.is_u64() => number,
                _ => {
                    return Err("Sample fact stack entries must be frame indexes.".to_string());
                }
            };
            let index = match index_number.as_u64() {
                Some(value) if (value as usize) < frames.len() => value as usize,
                _ => {
                    return Err(format!(
                        "Sample fact frame index {index_number} is out of range."
                    ));
                }
            };
            stack.push(frames[index].clone());
        }
        samples.push(SampleFact {
            sample_index: sample_index_field(entry)?,
            sample: Sample {
                location_ids: u64_items(
                    required_sample_key(entry, "location_ids")?,
                    "location_ids",
                )?,
                values: i64_items(required_sample_key(entry, "values")?, "values")?
                    .into_iter()
                    .map(TimeNs::from)
                    .collect(),
                primary_index: primary_value_index as usize,
            },
            stack,
        });
    }

    check_aggregate_bounds(&samples)?;
    let total_primary_value: TimeNs = samples.iter().map(SampleFact::primary_value).sum();
    let empty_sample_count = samples.iter().filter(|fact| fact.is_empty()).count();
    validate_summary(
        payload.get("summary"),
        samples.len(),
        total_primary_value,
        empty_sample_count,
    )?;
    Ok(ProfileFacts {
        samples,
        total_primary_value,
        empty_sample_count,
        value_types,
        period_type,
        period,
        default_sample_type,
        primary_value_index: primary_value_index as usize,
    })
}

fn frame_from_row(row: &Value, strings: &[String]) -> Result<Frame, String> {
    let Value::Array(cells) = row else {
        return Err(frame_row_error());
    };
    if cells.len() != 6 {
        return Err(frame_row_error());
    }
    let int_cell = |index: usize, label: &str| -> Result<i64, String> {
        cells[index]
            .as_i64()
            .ok_or_else(|| format!("Sample facts frame {label} must be an integer."))
    };
    let u64_cell = |index: usize, label: &str| -> Result<u64, String> {
        cells[index].as_u64().ok_or_else(|| {
            format!("Sample facts frame {label} must be an unsigned 64-bit integer.")
        })
    };
    let location_id = u64_cell(0, "location_id")?;
    let function_id = u64_cell(1, "function_id")?;
    let name_index = int_cell(2, "name index")?;
    let filename_index = int_cell(3, "filename index")?;
    let line = int_cell(4, "line")?;
    let Value::Bool(folded) = cells[5] else {
        return Err("Sample facts frame folded flag must be a boolean.".to_string());
    };
    for position in [name_index, filename_index] {
        if position < 0 || position as usize >= strings.len() {
            return Err(format!(
                "Sample facts frame string index {position} is out of range."
            ));
        }
    }
    Ok(Frame {
        location_id,
        function_id,
        name: strings[name_index as usize].clone(),
        filename: strings[filename_index as usize].clone(),
        line,
        location_is_folded: folded,
    })
}

fn frame_row_error() -> String {
    "Each sample facts frame must be a six-element array of [location_id, function_id, name, filename, line, folded].".to_string()
}

fn validate_summary(
    raw: Option<&Value>,
    sample_count: usize,
    total_primary_value: TimeNs,
    empty_sample_count: usize,
) -> Result<(), String> {
    let Some(Value::Object(summary)) = raw else {
        return Ok(());
    };
    let summary_int = |key: &str| -> Result<Option<i128>, String> {
        match summary.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_i64()
                .map(i128::from)
                .or_else(|| value.as_u64().map(i128::from))
                .map(Some)
                .ok_or_else(|| format!("Sample facts summary {key} must be an integer.")),
        }
    };
    let check = |key: &str, expected: i128, message: &str| -> Result<(), String> {
        match summary_int(key)? {
            Some(found) if found != expected => Err(message.to_string()),
            _ => Ok(()),
        }
    };
    check(
        "sample_count",
        sample_count as i128,
        "Sample facts summary sample count does not match samples.",
    )?;
    check(
        "total_primary_value",
        total_primary_value,
        "Sample facts summary total does not match samples.",
    )?;
    check(
        "empty_sample_count",
        empty_sample_count as i128,
        "Sample facts empty count does not match samples.",
    )?;
    check(
        "non_empty_sample_count",
        (sample_count - empty_sample_count) as i128,
        "Sample facts non-empty count does not match samples.",
    )
}

fn sample_facts_from_v1(payload: &Value) -> Result<ProfileFacts, String> {
    let Some(Value::Array(raw_samples)) = payload.get("samples") else {
        return Err("Sample facts payload must contain a samples array.".to_string());
    };
    let mut samples = Vec::with_capacity(raw_samples.len());
    for entry in raw_samples {
        let Value::Object(_) = entry else {
            return Err("Each sample facts entry must be an object.".to_string());
        };
        let raw_stack = match entry.get("stack") {
            None => &Vec::new(),
            Some(Value::Array(items)) => items,
            Some(_) => return Err("Sample fact stack must be an array.".to_string()),
        };
        let mut stack = Vec::with_capacity(raw_stack.len());
        for item in raw_stack {
            let Value::Object(frame_entry) = item else {
                return Err("Each sample fact frame must be an object.".to_string());
            };
            let required = |key: &str| -> Result<&Value, String> {
                frame_entry
                    .get(key)
                    .ok_or_else(|| format!("Sample facts payload missing required key: '{key}'."))
            };
            let frame_u64 = |key: &str| -> Result<u64, String> {
                required(key)?.as_u64().ok_or_else(|| {
                    format!("Sample facts frame {key} must be an unsigned 64-bit integer.")
                })
            };
            let frame_str = |key: &str| -> Result<String, String> {
                required(key)?
                    .as_str()
                    .map(String::from)
                    .ok_or_else(|| format!("Sample fact frame {key} must be a string."))
            };
            let line = match frame_entry.get("line") {
                None => 0,
                Some(value) => value
                    .as_i64()
                    .ok_or("Sample facts frame line must be an integer.")?,
            };
            let location_is_folded = match frame_entry.get("location_is_folded") {
                None => false,
                Some(value) => value
                    .as_bool()
                    .ok_or("Sample facts frame folded flag must be a boolean.")?,
            };
            stack.push(Frame {
                location_id: frame_u64("location_id")?,
                function_id: frame_u64("function_id")?,
                name: frame_str("name")?,
                filename: frame_str("filename")?,
                line,
                location_is_folded,
            });
        }
        let fact = SampleFact {
            sample_index: sample_index_field(entry)?,
            sample: Sample {
                location_ids: u64_list(entry, "location_ids", "location_ids")?,
                values: i64_list(entry, "values", "values")?
                    .into_iter()
                    .map(TimeNs::from)
                    .collect(),
                primary_index: 0,
            },
            stack,
        };
        match entry.get("primary_value") {
            None | Some(Value::Null) => {}
            Some(raw_primary) => {
                let expected = raw_primary
                    .as_i64()
                    .ok_or("Sample fact primary_value must be an integer.")?;
                if TimeNs::from(expected) != fact.primary_value() {
                    return Err("Sample fact primary value does not match values.".to_string());
                }
            }
        }
        match entry.get("is_empty") {
            None | Some(Value::Null) => {}
            Some(raw_is_empty) => {
                let expected = raw_is_empty
                    .as_bool()
                    .ok_or("Sample fact is_empty must be a boolean.")?;
                if expected != fact.is_empty() {
                    return Err("Sample fact is_empty does not match stack.".to_string());
                }
            }
        }
        samples.push(fact);
    }
    check_aggregate_bounds(&samples)?;
    let total_primary_value: TimeNs = samples.iter().map(SampleFact::primary_value).sum();
    let empty_sample_count = samples.iter().filter(|fact| fact.is_empty()).count();
    validate_summary(
        payload.get("summary"),
        samples.len(),
        total_primary_value,
        empty_sample_count,
    )?;
    Ok(ProfileFacts {
        samples,
        total_primary_value,
        empty_sample_count,
        ..empty_profile_facts()
    })
}

fn empty_profile_facts() -> ProfileFacts {
    ProfileFacts {
        samples: Vec::new(),
        total_primary_value: 0,
        empty_sample_count: 0,
        value_types: Vec::new(),
        period_type: None,
        period: 0,
        default_sample_type: String::new(),
        primary_value_index: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn import_rejects_unknown_schema_version() {
        let error =
            sample_facts_from_json(&json!({"schema_version": "clankerprof.sample_facts.v9"}))
                .unwrap_err();
        assert!(error.contains("Unsupported sample facts schema version"));
    }

    #[test]
    fn import_validates_v2_shapes() {
        let base = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
            "profile": {
                "value_types": [{"type": "cpu", "unit": "nanoseconds"}],
                "period_type": null,
                "period": 0,
                "default_sample_type": "",
                "primary_value_index": 0,
            },
            "strings": ["Leaf#work", "/srv/app/leaf.py"],
            "frames": [[1, 1, 0, 1, 3, false]],
            "samples": [
                {"sample_index": 0, "values": [7], "location_ids": [1], "stack": [0]}
            ],
        });
        let facts = sample_facts_from_json(&base).expect("valid payload");
        assert_eq!(facts.total_primary_value, 7);
        assert_eq!(facts.samples[0].stack[0].name, "Leaf#work");

        let mut bad = base.clone();
        bad["frames"][0][2] = json!(99);
        assert!(sample_facts_from_json(&bad)
            .unwrap_err()
            .contains("string index 99 is out of range"));

        let mut bad = base.clone();
        bad["samples"][0]["stack"] = json!([5]);
        assert!(sample_facts_from_json(&bad)
            .unwrap_err()
            .contains("frame index 5 is out of range"));

        let mut bad = base.clone();
        bad["profile"]["primary_value_index"] = json!(-1);
        assert!(sample_facts_from_json(&bad)
            .unwrap_err()
            .contains("must be non-negative"));
    }

    #[test]
    fn import_accepts_uint64_ids_and_rejects_non_integral_numbers() {
        let big_id: u64 = 1 << 63;
        let base = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
            "profile": {
                "value_types": [{"type": "cpu", "unit": "nanoseconds"}],
                "period_type": null,
                "period": 0,
                "default_sample_type": "",
                "primary_value_index": 0,
            },
            "strings": ["Target#render", "/app/target.rb"],
            "frames": [[big_id, big_id, 0, 1, 1, false]],
            "samples": [
                {"sample_index": 0, "values": [7], "location_ids": [big_id], "stack": [0]}
            ],
        });
        let facts = sample_facts_from_json(&base).expect("uint64 ids are valid");
        assert_eq!(facts.samples[0].stack[0].location_id, big_id);
        assert_eq!(facts.samples[0].sample.location_ids, vec![big_id]);
        assert_eq!(facts.total_primary_value, 7);

        let mut bad = base.clone();
        bad["samples"][0]["values"] = json!([7.9]);
        assert_eq!(
            sample_facts_from_json(&bad).unwrap_err(),
            "Sample fact values entries must be signed 64-bit integers."
        );

        let mut bad = base.clone();
        bad["samples"][0]["location_ids"] = json!([-1]);
        assert_eq!(
            sample_facts_from_json(&bad).unwrap_err(),
            "Sample fact location_ids entries must be unsigned 64-bit integers."
        );

        let mut bad = base.clone();
        bad["frames"][0][0] = json!(-1);
        assert_eq!(
            sample_facts_from_json(&bad).unwrap_err(),
            "Sample facts frame location_id must be an unsigned 64-bit integer."
        );
    }

    #[test]
    fn import_enforces_the_aggregate_bound_exactly() {
        let sample = |index: usize| json!({"sample_index": index, "values": [i64::MAX], "location_ids": [], "stack": []});
        let mut payload = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
            "profile": {
                "value_types": [{"type": "cpu", "unit": "nanoseconds"}],
                "period_type": null,
                "period": 0,
                "default_sample_type": "",
                "primary_value_index": 0,
            },
            "strings": [],
            "frames": [],
            "samples": [sample(0), sample(1)],
        });
        let facts = sample_facts_from_json(&payload).expect("two i64::MAX samples fit");
        assert_eq!(facts.total_primary_value, (u64::MAX - 1) as i128);
        assert_eq!(
            sample_facts_to_compact_json(&facts)
                .expect("exact big total serializes")
                .contains("18446744073709551614"),
            true
        );

        payload["samples"] = json!([sample(0), sample(1), sample(2)]);
        assert_eq!(
            sample_facts_from_json(&payload).unwrap_err(),
            "Aggregate sample values exceed the supported integer range."
        );
    }

    #[test]
    fn import_accepts_v1_and_checks_derived_fields() {
        let payload = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION_V1,
            "samples": [{
                "sample_index": 0,
                "primary_value": 7,
                "values": [7, 9],
                "location_ids": [1],
                "is_empty": false,
                "stack": [{
                    "location_id": 1, "function_id": 1, "name": "Legacy#call",
                    "filename": "/srv/app/legacy.py", "line": 3,
                    "location_is_folded": false,
                }],
            }],
        });
        let facts = sample_facts_from_json(&payload).expect("valid v1");
        assert_eq!(facts.total_primary_value, 7);

        let mut bad = payload.clone();
        bad["samples"][0]["primary_value"] = json!(999);
        assert!(sample_facts_from_json(&bad)
            .unwrap_err()
            .contains("primary value does not match"));
    }

    #[test]
    fn v2_samples_require_values_location_ids_and_stack() {
        let base = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
            "profile": {
                "value_types": [{"type": "cpu", "unit": "nanoseconds"}],
                "period_type": null,
                "period": 0,
                "default_sample_type": "",
                "primary_value_index": 0,
            },
            "strings": ["Leaf#work", "/srv/app/leaf.py"],
            "frames": [[1, 1, 0, 1, 3, false]],
            "samples": [
                {"sample_index": 0, "values": [7], "location_ids": [1], "stack": [0]}
            ],
        });
        for key in ["values", "location_ids", "stack"] {
            let mut bad = base.clone();
            bad["samples"][0]
                .as_object_mut()
                .expect("sample object")
                .remove(key);
            let error = sample_facts_from_json(&bad).unwrap_err();
            assert_eq!(
                error,
                format!("Sample facts payload missing required key: '{key}'.")
            );
        }
        // v1 keeps its documented leniency: missing per-sample arrays read as
        // empty instead of failing import.
        let lenient_v1 = json!({
            "schema_version": SAMPLE_FACTS_SCHEMA_VERSION_V1,
            "samples": [{"sample_index": 0}],
        });
        let facts = sample_facts_from_json(&lenient_v1).expect("lenient v1 sample");
        assert!(facts.samples[0].sample.values.is_empty());
        assert!(facts.samples[0].stack.is_empty());
    }
}
