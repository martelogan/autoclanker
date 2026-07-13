use crate::model::{ProfileFacts, TimeNs};
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
