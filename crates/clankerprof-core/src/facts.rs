use crate::model::{Frame, ProfileFacts, SampleFact, TimeNs};
use serde::Serialize;
use serde_json::{json, Value};

pub const SAMPLE_FACTS_SCHEMA_VERSION: &str = "clankerprof.sample_facts.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SampleFactsSummary {
    pub empty_sample_count: usize,
    pub non_empty_sample_count: usize,
    pub sample_count: usize,
    pub total_primary_value: TimeNs,
}

pub fn sample_facts_to_json_value(facts: &ProfileFacts) -> Value {
    json!({
        "schema_version": SAMPLE_FACTS_SCHEMA_VERSION,
        "samples": facts.samples.iter().map(sample_fact_to_json_value).collect::<Vec<_>>(),
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

fn sample_fact_to_json_value(fact: &SampleFact) -> Value {
    json!({
        "is_empty": fact.is_empty(),
        "location_ids": fact.sample.location_ids,
        "primary_value": fact.primary_value(),
        "sample_index": fact.sample_index,
        "stack": fact.stack.iter().map(frame_to_json_value).collect::<Vec<_>>(),
        "values": fact.sample.values,
    })
}

fn frame_to_json_value(frame: &Frame) -> Value {
    json!({
        "filename": frame.filename,
        "function_id": frame.function_id,
        "line": frame.line,
        "location_id": frame.location_id,
        "location_is_folded": frame.location_is_folded,
        "name": frame.name,
    })
}
