use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

pub type TimeNs = i64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub function_id: u64,
    pub name: String,
    pub system_name: String,
    pub filename: String,
    pub start_line: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Frame {
    pub location_id: u64,
    pub function_id: u64,
    pub name: String,
    pub filename: String,
    pub line: i64,
    pub location_is_folded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueType {
    pub type_name: String,
    pub unit: String,
}

/// Pick the value index projections aggregate, per pprof convention.
///
/// `default_sample_type` wins when it names a declared sample type; otherwise
/// the last declared type is the default. Profiles that declare no sample
/// types keep index 0.
pub fn select_primary_value_index(sample_types: &[ValueType], default_sample_type: &str) -> usize {
    if !default_sample_type.is_empty() {
        if let Some(index) = sample_types
            .iter()
            .position(|value_type| value_type.type_name == default_sample_type)
        {
            return index;
        }
    }
    sample_types.len().saturating_sub(1)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sample {
    pub location_ids: Vec<u64>,
    pub values: Vec<TimeNs>,
    pub primary_index: usize,
}

impl Sample {
    pub fn primary_value(&self) -> TimeNs {
        if self.values.is_empty() {
            return 0;
        }
        self.values
            .get(self.primary_index)
            .copied()
            .unwrap_or(self.values[0])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleFact {
    pub sample_index: usize,
    pub sample: Sample,
    pub stack: Vec<Frame>,
}

impl SampleFact {
    pub fn primary_value(&self) -> TimeNs {
        self.sample.primary_value()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileFacts {
    pub samples: Vec<SampleFact>,
    pub total_primary_value: TimeNs,
    pub empty_sample_count: usize,
    pub value_types: Vec<ValueType>,
    pub period_type: Option<ValueType>,
    pub period: i64,
    pub default_sample_type: String,
    pub primary_value_index: usize,
}

impl ProfileFacts {
    pub fn non_empty_sample_count(&self) -> usize {
        self.samples.len() - self.empty_sample_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Location {
    pub location_id: u64,
    pub lines: Vec<(u64, i64)>,
    pub is_folded: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub string_table: Vec<String>,
    pub functions: BTreeMap<u64, Function>,
    pub locations: BTreeMap<u64, Location>,
    pub samples: Vec<Sample>,
    pub sample_types: Vec<ValueType>,
    pub period_type: Option<ValueType>,
    pub period: i64,
    pub default_sample_type: String,
    pub primary_value_index: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunctionMetrics {
    pub count: usize,
    pub cpu_time: TimeNs,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CallerMetrics {
    pub count: usize,
    pub cpu_time: TimeNs,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SemanticCallerMetrics {
    pub count: usize,
    pub caller_names: BTreeMap<String, usize>,
    pub caller_files: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CategoryStats {
    pub cpu_time: TimeNs,
    pub sample_count: usize,
    pub functions: BTreeMap<String, FunctionMetrics>,
    pub files: BTreeSet<String>,
    pub folded_from: BTreeMap<String, TimeNs>,
    pub semantic_callers: BTreeMap<String, SemanticCallerMetrics>,
    pub caller_leaf_pairs: BTreeMap<String, CallerMetrics>,
}

impl CategoryStats {
    pub fn add_function(&mut self, name: &str, value: TimeNs) {
        let metrics = self.functions.entry(name.to_string()).or_default();
        metrics.count += 1;
        metrics.cpu_time += value;
    }

    pub fn add_caller_leaf_pair(&mut self, caller: &str, leaf: &str, value: TimeNs) {
        let metrics = self
            .caller_leaf_pairs
            .entry(format!("{caller} -> {leaf}"))
            .or_default();
        metrics.count += 1;
        metrics.cpu_time += value;
    }

    pub fn add_semantic_caller(&mut self, leaf: &str, caller: &Frame) {
        let metrics = self.semantic_callers.entry(leaf.to_string()).or_default();
        metrics.count += 1;
        *metrics.caller_names.entry(caller.name.clone()).or_insert(0) += 1;
        *metrics
            .caller_files
            .entry(caller.filename.clone())
            .or_insert(0) += 1;
    }
}

impl Profile {
    pub fn stack_for_sample(&self, sample: &Sample) -> Vec<Frame> {
        let mut frames = Vec::new();
        for location_id in &sample.location_ids {
            let Some(location) = self.locations.get(location_id) else {
                continue;
            };
            for (function_id, line) in &location.lines {
                let Some(function) = self.functions.get(function_id) else {
                    continue;
                };
                frames.push(Frame {
                    location_id: *location_id,
                    function_id: *function_id,
                    name: function.name.clone(),
                    filename: function.filename.clone(),
                    line: *line,
                    location_is_folded: location.is_folded,
                });
            }
        }
        frames
    }

    pub fn to_sample_facts(&self) -> ProfileFacts {
        let samples: Vec<SampleFact> = self
            .samples
            .iter()
            .enumerate()
            .map(|(sample_index, sample)| SampleFact {
                sample_index,
                sample: sample.clone(),
                stack: self.stack_for_sample(sample),
            })
            .collect();
        let total_primary_value = samples.iter().map(SampleFact::primary_value).sum();
        let empty_sample_count = samples.iter().filter(|sample| sample.is_empty()).count();
        ProfileFacts {
            samples,
            total_primary_value,
            empty_sample_count,
            value_types: self.sample_types.clone(),
            period_type: self.period_type.clone(),
            period: self.period,
            default_sample_type: self.default_sample_type.clone(),
            primary_value_index: self.primary_value_index,
        }
    }
}
