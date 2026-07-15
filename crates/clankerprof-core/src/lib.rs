//! Rust core for clankerprof profile facts.
//!
//! The public compatibility target is the `clankerprof.sample_facts.v2` JSON
//! schema documented in `docs/CLANKERPROF_SPEC.md`.

pub mod categorize;
pub mod compare;
pub mod facts;
pub mod model;
pub mod proto;
pub mod pyjson;
pub mod render;
pub mod rules;
pub mod scopes;
pub mod slices;
pub mod targets;

pub use facts::{
    sample_facts_to_compact_json, sample_facts_to_json_value, sample_facts_to_pretty_json,
    SampleFactsSummary, SAMPLE_FACTS_SCHEMA_VERSION, SAMPLE_FACTS_SCHEMA_VERSION_V1,
};
pub use model::{Frame, Function, Location, Profile, ProfileFacts, Sample, SampleFact, ValueType};
pub use proto::{decode_profile_bytes, load_profile, PprofDecodeError};
