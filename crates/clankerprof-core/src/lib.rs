//! Rust core for clankerprof profile facts.
//!
//! The public compatibility target is the `clankerprof.sample_facts.v1` JSON
//! schema documented in `docs/CLANKERPROF_SPEC.md`.

pub mod compare;
pub mod facts;
pub mod model;
pub mod proto;
pub mod slices;
pub mod targets;

pub use facts::{
    sample_facts_to_json_value, sample_facts_to_pretty_json, SampleFactsSummary,
    SAMPLE_FACTS_SCHEMA_VERSION,
};
pub use model::{Frame, Function, Location, Profile, ProfileFacts, Sample, SampleFact};
pub use proto::{decode_profile_bytes, load_profile, PprofDecodeError};
