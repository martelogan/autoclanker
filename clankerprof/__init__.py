from __future__ import annotations

from clankerprof.categorize import (
    categorize_frame,
    load_default_ruby_core_classes,
    load_ruby_core_classes,
    ruby_rules,
    runtime_rules_from_file,
)
from clankerprof.compare import (
    CompareOptions,
    compare_boundary_json,
    compare_json,
    compare_slice_json,
)
from clankerprof.facts import (
    SAMPLE_FACTS_SCHEMA_VERSION,
    SAMPLE_FACTS_SCHEMA_VERSION_V1,
    ProfileFactIndex,
    dumps_sample_facts,
    loads_sample_facts,
    read_sample_facts,
    sample_facts_from_jsonable,
    sample_facts_to_jsonable,
    write_sample_facts,
)
from clankerprof.model import (
    Frame,
    Function,
    Location,
    Profile,
    ProfileFacts,
    Sample,
    SampleFact,
    ValueType,
)
from clankerprof.proto import PprofDecodeError, decode_profile_bytes, load_profile
from clankerprof.rules import (
    RUNTIME_RULES_SCHEMA_VERSION,
    RuntimeMatchRule,
    RuntimeRuleSet,
    load_runtime_rules,
    load_runtime_rules_file,
    runtime_rules_from_mapping,
)
from clankerprof.scopes import (
    BoundaryAnalysisOptions,
    BoundaryAnalysisResult,
    analyze_boundaries,
    analyze_boundary_facts,
)
from clankerprof.slices import (
    SliceAnalysisOptions,
    SliceAnalysisResult,
    SliceDefinition,
    analyze_slice_facts,
    analyze_slices,
)
from clankerprof.targets import (
    TargetAnalysisOptions,
    analyze_target_facts,
    analyze_targets,
)

CLANKERPROF_VERSION = "0.1.0"
__version__ = CLANKERPROF_VERSION

__all__ = [
    "CLANKERPROF_VERSION",
    "RUNTIME_RULES_SCHEMA_VERSION",
    "SAMPLE_FACTS_SCHEMA_VERSION",
    "SAMPLE_FACTS_SCHEMA_VERSION_V1",
    "BoundaryAnalysisOptions",
    "BoundaryAnalysisResult",
    "CompareOptions",
    "Frame",
    "Function",
    "Location",
    "PprofDecodeError",
    "Profile",
    "ProfileFactIndex",
    "ProfileFacts",
    "RuntimeMatchRule",
    "RuntimeRuleSet",
    "Sample",
    "SampleFact",
    "SliceAnalysisOptions",
    "SliceAnalysisResult",
    "SliceDefinition",
    "TargetAnalysisOptions",
    "ValueType",
    "__version__",
    "analyze_boundaries",
    "analyze_boundary_facts",
    "analyze_slice_facts",
    "analyze_slices",
    "analyze_target_facts",
    "analyze_targets",
    "categorize_frame",
    "compare_boundary_json",
    "compare_json",
    "compare_slice_json",
    "decode_profile_bytes",
    "dumps_sample_facts",
    "load_default_ruby_core_classes",
    "load_profile",
    "load_ruby_core_classes",
    "load_runtime_rules",
    "load_runtime_rules_file",
    "loads_sample_facts",
    "read_sample_facts",
    "ruby_rules",
    "runtime_rules_from_file",
    "runtime_rules_from_mapping",
    "sample_facts_from_jsonable",
    "sample_facts_to_jsonable",
    "write_sample_facts",
]
