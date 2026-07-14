from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any, cast

import pytest

import clankerprof.analysis as clankerprof_analysis
import clankerprof.cli as clankerprof_cli

from autoclanker.cli import main as autoclanker_main
from clankerprof.analysis import (
    DEFAULT_RUNTIME_RULES,
    AttributionRule,
    BoundaryAnalysisOptions,
    BoundaryCategoryDefinition,
    BoundaryDefinition,
    BoundaryDomainDefinition,
    FramePredicateExpr,
    SliceAnalysisOptions,
    SliceDefinition,
    TargetAnalysisOptions,
    analyze_boundaries,
    analyze_boundary_facts,
    analyze_slice_facts,
    analyze_slices,
    analyze_target_facts,
    analyze_targets,
    categorize_ruby_frame,
    extract_gem_name,
    extract_library_name,
    frame_predicate_expr,
    is_runtime_stdlib_path,
    load_default_ruby_core_classes,
    match_category_pattern,
    match_path_pattern,
    parse_frame_predicates,
    ruby_rules,
    runtime_rules_from_file,
)
from clankerprof.cli import main as clankerprof_main
from clankerprof.compare import (
    CompareOptions,
    compare_boundary_json,
    compare_json,
    compare_slice_json,
)
from clankerprof.facts import (
    SAMPLE_FACTS_SCHEMA_VERSION,
    ProfileFactIndex,
    dumps_sample_facts,
    loads_sample_facts,
    sample_facts_to_jsonable,
)
from clankerprof.model import Frame, ValueType
from clankerprof.proto import PprofDecodeError, decode_profile_bytes, load_profile
from clankerprof.render import (
    render_boundary_json,
    render_json_payload,
    render_semantic_callers_csv,
    render_slice_json,
    render_target_csv,
    render_target_json,
    render_target_text,
)
from clankerprof.rules import (
    RuntimeRuleSet,
    load_runtime_rules,
    load_runtime_rules_file,
)
from tests.compliance import covers
from tests.fixtures.pprof_builder import PprofFixtureBuilder


def _target_profile_bytes(*, gzipped: bool = False) -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    template = builder.location(
        builder.function("Template#emit", "/app/templates/view.rb")
    )
    template_native = builder.location(
        builder.function("TemplateEngine::Native#render", "<cfunc>")
    )
    gem = builder.location(
        builder.function("OtherGem#run", "/gems/other-gem/lib/run.rb")
    )
    random = builder.location(builder.function("RandomWork", "/elsewhere/random.rb"))
    builder.sample((template, target), 1_000_000)
    builder.sample((template_native, target), 3_000_000)
    builder.sample((random, gem, target), 4_000_000)
    return builder.encode(gzipped=gzipped)


def _ruby_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    app = builder.location(
        builder.function("App::Presenter#build", "/app/presenter.rb")
    )
    string_gsub = builder.location(builder.function("String#gsub", "<cfunc>"))
    statsd = builder.location(
        builder.function(
            "StatsD::Instrument::Aggregator#increment",
            "/gems/statsd-instrument/lib/statsd/instrument/aggregator.rb",
        )
    )
    kernel_clone = builder.location(
        builder.function("Kernel#clone", "<internal:kernel>")
    )
    marshal_load = builder.location(builder.function("Marshal.load", "<cfunc>"))
    template_native = builder.location(
        builder.function("TemplateEngine::Native#render", "<cfunc>")
    )
    builder.sample((string_gsub, app, target), 1_000_000_000)
    builder.sample((kernel_clone, statsd, target), 2_000_000_000)
    builder.sample((marshal_load, target), 3_000_000_000)
    builder.sample((template_native, target), 4_000_000_000)
    return builder.encode()


def _ruby_core_classes() -> set[str]:
    return {
        "Array",
        "BigDecimal",
        "CGI",
        "Class",
        "Digest",
        "Dir",
        "Enumerable",
        "Enumerator",
        "File",
        "GC",
        "Hash",
        "Integer",
        "IO",
        "JSON",
        "Kernel",
        "Marshal",
        "Module",
        "Monitor",
        "Object",
        "Psych",
        "Ractor",
        "Random",
        "Set",
        "String",
        "StringScanner",
        "Thread",
        "Time",
        "YAML",
    }


def _legacy_flag_matrix_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function(
            "Example::HtmlResponder#render_template",
            "/app/controllers/responders/base_html_responder.rb",
        )
    )

    string_gsub = builder.location(builder.function("String#gsub", "<cfunc>"))
    array_map = builder.location(builder.function("Array#map", "<internal:array>"))
    hash_each = builder.location(builder.function("Hash#each", "<internal:array>"))
    monitor_sync = builder.location(builder.function("Monitor#synchronize", "<cfunc>"))
    time_at = builder.location(builder.function("Time.at", "<internal:timev>"))
    kernel_clone = builder.location(
        builder.function("Kernel#clone", "<internal:kernel>")
    )
    gc_stat = builder.location(builder.function("GC.stat", "<internal:gc>"))
    statsd = builder.location(
        builder.function(
            "StatsD::Instrument::Aggregator#increment",
            "/gems/statsd-instrument/lib/statsd/aggregator.rb",
        )
    )
    opentelemetry = builder.location(
        builder.function(
            "OpenTelemetry::SDK::Trace::Span#attributes",
            "/gems/opentelemetry-sdk/lib/opentelemetry/trace/span.rb",
        )
    )
    message_codec = builder.location(
        builder.function(
            "MessageCodec::Types.time_unpack",
            "/gems/message-codec/lib/message_codec.rb",
        )
    )
    i18n = builder.location(
        builder.function("I18n::Backend::Base#translate", "/gems/i18n/lib/i18n.rb")
    )
    marshal_load = builder.location(builder.function("Marshal.load", "<cfunc>"))
    openssl = builder.location(
        builder.function("OpenSSL::Digest#initialize", "<cfunc>")
    )
    template_native = builder.location(
        builder.function("TemplateEngine::Native#render", "<cfunc>")
    )

    builder.sample((string_gsub, statsd, target), 1_000_000_000)
    builder.sample((kernel_clone, opentelemetry, target), 2_000_000_000)
    builder.sample((marshal_load, target), 1_500_000_000)
    builder.sample((array_map, target), 1_000_000_000)
    builder.sample((monitor_sync, target), 500_000_000)
    builder.sample((time_at, message_codec, target), 800_000_000)
    builder.sample((openssl, target), 600_000_000)
    builder.sample((template_native, target), 1_200_000_000)
    builder.sample((gc_stat, target), 100_000_000)
    builder.sample((hash_each, i18n, target), 700_000_000)
    return builder.encode()


def _main_category_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function(
            "Example::HtmlResponder#render_template",
            "/app/responders/base_html_responder.rb",
        )
    )
    marshal_load = builder.location(builder.function("Marshal.load", "/app/file.rb"))
    json_parse = builder.location(builder.function("JSON.parse", "/app/parser.rb"))
    io_read = builder.location(builder.function("IO.read", "<cfunc>"))
    trilogy = builder.location(builder.function("Trilogy#query", "<cfunc>"))
    statsd = builder.location(builder.function("StatsD.increment", "<cfunc>"))
    otel = builder.location(
        builder.function(
            "OpenTelemetry::Trace#span", "/gems/opentelemetry/lib/trace.rb"
        )
    )
    active_support = builder.location(
        builder.function(
            "ActiveSupport::Cache#fetch", "/gems/activesupport/lib/cache.rb"
        )
    )
    i18n = builder.location(
        builder.function("I18n.translate", "/gems/i18n/lib/i18n.rb")
    )

    builder.sample((marshal_load, target), 2_000_000_000)
    builder.sample((json_parse, target), 1_000_000_000)
    builder.sample((io_read, target), 1_500_000_000)
    builder.sample((trilogy, target), 1_000_000_000)
    builder.sample((statsd, target), 800_000_000)
    builder.sample((otel, target), 700_000_000)
    builder.sample((active_support, target), 1_200_000_000)
    builder.sample((i18n, target), 800_000_000)
    return builder.encode()


def _request_rendering_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    view_model = builder.location(
        builder.function("ReportView#build", "/app/view_models/report_view.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/product_card.rb")
    )
    cache = builder.location(
        builder.function(
            "CacheClient#get_multi",
            "/vendor/cache-client-1.2.3/lib/client.rb",
        )
    )
    template_engine = builder.location(
        builder.function("TemplateEngine::Native#render", "<cfunc>")
    )
    router = builder.location(
        builder.function("Router#dispatch", "/app/http/router.rb")
    )
    background_job = builder.location(
        builder.function("BackgroundJob#perform", "/app/jobs/background_job.rb")
    )
    builder.sample((view_model, request), 10_000_000)
    builder.sample((component, request), 20_000_000)
    builder.sample((cache, component, request), 30_000_000)
    builder.sample((template_engine, component, request), 40_000_000)
    builder.sample((router,), 50_000_000)
    builder.sample((background_job,), 60_000_000)
    return builder.encode()


def _boundary_decomposition_profile_bytes(sample_repetitions: int = 1) -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    middleware = builder.location(
        builder.function("MiddlewareStack#call", "/app/http/middleware.rb")
    )
    template = builder.location(
        builder.function("TemplateRenderer#render", "/app/rendering/template.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/card.rb")
    )
    view_model = builder.location(
        builder.function("ReportView#build", "/app/view_models/report_view.rb")
    )
    cache = builder.location(
        builder.function(
            "CacheClient#get_multi",
            "/srv/vendor/cache-client-1.2.3/lib/client.rb",
        )
    )
    json_generate = builder.location(
        builder.function("JSON.generate", "/runtime/json/encoder.rb")
    )
    worker = builder.location(
        builder.function("BackgroundJob#perform", "/app/jobs/background_job.rb")
    )
    for _ in range(sample_repetitions):
        builder.sample((view_model, template, request, middleware), 10_000_000)
        builder.sample((component, template, request, middleware), 20_000_000)
        builder.sample((cache, component, template, request, middleware), 30_000_000)
        builder.sample(
            (json_generate, view_model, template, request, middleware),
            40_000_000,
        )
        builder.sample((worker,), 50_000_000)
    return builder.encode()


def _descendant_attribute_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/card.rb")
    )
    wrapper = builder.location(
        builder.function("TelemetryWrapper#call", "/app/lib/telemetry.rb")
    )
    cache = builder.location(
        builder.function("CacheClient#get", "/vendor/cache-client-1.2.3/lib/client.rb")
    )
    builder.sample((cache, wrapper, component, request), 70_000_000)
    return builder.encode()


def _slice_semantics_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/card.rb")
    )
    native = builder.location(builder.function("NativeRenderer#render", "<cfunc>"))
    gem = builder.location(
        builder.function("Instrumentation#wrap", "/gems/statsd-instrument/lib/wrap.rb")
    )
    gc_mark = builder.location(builder.function("(marking)", "<internal:gc>"))

    builder.sample((native, component, request), 10_000_000)
    builder.sample((gem,), 20_000_000)
    builder.sample((gc_mark, request), 30_000_000)
    return builder.encode()


def _stdlib_caller_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    app = builder.location(builder.function("App::Presenter#call", "/app/presenter.rb"))
    forwardable = builder.location(
        builder.function(
            "Forwardable#_delegator_method", "/usr/local/lib/ruby/3.2.0/forwardable.rb"
        )
    )
    string_gsub = builder.location(builder.function("String#gsub", "<cfunc>"))
    builder.sample((string_gsub, forwardable, app, target), 12_000_000)
    return builder.encode()


def _inline_target_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    leaf = builder.function("InlineLeaf#work", "/app/inline_leaf.rb")
    target = builder.function("Target#render", "/app/responders/target.rb")
    inline_location = builder.inline_location((leaf, target))
    builder.sample((inline_location,), 9_000_000)
    return builder.encode()


def _folded_location_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    folded_leaf = builder.folded_location(
        builder.function("FoldedLeaf#work", "/app/folded_leaf.rb")
    )
    builder.sample((folded_leaf, target), 6_000_000)
    return builder.encode()


def _non_contiguous_function_id_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.function("Target#render", "/app/responders/target.rb")
    leaf = builder.function("Leaf#work", "/app/leaf.rb")
    target_entry = builder.functions[target - 1]
    leaf_entry = builder.functions[leaf - 1]
    builder.functions[target - 1] = (100, target_entry[1], target_entry[2])
    builder.functions[leaf - 1] = (250, leaf_entry[1], leaf_entry[2])
    leaf_location = builder.location(250)
    target_location = builder.location(100)
    builder.sample((leaf_location, target_location), 13_000_000)
    return builder.encode()


def _legacy_no_enhanced_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    app_caller = builder.location(
        builder.function("App::Renderer#render", "/app/renderers/app_renderer.rb")
    )
    native_leaf = builder.location(builder.function("NativeExtension#work", "<cfunc>"))
    builder.sample((native_leaf, app_caller, target), 5_000_000)
    return builder.encode()


def _rule_driven_native_path_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/card.rb")
    )
    native_runtime = builder.location(
        builder.function(
            "RuntimeCore#dispatch",
            "/opt/runtime/ruby/4.0.0/core/runtime_core.rb",
        )
    )
    builder.sample((native_runtime, component, request), 15_000_000)
    return builder.encode()


def _uncollapsible_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    root = builder.location(builder.function("Root#call", "/app/root.rb"))
    collapsed_mid = builder.location(builder.function("Collapsed#wrap", "/app/mid.rb"))
    collapsed_leaf = builder.location(
        builder.function("Collapsed#leaf", "/app/leaf.rb")
    )
    builder.sample((collapsed_leaf, collapsed_mid, root), 40_000_000)
    return builder.encode()


@covers("M9-001")
def test_clankerprof_decodes_raw_and_gzipped_pprof_profiles() -> None:
    raw = decode_profile_bytes(_target_profile_bytes())
    gzipped = decode_profile_bytes(_target_profile_bytes(gzipped=True))

    assert len(raw.samples) == 3
    assert len(gzipped.samples) == 3
    assert raw.stack_for_sample(raw.samples[0])[0].name == "Template#emit"
    assert gzipped.stack_for_sample(gzipped.samples[1])[0].filename == "<cfunc>"


@covers("M9-002")
def test_clankerprof_expands_inline_location_frames_for_target_traversal() -> None:
    profile = decode_profile_bytes(_inline_target_profile_bytes())
    stack = profile.stack_for_sample(profile.samples[0])
    assert [frame.name for frame in stack] == ["InlineLeaf#work", "Target#render"]

    results = analyze_targets(
        profile,
        {"Target#render": {"Application": r"[/\\]app[/\\]"}},
    )["Target#render"]

    assert results["Application"].cpu_time == 9_000_000
    assert results["Application"].functions["InlineLeaf#work"].cpu_time == 9_000_000


@covers("M9-001")
def test_clankerprof_supports_non_contiguous_pprof_function_ids() -> None:
    profile = decode_profile_bytes(_non_contiguous_function_id_profile_bytes())
    assert sorted(profile.functions) == [100, 250]

    stack = profile.stack_for_sample(profile.samples[0])
    assert [frame.function_id for frame in stack] == [250, 100]
    assert [frame.name for frame in stack] == ["Leaf#work", "Target#render"]

    results = analyze_targets(
        profile,
        {"Target#render": {"Application": r"[/\\]app[/\\]"}},
    )["Target#render"]

    assert results["Application"].cpu_time == 13_000_000
    assert results["Application"].functions["Leaf#work"].cpu_time == 13_000_000


@covers("M9-001", "M9-002")
def test_clankerprof_sample_facts_are_the_shared_projection_surface() -> None:
    profile = decode_profile_bytes(_inline_target_profile_bytes())
    profile_facts = profile.to_sample_facts()
    facts = profile.sample_facts()

    assert profile_facts.samples == facts
    assert profile_facts.total_primary_value == 9_000_000
    assert profile_facts.empty_sample_count == 0
    assert profile_facts.non_empty_sample_count == 1
    assert profile_facts.non_empty_samples() == facts
    assert profile.total_primary_value() == 9_000_000
    assert len(facts) == 1
    assert facts[0].sample_index == 0
    assert facts[0].primary_value == 9_000_000
    assert facts[0].leaf is not None
    assert facts[0].leaf.name == "InlineLeaf#work"
    assert [frame.name for frame in facts[0].stack] == [
        "InlineLeaf#work",
        "Target#render",
    ]
    assert [frame.location_id for frame in facts[0].stack] == [1, 1]


@covers("M9-001", "M9-006")
def test_clankerprof_sample_facts_export_round_trips_projection_inputs() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    config = {
        "RequestHandler#render_response": {
            "View Model": "app/view_models/**",
            "Components": "path:app/components/**",
            "Cache Client": "library:cache-client",
        }
    }

    exported = sample_facts_to_jsonable(profile.to_sample_facts())
    assert exported["schema_version"] == SAMPLE_FACTS_SCHEMA_VERSION
    assert exported["summary"] == {
        "empty_sample_count": 0,
        "non_empty_sample_count": 6,
        "sample_count": 6,
        "total_primary_value": 210_000_000,
    }
    imported = loads_sample_facts(json.dumps(exported))

    assert imported == profile.to_sample_facts()
    assert render_target_json(
        analyze_target_facts(imported, config)
    ) == render_target_json(analyze_targets(profile, config))
    options = SliceAnalysisOptions(
        slices=(
            SliceDefinition("components", ("app/components/**",)),
            SliceDefinition("default", is_default=True),
        ),
        collapse=("library:*",),
        unattributed_libraries=1,
    )
    imported_slice_json = render_slice_json(
        analyze_slice_facts(imported, options), options
    )
    profile_slice_json = render_slice_json(analyze_slices(profile, options), options)
    assert imported_slice_json == profile_slice_json


@covers("M9-001", "M9-006")
def test_clankerprof_sample_facts_import_rejects_malformed_frames() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    exported = sample_facts_to_jsonable(profile.to_sample_facts())
    samples = cast(list[dict[str, object]], exported["samples"])
    samples[0]["stack"] = ["not-a-frame"]

    with pytest.raises(ValueError, match="stack entries must be frame indexes"):
        loads_sample_facts(json.dumps(exported))

    exported = sample_facts_to_jsonable(profile.to_sample_facts())
    samples = cast(list[dict[str, object]], exported["samples"])
    samples[0]["stack"] = [10_000]
    with pytest.raises(ValueError, match="frame index 10000 is out of range"):
        loads_sample_facts(json.dumps(exported))

    exported = sample_facts_to_jsonable(profile.to_sample_facts())
    frames = cast(list[list[object]], exported["frames"])
    frames[0] = frames[0][:5]
    with pytest.raises(ValueError, match="six-element array"):
        loads_sample_facts(json.dumps(exported))

    exported = sample_facts_to_jsonable(profile.to_sample_facts())
    summary = cast(dict[str, object], exported["summary"])
    summary["total_primary_value"] = 1
    with pytest.raises(ValueError, match="summary total does not match"):
        loads_sample_facts(json.dumps(exported))


def _multi_value_profile_bytes(
    *,
    default_sample_type: str | None = None,
    packed_samples: bool = False,
) -> bytes:
    builder = PprofFixtureBuilder.create(
        sample_types=(("samples", "count"), ("cpu", "nanoseconds")),
        default_sample_type=default_sample_type,
        period_type=("cpu", "nanoseconds"),
        period=10_000_000,
    )
    handler = builder.location(
        builder.function("RequestHandler#call", "/srv/app/handler.py")
    )
    worker = builder.location(builder.function("Worker#perform", "/srv/app/worker.py"))
    builder.sample((worker, handler), (3, 30_000_000))
    builder.sample((handler,), (2, 20_000_000))
    return builder.encode(packed_samples=packed_samples)


def _recursive_target_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    helper = builder.location(builder.function("Helper#step", "/srv/app/helper.py"))
    leaf = builder.location(builder.function("Leaf#work", "/srv/app/leaf.py"))
    builder.sample((leaf, target, helper, target), 8_000_000)
    builder.sample((leaf, target), 2_000_000)
    return builder.encode()


def test_clankerprof_targets_recursive_frames_count_once_per_sample() -> None:
    profile = decode_profile_bytes(_recursive_target_profile_bytes())
    config = {"Target#render": {"App": "path:srv/app"}}

    categories = analyze_targets(profile, config)["Target#render"]
    total = sum(stats.cpu_time for stats in categories.values())
    assert total == 10_000_000
    assert sum(stats.sample_count for stats in categories.values()) == 2

    rendered = render_target_json(analyze_targets(profile, config))
    parents = cast(dict[str, Any], rendered["parents"])
    assert parents["Target#render"]["total_time_ns"] == 10_000_000


def test_clankerprof_target_category_precedence_and_tie_order() -> None:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    zebra = builder.location(builder.function("ZebraWork#run", "/srv/app/zone/z.py"))
    alpha = builder.location(builder.function("AlphaWork#run", "/srv/app/azone/a.py"))
    shared = builder.location(builder.function("BothWork#run", "/srv/app/shared/b.py"))
    builder.sample((zebra, target), 5_000_000)
    builder.sample((alpha, target), 5_000_000)
    builder.sample((shared, target), 2_000_000)

    config = {
        "Target#render": {
            "Zebra": "path:srv/app/zone",
            "Alpha": "path:srv/app/azone",
            "Shared Z": "path:srv/app/shared",
            "Shared A": "path:srv/app/shared",
        }
    }
    profile = decode_profile_bytes(builder.encode())
    rendered = render_target_json(analyze_targets(profile, config))
    parent = cast(
        dict[str, Any],
        cast(dict[str, Any], rendered["parents"])["Target#render"],
    )
    names = [
        cast(str, category["name"])
        for category in cast(list[dict[str, Any]], parent["categories"])
    ]
    assert names == ["Zebra", "Alpha", "Shared Z"]


def test_clankerprof_primary_value_defaults_to_last_value_type() -> None:
    profile = decode_profile_bytes(_multi_value_profile_bytes())
    assert profile.sample_types == (
        ValueType(type_name="samples", unit="count"),
        ValueType(type_name="cpu", unit="nanoseconds"),
    )
    assert profile.period_type == ValueType(type_name="cpu", unit="nanoseconds")
    assert profile.period == 10_000_000
    assert profile.primary_value_index == 1

    facts = profile.to_sample_facts()
    assert facts.total_primary_value == 50_000_000
    assert [fact.primary_value for fact in facts.samples] == [30_000_000, 20_000_000]


def test_clankerprof_primary_value_honors_default_sample_type() -> None:
    profile = decode_profile_bytes(
        _multi_value_profile_bytes(default_sample_type="samples")
    )
    assert profile.default_sample_type == "samples"
    assert profile.primary_value_index == 0
    assert profile.to_sample_facts().total_primary_value == 5


def test_clankerprof_primary_value_unknown_default_falls_back_to_last() -> None:
    profile = decode_profile_bytes(
        _multi_value_profile_bytes(default_sample_type="walltime")
    )
    assert profile.primary_value_index == 1
    assert profile.to_sample_facts().total_primary_value == 50_000_000


def test_clankerprof_decodes_packed_sample_encoding_identically() -> None:
    unpacked = decode_profile_bytes(_multi_value_profile_bytes())
    packed = decode_profile_bytes(_multi_value_profile_bytes(packed_samples=True))
    assert packed == unpacked


def test_clankerprof_decodes_signed_int64_fields_as_twos_complement() -> None:
    builder = PprofFixtureBuilder.create()
    negative_line = builder.location(
        builder.function("Native#call", "<cfunc>"),
        line=-1,
    )
    builder.sample((negative_line,), -5)

    profile = decode_profile_bytes(builder.encode())
    frame = profile.stack_for_sample(profile.samples[0])[0]
    assert frame.line == -1
    assert profile.samples[0].values == (-5,)
    assert profile.samples[0].primary_value == -5


def test_clankerprof_facts_replay_matches_direct_decode_for_multi_value() -> None:
    profile = decode_profile_bytes(_multi_value_profile_bytes())
    facts = profile.to_sample_facts()

    imported = loads_sample_facts(dumps_sample_facts(facts))
    assert imported == facts
    assert imported.primary_value_index == 1
    assert imported.value_types == facts.value_types
    assert imported.period == 10_000_000

    config = {"RequestHandler#call": {"Workers": "path:srv/app"}}
    assert render_target_json(
        analyze_target_facts(imported, config)
    ) == render_target_json(analyze_target_facts(facts, config))


def test_clankerprof_facts_import_accepts_v1_payloads() -> None:
    payload = {
        "schema_version": "clankerprof.sample_facts.v1",
        "tool": "clankerprof_facts",
        "summary": {
            "sample_count": 1,
            "empty_sample_count": 0,
            "non_empty_sample_count": 1,
            "total_primary_value": 7,
        },
        "samples": [
            {
                "sample_index": 0,
                "primary_value": 7,
                "values": [7, 9],
                "location_ids": [1],
                "is_empty": False,
                "stack": [
                    {
                        "location_id": 1,
                        "function_id": 1,
                        "name": "Legacy#call",
                        "filename": "/srv/app/legacy.py",
                        "line": 3,
                        "location_is_folded": False,
                    }
                ],
            }
        ],
    }

    imported = loads_sample_facts(json.dumps(payload))
    assert imported.total_primary_value == 7
    assert imported.samples[0].stack[0].name == "Legacy#call"
    assert imported.primary_value_index == 0


def test_clankerprof_facts_import_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="Unsupported sample facts schema version"):
        loads_sample_facts(
            json.dumps({"schema_version": "clankerprof.sample_facts.v9"})
        )


def test_clankerprof_facts_export_is_compact_with_pretty_opt_in() -> None:
    facts = decode_profile_bytes(_multi_value_profile_bytes()).to_sample_facts()

    compact = dumps_sample_facts(facts)
    assert "\n" not in compact
    assert '"schema_version":"clankerprof.sample_facts.v2"' in compact

    pretty = dumps_sample_facts(facts, pretty=True)
    assert pretty.startswith("{\n  ")
    assert json.loads(pretty) == json.loads(compact)


def _compare_slice_report(slices: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "tool": "clankerprof_slices",
        "summary": {
            "matching_time_ns": 100,
            "total_time_ns": 100,
            "matching_pct": 100.0,
        },
        "slices": slices,
    }


def _compare_boundary_report(boundaries: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "tool": "clankerprof_boundaries",
        "summary": {"total_time_ns": 1_000},
        "boundaries": boundaries,
    }


def test_clankerprof_compare_new_rows_emit_finite_json() -> None:
    before = _compare_slice_report(
        [{"name": "existing", "pct": 50.0, "frames": []}]
    )
    after = _compare_slice_report(
        [
            {"name": "existing", "pct": 40.0, "frames": []},
            {"name": "brand-new", "pct": 10.0, "frames": []},
        ]
    )

    compared = compare_slice_json(before, after)
    by_name = {
        cast(str, row["name"]): row
        for row in cast(list[dict[str, Any]], compared["slices"])
    }
    assert by_name["brand-new"]["delta_rel"] is None
    assert by_name["existing"]["delta_rel"] == -20.0

    rendered = render_json_payload(compared)
    assert "Infinity" not in rendered

    boundary_compared = compare_boundary_json(
        _compare_boundary_report([]),
        _compare_boundary_report([{"name": "fresh", "pct_of_profile": 12.0}]),
    )
    fresh_row = cast(list[dict[str, Any]], boundary_compared["rows"])[0]
    assert fresh_row["delta_rel"] is None
    assert "Infinity" not in render_json_payload(boundary_compared)


def test_clankerprof_boundary_compare_orders_top_improvements_by_magnitude() -> None:
    before = _compare_boundary_report(
        [
            {"name": "alpha", "pct_of_profile": 40.0},
            {"name": "beta", "pct_of_profile": 30.0},
            {"name": "gamma", "pct_of_profile": 20.0},
        ]
    )
    after = _compare_boundary_report(
        [
            {"name": "alpha", "pct_of_profile": 9.0},
            {"name": "beta", "pct_of_profile": 25.0},
            {"name": "gamma", "pct_of_profile": 19.5},
        ]
    )

    compared = compare_boundary_json(before, after)
    improvements = cast(list[dict[str, Any]], compared["top_improvements"])
    assert [row["name"] for row in improvements] == ["alpha", "beta", "gamma"]
    assert [row["delta_abs"] for row in improvements] == [-31.0, -5.0, -0.5]


def test_clankerprof_compare_rejects_wrong_payload_types() -> None:
    facts_like: dict[str, Any] = {"tool": "clankerprof_facts"}
    with pytest.raises(
        ValueError,
        match="must be clankerprof_slices or clankerprof_boundaries",
    ):
        compare_json(facts_like, dict(facts_like))

    with pytest.raises(ValueError, match="same clankerprof projection"):
        compare_json(
            {"tool": "clankerprof_slices"},
            {"tool": "clankerprof_boundaries"},
        )


def _runtime_args(**overrides: object) -> argparse.Namespace:
    args: dict[str, object] = {
        "runtime": "generic",
        "no_enhanced": False,
        "runtime_rules": None,
        "ruby_core_classes": None,
        "verbose_runtime_internals": False,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_clankerprof_no_enhanced_generic_runtime_keeps_generic_rules() -> None:
    generic_no_enhanced = clankerprof_cli._runtime_rules(  # pyright: ignore[reportPrivateUsage]
        _runtime_args(no_enhanced=True)
    )
    assert generic_no_enhanced.name == "generic"
    assert generic_no_enhanced is DEFAULT_RUNTIME_RULES

    ruby_no_enhanced = clankerprof_cli._runtime_rules(  # pyright: ignore[reportPrivateUsage]
        _runtime_args(runtime="ruby", no_enhanced=True)
    )
    assert ruby_no_enhanced.name == "ruby"

    with pytest.raises(ValueError, match="Unsupported runtime"):
        clankerprof_cli._runtime_rules(  # pyright: ignore[reportPrivateUsage]
            _runtime_args(runtime="python")
        )


def test_clankerprof_decoder_rejects_truncated_and_overlong_fields() -> None:
    overlong_varint = bytes([0x48]) + b"\xff" * 10 + b"\x01"
    with pytest.raises(PprofDecodeError, match="Invalid protobuf varint"):
        decode_profile_bytes(overlong_varint)

    truncated_fixed64 = bytes([0x79, 0x01, 0x02])
    with pytest.raises(PprofDecodeError, match="Skip extends beyond stream"):
        decode_profile_bytes(truncated_fixed64)

    truncated_fixed32 = bytes([0x7D, 0x01])
    with pytest.raises(PprofDecodeError, match="Skip extends beyond stream"):
        decode_profile_bytes(truncated_fixed32)

    ten_byte_varint = bytes([0x48]) + b"\xff" * 9 + b"\x01"
    decode_profile_bytes(ten_byte_varint)


def test_clankerprof_rule_packs_reject_unknown_keys_and_versions(
    tmp_path: Path,
) -> None:
    bad_key = tmp_path / "bad-key.yml"
    bad_key.write_text("name: x\nsurprise_key: 1\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="Unknown runtime rule pack keys: surprise_key"
    ):
        load_runtime_rules_file(bad_key)

    bad_version = tmp_path / "bad-version.yml"
    bad_version.write_text(
        "schema_version: clankerprof.runtime_rules.v9\nname: x\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported runtime rules schema version"):
        load_runtime_rules_file(bad_version)

    bad_rule = tmp_path / "bad-rule.yml"
    bad_rule.write_text(
        "name: x\nsemantic_rules:\n  - category: A\n    name_glob: '*'\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="Unknown semantic_rules entry keys: name_glob"
    ):
        load_runtime_rules_file(bad_rule)

    versioned = tmp_path / "ok.yml"
    versioned.write_text(
        "schema_version: clankerprof.runtime_rules.v1\nname: ok\n",
        encoding="utf-8",
    )
    assert load_runtime_rules_file(versioned).name == "ok"


def _error_envelope(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    captured = capsys.readouterr()
    assert captured.out == ""
    envelope = json.loads(captured.err)
    assert envelope["ok"] is False
    return cast(dict[str, Any], envelope)


def test_clankerprof_error_envelope_for_truncated_gzip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import gzip as gzip_module

    profile_path = tmp_path / "profile.pb.gz"
    profile_path.write_bytes(gzip_module.compress(_target_profile_bytes())[:-5])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"Target#render": {}}), encoding="utf-8")

    exit_code = clankerprof_main(
        ["targets", "--profile", str(profile_path), "--config", str(config_path)]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "Truncated or corrupt gzip" in cast(str, envelope["error"])


def test_clankerprof_error_envelope_for_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"Target#render": {}}), encoding="utf-8")

    exit_code = clankerprof_main(
        [
            "targets",
            "--profile",
            str(tmp_path / "does-not-exist.pb"),
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "does-not-exist.pb" in cast(str, envelope["error"])


def test_clankerprof_error_envelope_for_malformed_yaml(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_slice_semantics_profile_bytes())
    slices_path = tmp_path / "slices.yml"
    slices_path.write_text("slices: [unclosed\n  - broken", encoding="utf-8")

    exit_code = clankerprof_main(
        ["slices", "--profile", str(profile_path), "--slices", str(slices_path)]
    )
    assert exit_code == 2
    _error_envelope(capsys)


def test_clankerprof_error_envelope_for_facts_missing_keys(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(
        json.dumps(
            {
                "schema_version": "clankerprof.sample_facts.v1",
                "samples": [{"values": [1], "location_ids": [1], "stack": []}],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"Target#render": {}}), encoding="utf-8")

    exit_code = clankerprof_main(
        ["targets", "--facts", str(facts_path), "--config", str(config_path)]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "missing required key" in cast(str, envelope["error"])


def test_clankerprof_csv_format_stdout_is_not_mixed_with_envelope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "config.json"
    profile_path.write_bytes(_target_profile_bytes())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": "path:app/**"}}),
        encoding="utf-8",
    )

    exit_code = clankerprof_main(
        [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--format",
            "csv",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("Parent Function,")
    assert "{" not in captured.out

    exit_code = autoclanker_main(
        [
            "pprof",
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--format",
            "text",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "{" not in captured.out
    assert captured.out.strip()


def test_clankerprof_standalone_global_output_is_honored(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "targets.json"
    profile_path.write_bytes(_target_profile_bytes())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": "path:app/**"}}),
        encoding="utf-8",
    )

    exit_code = clankerprof_main(
        [
            "--output",
            str(output_path),
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tool"] == "clankerprof_targets"


def test_clankerprof_filter_validation_applies_without_slices_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_slice_semantics_profile_bytes())

    exit_code = clankerprof_main(
        ["slices", "--profile", str(profile_path), "--filter", "name:"]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "Filter must be" in cast(str, envelope["error"])

    exit_code = clankerprof_main(
        ["slices", "--profile", str(profile_path), "--filter", "bogus:value"]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "Unsupported filter key" in cast(str, envelope["error"])


def test_clankerprof_csv_layout_compat_rejected_with_json_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "config.json"
    profile_path.write_bytes(_target_profile_bytes())
    config_path.write_text(json.dumps({"Target#render": {}}), encoding="utf-8")

    exit_code = clankerprof_main(
        [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--target-csv-layout",
            "compat",
            "--format",
            "json",
        ]
    )
    assert exit_code == 2
    envelope = _error_envelope(capsys)
    assert "requires --format csv" in cast(str, envelope["error"])


def test_clankerprof_slice_filter_negation_respects_descendant_attribution() -> None:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/app/http/request.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/card.rb")
    )
    wrapper = builder.location(
        builder.function("TelemetryWrapper#call", "/app/lib/telemetry.rb")
    )
    cache = builder.location(
        builder.function("CacheClient#get", "/vendor/cache-client-1.2.3/lib/client.rb")
    )
    builder.sample((cache, wrapper, component, request), 70_000_000)
    builder.sample((cache, component, request), 30_000_000)
    profile = decode_profile_bytes(builder.encode())

    options = SliceAnalysisOptions(
        filters=("!slice:instrumentation",),
        slices=(
            SliceDefinition("components", ("app/components/**",)),
            SliceDefinition("instrumentation", ()),
        ),
        attributes=(
            AttributionRule(
                "name",
                "TelemetryWrapper#call",
                "instrumentation",
                descendant=True,
            ),
        ),
    )
    result = analyze_slices(profile, options)
    assert result.matching_time_ns == 30_000_000

    positive = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=("slice:instrumentation",),
            slices=options.slices,
            attributes=options.attributes,
        ),
    )
    assert positive.matching_time_ns == 70_000_000


def test_clankerprof_duplicate_default_slices_rejected() -> None:
    profile = decode_profile_bytes(_slice_semantics_profile_bytes())
    options = SliceAnalysisOptions(
        slices=(
            SliceDefinition("first", is_default=True),
            SliceDefinition("second", is_default=True),
        ),
    )
    with pytest.raises(ValueError, match="multiple default slices"):
        analyze_slices(profile, options)


def test_clankerprof_native_predicate_value_honored(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(ValueError, match="native: predicate value must be"):
        parse_frame_predicates(("native:maybe",))

    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    native_leaf = builder.location(builder.function("NativeExtension#work", "<cfunc>"))
    app_leaf = builder.location(builder.function("Presenter#call", "/app/presenter.rb"))
    builder.sample((native_leaf, target), 4_000_000)
    builder.sample((app_leaf, target), 6_000_000)

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "boundaries.toml"
    profile_path.write_bytes(builder.encode())
    config_path.write_text(
        """
[category]
"Native work" = "native:true"
"Pure app" = "native:false"

[[boundary]]
label = "Request render"
match = "name_eq:Target#render"

[boundary.bucket]
"Native" = ["Native work"]
"App" = ["Pure app"]
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "boundaries",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    boundary = cast(list[dict[str, object]], payload["boundaries"])[0]
    buckets = {
        item["name"]: item
        for item in cast(list[dict[str, object]], boundary["buckets"])
    }
    assert buckets["Native"]["time_ns"] == 4_000_000
    assert buckets["App"]["time_ns"] == 6_000_000


def _fold_window_profile_bytes(*, inline_padding: bool) -> bytes:
    builder = PprofFixtureBuilder.create()
    target_fn = builder.function("Target#render", "/app/responders/target.rb")
    gem_fn = builder.function(
        "StatsD::Batcher#flush",
        "/gems/statsd-instrument-3.5.0/lib/batcher.rb",
    )
    leaf_fn = builder.function("Array#map", "<cfunc>")
    if inline_padding:
        helper_a = builder.function("Helper#a", "/app/lib/helper_a.rb")
        helper_b = builder.function("Helper#b", "/app/lib/helper_b.rb")
        leaf_location = builder.inline_location((leaf_fn, helper_a, helper_b))
    else:
        leaf_location = builder.location(leaf_fn)
    gem_location = builder.location(gem_fn)
    target_location = builder.location(target_fn)
    builder.sample((leaf_location, gem_location, target_location), 9_000_000)
    return builder.encode()


def test_clankerprof_fold_heuristic_ignores_leaf_inline_expansion() -> None:
    config = {"Target#render": {"Application": r"[/\\]app[/\\]"}}
    for inline_padding in (False, True):
        profile = decode_profile_bytes(
            _fold_window_profile_bytes(inline_padding=inline_padding)
        )
        categories = analyze_targets(
            profile,
            config,
            TargetAnalysisOptions(runtime_rules=ruby_rules(_ruby_core_classes())),
        )["Target#render"]
        folded = {
            name: dict(stats.folded_from)
            for name, stats in categories.items()
            if stats.folded_from
        }
        assert folded, f"expected runtime-internal folding (inline={inline_padding})"
        folded_leaves = {leaf for leaves in folded.values() for leaf in leaves}
        assert folded_leaves == {"Array#map"}


def test_clankerprof_no_enhanced_runtime_selection_is_observable(
    tmp_path: Path,
) -> None:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function("Target#render", "/app/responders/target.rb")
    )
    app_caller = builder.location(
        builder.function("App::Renderer#render", "/app/renderers/app_renderer.rb")
    )
    stdlib_delegator = builder.location(
        builder.function(
            "Forwardable#def_delegator",
            "/usr/local/lib/ruby/3.2.0/forwardable.rb",
        )
    )
    native_leaf = builder.location(builder.function("NativeExtension#work", "<cfunc>"))
    builder.sample((native_leaf, stdlib_delegator, app_caller, target), 5_000_000)

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    profile_path.write_bytes(builder.encode())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": r"[/\\]app[/\\]"}}),
        encoding="utf-8",
    )

    def top_category(*extra: str) -> str:
        output_path = tmp_path / f"out-{len(extra)}.json"
        assert (
            clankerprof_main(
                [
                    "targets",
                    "--profile",
                    str(profile_path),
                    "--config",
                    str(config_path),
                    "--no-enhanced",
                    *extra,
                    "--output",
                    str(output_path),
                ]
            )
            == 0
        )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        parent = cast(
            dict[str, Any],
            cast(dict[str, Any], payload["parents"])["Target#render"],
        )
        return cast(str, cast(list[dict[str, Any]], parent["categories"])[0]["name"])

    assert top_category() == "Other"
    assert top_category("--runtime", "ruby") == "Application"


def test_clankerprof_semantic_rules_do_not_claim_app_frames_by_substring() -> None:
    rules = ruby_rules(_ruby_core_classes())

    def category(name: str, filename: str) -> str | None:
        return categorize_ruby_frame(
            Frame(location_id=1, function_id=1, name=name, filename=filename),
            rules,
        )

    assert category("MyStatsDHelper#emit", "/srv/app/lib/my_statsd_helper.rb") is None
    assert category("ShopI18nAudit#run", "/app/services/shop_i18n_audit.rb") is None
    assert (
        category("ActiveRecordish::Auditor#call", "/app/models/auditor.rb") is None
    )

    assert (
        category(
            "StatsD::Batcher#flush",
            "/gems/statsd-instrument-3.5.0/lib/batcher.rb",
        )
        == "StatsD Gem"
    )
    assert (
        category("StatsD.distribution", "/usr/local/lib/ruby/3.2.0/forwardable.rb")
        == "StatsD Gem"
    )
    assert category("StatsD.increment", "<cfunc>") == "StatsD (Native)"


def test_clankerprof_special_namespace_guard_covers_bare_module_names() -> None:
    rules = ruby_rules(_ruby_core_classes())

    def category(name: str, filename: str) -> str | None:
        return categorize_ruby_frame(
            Frame(location_id=1, function_id=1, name=name, filename=filename),
            rules,
        )

    assert category("Zlib.inflate", "<cfunc>") == "Compression (Native)"
    assert (
        category("OpenSSL.fixed_length_secure_compare", "<cfunc>")
        == "OpenSSL (Native)"
    )
    assert category("Zlib::Deflate.deflate", "<cfunc>") == "Compression (Native)"
    assert category("OpenSSL::Cipher#encrypt", "<cfunc>") == "OpenSSL (Native)"


@covers("M9-001", "M9-006")
def test_clankerprof_fact_index_exposes_shared_stack_operations() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    index = ProfileFactIndex.from_input(profile.to_sample_facts())
    fact = next(
        sample
        for sample in index.samples()
        if any(frame.name == "RequestHandler#render_response" for frame in sample.stack)
        and any(frame.name == "ComponentRenderer#render" for frame in sample.stack)
        and any(frame.name == "CacheClient#get_multi" for frame in sample.stack)
    )

    assert index.total_primary_value == profile.total_primary_value()
    assert [
        frame.name
        for frame in index.target_frames(fact, {"RequestHandler#render_response"})
    ] == ["RequestHandler#render_response"]
    assert index.any_frame_matches(
        fact,
        lambda frame: frame.name == "ComponentRenderer#render",
    )
    caller = index.first_caller_after_leaf(
        fact,
        lambda frame: frame.name == "ComponentRenderer#render",
    )
    assert caller is not None
    assert caller.name == "ComponentRenderer#render"

    selection = index.select_bottom_frame(
        fact,
        is_eligible=lambda frame: not frame.filename.startswith("<"),
        is_collapsed=lambda frame: "cache-client" in frame.filename,
    )

    assert selection is not None
    assert selection.bottom.name == "ComponentRenderer#render"
    assert selection.found_uncollapsed_eligible is True


@covers("M9-001")
def test_clankerprof_sample_facts_preserve_location_folded_markers() -> None:
    profile = decode_profile_bytes(_folded_location_profile_bytes())
    facts = profile.sample_facts()

    assert facts[0].leaf is not None
    assert facts[0].leaf.name == "FoldedLeaf#work"
    assert facts[0].leaf.location_is_folded is True
    assert facts[0].stack[1].location_is_folded is False


@covers("M9-001", "M9-002")
def test_clankerprof_target_projection_matches_sample_fact_projection() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    config = {
        "RequestHandler#render_response": {
            "View Model": "app/view_models/**",
            "Components": "path:app/components/**",
            "Cache Client": "library:cache-client",
        }
    }

    profile_results = render_target_json(analyze_targets(profile, config))
    fact_results = render_target_json(
        analyze_target_facts(profile.to_sample_facts(), config)
    )

    assert fact_results == profile_results


@covers("M9-001", "M9-006")
def test_clankerprof_slice_projection_matches_sample_fact_projection() -> None:
    profile = decode_profile_bytes(_slice_semantics_profile_bytes())
    options = SliceAnalysisOptions(
        slices=(
            SliceDefinition("components", ("app/components/**",)),
            SliceDefinition("default", is_default=True),
        ),
        collapse=("library:*",),
        runtime_rules=ruby_rules(_ruby_core_classes()),
        unattributed_libraries=1,
    )

    profile_results = render_slice_json(analyze_slices(profile, options), options)
    fact_results = render_slice_json(
        analyze_slice_facts(profile.to_sample_facts(), options),
        options,
    )

    assert fact_results == profile_results


@covers("M9-002")
def test_clankerprof_preserves_target_attribution_parity() -> None:
    profile = decode_profile_bytes(_target_profile_bytes())
    results = analyze_targets(
        profile,
        {
            "Target#render": {
                "Template Engine Gem": r"[/\\]gems[/\\]template[_-]engine",
                "App Templates": r"[/\\]app[/\\]templates[/\\]",
                "Ruby Gems": r"[/\\]gems[/\\]",
            }
        },
    )

    categories = results["Target#render"]
    assert categories["App Templates"].cpu_time == 1_000_000
    assert categories["Other"].cpu_time == 7_000_000
    assert sum(stats.cpu_time for stats in categories.values()) == 8_000_000

    rendered = render_target_json(results)
    parent = cast(
        dict[str, object], cast(dict[str, object], rendered["parents"])["Target#render"]
    )
    assert parent["total_time_ns"] == 8_000_000


@covers("M9-002")
def test_clankerprof_supports_generic_request_rendering_attribution() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    results = analyze_targets(
        profile,
        {
            "RequestHandler#render_response": {
                "View Model": r"[/\\]app[/\\]view_models[/\\]",
                "Components": r"[/\\]app[/\\]components[/\\]",
                "Cache Client": r"[/\\]vendor[/\\]cache-client",
            }
        },
    )["RequestHandler#render_response"]

    assert results["View Model"].cpu_time == 10_000_000
    assert results["Components"].cpu_time == 20_000_000
    assert results["Cache Client"].cpu_time == 30_000_000
    assert results["Other"].cpu_time == 40_000_000
    assert sum(item.cpu_time for item in results.values()) == 100_000_000


@covers("M9-002")
def test_clankerprof_target_categories_support_simplified_path_patterns() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())
    results = analyze_targets(
        profile,
        {
            "RequestHandler#render_response": {
                "View Model": "app/view_models/**",
                "Components": "path:app/components/**",
                "Cache Client": "library:cache-client",
            }
        },
    )["RequestHandler#render_response"]

    assert results["View Model"].cpu_time == 10_000_000
    assert results["Components"].cpu_time == 20_000_000
    assert results["Cache Client"].cpu_time == 30_000_000
    assert results["Other"].cpu_time == 40_000_000


def _generic_boundary_options() -> BoundaryAnalysisOptions:
    return BoundaryAnalysisOptions(
        categories=(
            BoundaryCategoryDefinition(
                "View Models",
                parse_frame_predicates(("path:app/view_models/**",)),
            ),
            BoundaryCategoryDefinition(
                "Components",
                parse_frame_predicates(("path:app/components/**",)),
            ),
            BoundaryCategoryDefinition(
                "Cache Client",
                parse_frame_predicates(("library:cache-client",)),
            ),
            BoundaryCategoryDefinition(
                "Serialization",
                parse_frame_predicates(("name:JSON.generate",)),
            ),
        ),
        domains=(
            BoundaryDomainDefinition(
                "Component rendering",
                parse_frame_predicates(("path:app/components/**",)),
            ),
            BoundaryDomainDefinition(
                "View models",
                parse_frame_predicates(("path:app/view_models/**",)),
            ),
            BoundaryDomainDefinition(
                "Rendering fallback",
                parse_frame_predicates(("path:app/rendering/**",)),
                fallback=True,
            ),
        ),
        boundaries=(
            BoundaryDefinition(
                name="Request render",
                predicates=parse_frame_predicates(
                    ("name_eq:RequestHandler#render_response",),
                    default_key="name_eq",
                ),
                buckets={
                    "Application code": ("View Models", "Components"),
                    "Mechanics": ("Cache Client", "Serialization"),
                },
                attributables={"p90_ms": 200.0},
            ),
        ),
    )


@covers("M9-007")
def test_clankerprof_boundary_decomposition_tracks_domain_cost_kinds() -> None:
    profile = decode_profile_bytes(_boundary_decomposition_profile_bytes())
    result = analyze_boundaries(profile, _generic_boundary_options())
    rendered = render_boundary_json(result)

    boundary = cast(list[dict[str, object]], rendered["boundaries"])[0]
    assert boundary["name"] == "Request render"
    assert boundary["total_time_ns"] == 100_000_000

    buckets = {
        item["name"]: item
        for item in cast(list[dict[str, object]], boundary["buckets"])
    }
    assert buckets["Application code"]["time_ns"] == 30_000_000
    assert buckets["Mechanics"]["time_ns"] == 70_000_000
    assert cast(dict[str, float], buckets["Mechanics"]["attributable_estimates"]) == {
        "p90_ms": 140.0
    }

    domains = {
        item["name"]: item
        for item in cast(list[dict[str, object]], boundary["domains"])
    }
    component = domains["Component rendering"]
    assert component["time_ns"] == 50_000_000
    component_costs = {
        item["name"]: item
        for item in cast(list[dict[str, object]], component["cost_kinds"])
    }
    assert component_costs["Cache Client"]["time_ns"] == 30_000_000
    assert component_costs["Components"]["time_ns"] == 20_000_000

    view_models = domains["View models"]
    assert view_models["time_ns"] == 50_000_000
    view_costs = {
        item["name"]: item
        for item in cast(list[dict[str, object]], view_models["cost_kinds"])
    }
    assert view_costs["Serialization"]["time_ns"] == 40_000_000
    files = cast(list[dict[str, object]], view_models["files"])
    assert files[0]["filename"] == "/app/view_models/report_view.rb"
    pairs = cast(list[dict[str, object]], files[0]["caller_leaf_pairs"])
    assert pairs[0]["caller"] == "ReportView#build"
    assert pairs[0]["leaf"] == "JSON.generate"


@covers("M9-007")
def test_clankerprof_boundary_config_cli_replays_sample_facts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    facts_path = tmp_path / "facts.json"
    config_path = tmp_path / "boundaries.toml"
    output_path = tmp_path / "boundaries.json"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    facts_path.write_text(
        dumps_sample_facts(load_profile(str(profile_path)).to_sample_facts()),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[category]
"View Models" = "path:app/view_models/**"
"Components" = "path:app/components/**"
"Cache Client" = "library:cache-client"
"Serialization" = "name:JSON.generate"

[domain]
"Component rendering" = "path:app/components/**"
"View models" = "path:app/view_models/**"
"Rendering fallback" = { patterns = ["path:app/rendering/**"], fallback = true }

[[boundary]]
label = "Request render"
function = "RequestHandler#render_response"

[boundary.attributables]
p90_ms = 200.0

[boundary.bucket]
"Application code" = ["View Models", "Components"]
"Mechanics" = ["Cache Client", "Serialization"]
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "boundaries",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    profile_payload = json.loads(capsys.readouterr().out)
    boundary = cast(list[dict[str, object]], profile_payload["boundaries"])[0]
    assert boundary["total_time_ns"] == 100_000_000
    assert boundary["domains"]

    assert (
        autoclanker_main(
            [
                "pprof",
                "boundaries",
                "--facts",
                str(facts_path),
                "--config",
                str(config_path),
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    alias_payload = json.loads(capsys.readouterr().out)
    assert alias_payload == {
        "tool": "clankerprof_boundaries",
        "ok": True,
        "output": str(output_path),
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == profile_payload


@covers("M9-007")
def test_clankerprof_boundary_function_shorthand_handles_colon_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = PprofFixtureBuilder.create()
    boundary = builder.location(
        builder.function(
            "Example::BaseResponder#render",
            "/app/responders/base_responder.rb",
        )
    )
    leaf = builder.location(builder.function("Presenter#call", "/app/presenter.rb"))
    builder.sample((leaf, boundary), 7_000_000)

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "boundaries.toml"
    profile_path.write_bytes(builder.encode())
    config_path.write_text(
        """
[category]
"Application" = "path:app/**"

[[boundary]]
label = "Request render"
function = "Example::BaseResponder#render"
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "boundaries",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    boundary_payload = cast(list[dict[str, object]], payload["boundaries"])[0]
    assert boundary_payload["total_time_ns"] == 7_000_000


@covers("M9-007")
def test_clankerprof_boundary_config_supports_predicate_expressions_and_category_refs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    rules_path = tmp_path / "runtime.yml"
    config_path = tmp_path / "boundaries.toml"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    rules_path.write_text(
        """
name: test-runtime
semantic_rules:
  - category: Runtime Serialization
    name_prefixes:
      - JSON.
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        """
[category]
"View Models" = { all = ["path:app/**"], not = "path:app/components/**" }
"Components" = "path:app/components/**"
"Serialization grouped" = "runtime_category:Runtime Serialization"

[domain]
"Non-component application" = { all = ["path:app/**"], not = "path:app/components/**" }
"Component rendering" = "category:Components"

[[boundary]]
label = "Request render"
match = { all = ["name_eq:RequestHandler#render_response"], not = "path:app/jobs/**" }

[boundary.bucket]
"Application code" = ["View Models", "Components"]
"Runtime work" = ["Serialization grouped"]
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "boundaries",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime-rules",
                str(rules_path),
                "--no-enhanced",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    boundary = cast(list[dict[str, object]], payload["boundaries"])[0]
    buckets = {
        item["name"]: item
        for item in cast(list[dict[str, object]], boundary["buckets"])
    }
    assert buckets["Application code"]["time_ns"] == 30_000_000
    assert buckets["Runtime work"]["time_ns"] == 40_000_000

    domains = {
        item["name"]: item
        for item in cast(list[dict[str, object]], boundary["domains"])
    }
    assert domains["Non-component application"]["time_ns"] == 50_000_000
    assert domains["Component rendering"]["time_ns"] == 50_000_000
    domain_costs = {
        item["name"]: item
        for item in cast(
            list[dict[str, object]],
            domains["Non-component application"]["cost_kinds"],
        )
    }
    assert domain_costs["Serialization grouped"]["time_ns"] == 40_000_000


@covers("M9-007")
def test_clankerprof_scope_config_aliases_preserve_boundary_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    rules_path = tmp_path / "runtime.yml"
    legacy_config_path = tmp_path / "boundaries.toml"
    scope_config_path = tmp_path / "scopes.toml"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    rules_path.write_text(
        """
name: test-runtime
semantic_rules:
  - category: Runtime Serialization
    name_prefixes:
      - JSON.
""".strip(),
        encoding="utf-8",
    )
    legacy_config_path.write_text(
        """
[category]
"View Models" = { all = ["path:app/**"], not = "path:app/components/**" }
"Components" = "path:app/components/**"
"Serialization grouped" = "runtime_category:Runtime Serialization"

[domain]
"Non-component application" = { all = ["path:app/**"], not = "path:app/components/**" }
"Component rendering" = "category:Components"

[[boundary]]
label = "Request render"
match = { all = ["name_eq:RequestHandler#render_response"], not = "path:app/jobs/**" }

[boundary.bucket]
"Application code" = ["View Models", "Components"]
"Runtime work" = ["Serialization grouped"]
""".strip(),
        encoding="utf-8",
    )
    scope_config_path.write_text(
        """
[cost_kind]
"View Models" = { all = ["path:app/**"], not = "path:app/components/**" }
"Components" = "path:app/components/**"
"Serialization grouped" = "runtime_label:Runtime Serialization"

[owner]
"Non-component application" = { all = ["path:app/**"], not = "path:app/components/**" }
"Component rendering" = "cost_kind:Components"

[[scope]]
label = "Request render"
selector = { all = ["name_eq:RequestHandler#render_response"], not = "path:app/jobs/**" }

[scope.rollup]
"Application code" = ["View Models", "Components"]
"Runtime work" = ["Serialization grouped"]
""".strip(),
        encoding="utf-8",
    )

    legacy_args = [
        "boundaries",
        "--profile",
        str(profile_path),
        "--config",
        str(legacy_config_path),
        "--runtime-rules",
        str(rules_path),
        "--no-enhanced",
    ]
    scope_args = [
        "scopes",
        "--profile",
        str(profile_path),
        "--config",
        str(scope_config_path),
        "--runtime-rules",
        str(rules_path),
        "--no-enhanced",
    ]

    assert clankerprof_main(legacy_args) == 0
    legacy_payload = json.loads(capsys.readouterr().out)
    assert clankerprof_main(scope_args) == 0
    scope_payload = json.loads(capsys.readouterr().out)

    assert scope_payload == legacy_payload
    assert scope_payload["tool"] == "clankerprof_boundaries"
    assert "boundaries" in scope_payload


@covers("M9-007")
def test_clankerprof_scope_config_rejects_mixed_preferred_and_legacy_sections(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "scopes.toml"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    config_path.write_text(
        """
[cost_kind]
"Application" = "path:app/**"

[category]
"Runtime" = "name:JSON.generate"

[[scope]]
label = "Request render"
function = "RequestHandler#render_response"
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "scopes",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().err)
    assert "Use only one of [cost_kind] or [category]" in payload["error"]


@covers("M9-007")
def test_clankerprof_boundary_config_rejects_recursive_category_predicates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "boundaries.toml"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    config_path.write_text(
        """
[category]
"Outer" = "category:Inner"
"Inner" = "path:app/**"

[[boundary]]
label = "Request render"
function = "RequestHandler#render_response"
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "boundaries",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().err)
    assert "cannot reference category:" in payload["error"]


@covers("M9-007")
def test_clankerprof_boundary_exclusions_build_residual_scopes() -> None:
    options = BoundaryAnalysisOptions(
        categories=(
            BoundaryCategoryDefinition(
                "Application",
                parse_frame_predicates(("path:app/**",)),
            ),
        ),
        boundaries=(
            BoundaryDefinition(
                name="Request total",
                predicates=parse_frame_predicates(
                    ("name_eq:MiddlewareStack#call",),
                    default_key="name_eq",
                ),
                buckets={"Application": ("Application",)},
            ),
            BoundaryDefinition(
                name="Request outside render",
                predicates=parse_frame_predicates(
                    ("name_eq:MiddlewareStack#call",),
                    default_key="name_eq",
                ),
                buckets={"Application": ("Application",)},
                exclude_descendants=parse_frame_predicates(
                    ("name_eq:RequestHandler#render_response",),
                    default_key="name_eq",
                ),
            ),
        ),
    )

    rendered = render_boundary_json(
        analyze_boundaries(
            decode_profile_bytes(_boundary_decomposition_profile_bytes()),
            options,
        )
    )
    boundaries = {
        item["name"]: item
        for item in cast(list[dict[str, object]], rendered["boundaries"])
    }
    assert boundaries["Request total"]["total_time_ns"] == 100_000_000
    assert boundaries["Request outside render"]["total_time_ns"] == 0


@covers("M9-007")
def test_clankerprof_boundary_predicate_matching_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = decode_profile_bytes(
        _boundary_decomposition_profile_bytes(sample_repetitions=100)
    )
    options = _generic_boundary_options()
    original = clankerprof_analysis.match_path_pattern
    call_count = 0

    def counted_match_path_pattern(
        pattern: str,
        path: str,
        rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        return original(pattern, path, rules)

    monkeypatch.setattr(
        clankerprof_analysis,
        "match_path_pattern",
        counted_match_path_pattern,
    )

    result = analyze_boundary_facts(profile.to_sample_facts(), options)

    assert result.boundaries[0].total_time == 10_000_000_000
    assert call_count < 40


@covers("M9-007")
def test_clankerprof_boundary_expression_matching_stays_frame_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = decode_profile_bytes(
        _boundary_decomposition_profile_bytes(sample_repetitions=100)
    )
    options = BoundaryAnalysisOptions(
        categories=(
            BoundaryCategoryDefinition(
                "Non-job application",
                FramePredicateExpr(
                    all=(frame_predicate_expr(("path:app/**",)),),
                    not_=(frame_predicate_expr(("path:app/jobs/**",)),),
                ),
            ),
        ),
        boundaries=(
            BoundaryDefinition(
                name="Request render",
                predicates=FramePredicateExpr(
                    all=(
                        frame_predicate_expr(
                            ("name_eq:RequestHandler#render_response",),
                            default_key="name_eq",
                        ),
                    ),
                    not_=(frame_predicate_expr(("path:app/jobs/**",)),),
                ),
            ),
        ),
    )
    original = clankerprof_analysis.match_path_pattern
    call_count = 0

    def counted_match_path_pattern(
        pattern: str,
        path: str,
        rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        return original(pattern, path, rules)

    monkeypatch.setattr(
        clankerprof_analysis,
        "match_path_pattern",
        counted_match_path_pattern,
    )

    result = analyze_boundary_facts(profile.to_sample_facts(), options)

    assert result.boundaries[0].total_time == 10_000_000_000
    assert call_count < 50


@covers("M9-007")
def test_clankerprof_boundary_category_predicates_stay_frame_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = decode_profile_bytes(
        _boundary_decomposition_profile_bytes(sample_repetitions=100)
    )
    options = BoundaryAnalysisOptions(
        categories=(
            BoundaryCategoryDefinition(
                "View Models",
                parse_frame_predicates(("path:app/view_models/**",)),
            ),
            BoundaryCategoryDefinition(
                "Components",
                parse_frame_predicates(("path:app/components/**",)),
            ),
            BoundaryCategoryDefinition(
                "Serialization",
                parse_frame_predicates(("name:JSON.generate",)),
            ),
        ),
        domains=(
            BoundaryDomainDefinition(
                "Component rendering",
                parse_frame_predicates(("category:Components",)),
            ),
            BoundaryDomainDefinition(
                "View models",
                parse_frame_predicates(("category:View Models",)),
            ),
        ),
        boundaries=(
            BoundaryDefinition(
                name="Request render",
                predicates=parse_frame_predicates(
                    ("name_eq:RequestHandler#render_response",),
                    default_key="name_eq",
                ),
            ),
        ),
    )
    original = clankerprof_analysis.match_path_pattern
    call_count = 0

    def counted_match_path_pattern(
        pattern: str,
        path: str,
        rules: RuntimeRuleSet = DEFAULT_RUNTIME_RULES,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        return original(pattern, path, rules)

    monkeypatch.setattr(
        clankerprof_analysis,
        "match_path_pattern",
        counted_match_path_pattern,
    )

    result = analyze_boundary_facts(profile.to_sample_facts(), options)

    assert result.boundaries[0].total_time == 10_000_000_000
    assert call_count < 50


@covers("M9-002")
def test_clankerprof_path_pattern_matching_keeps_legacy_regex_and_gem_compatibility() -> (
    None
):
    path = "/workspace/app/view_models/card.rb"

    assert match_category_pattern(r"[/\\]app[/\\]view_models[/\\]", path)
    assert match_category_pattern("app/view_models/**", path)
    assert match_category_pattern("path:app/view_models/**", path)
    assert match_category_pattern(r"regex:[/\\]app[/\\]view_models[/\\]", path)
    assert match_path_pattern("app/view_models/**", path)
    assert match_path_pattern("regex:[/\\\\]app[/\\\\]view_models[/\\\\]", path)
    assert match_category_pattern(
        "gem:cache-client",
        "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
    )
    assert match_category_pattern(
        "library:cache-client",
        "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
        ruby_rules(_ruby_core_classes()),
    )


@covers("M9-003", "M9-004")
def test_clankerprof_dependency_selectors_are_runtime_rule_driven() -> None:
    generic_rules = load_runtime_rules("generic")
    ruby = ruby_rules(_ruby_core_classes())

    assert (
        extract_library_name(
            "/srv/app/vendor/cache-client-1.2.3/lib/client.rb",
            generic_rules,
        )
        == "cache-client"
    )
    assert (
        extract_library_name(
            "/workspace/node_modules/@maps/tile-cache/dist/index.js",
            generic_rules,
        )
        == "@maps/tile-cache"
    )
    assert (
        extract_library_name(
            "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
            generic_rules,
        )
        is None
    )
    assert (
        extract_library_name(
            "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
            ruby,
        )
        == "cache-client"
    )
    assert (
        extract_gem_name("/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb")
        == "cache-client"
    )
    assert match_path_pattern(
        "dependency:cache-client",
        "/srv/app/vendor/cache-client-1.2.3/lib/client.rb",
        generic_rules,
    )
    assert match_path_pattern(
        "library:cache-client",
        "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
        ruby,
    )
    assert match_path_pattern(
        "gem:cache-client",
        "/bundle/ruby/4.0.0/gems/cache-client-1.2.3/lib/client.rb",
        generic_rules,
    )


@covers("M9-005")
def test_clankerprof_target_cli_supports_minimal_target_shortcut(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_request_rendering_profile_bytes())

    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--target",
                "RequestHandler#render_response",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    parent = cast(
        dict[str, object],
        cast(dict[str, object], payload["parents"])["RequestHandler#render_response"],
    )
    categories = cast(list[dict[str, object]], parent["categories"])
    assert parent["total_time_ns"] == 100_000_000
    assert categories[0]["name"] == "Other"
    assert categories[0]["time_ns"] == 100_000_000
    assert categories[0]["samples"] == 4


@covers("M9-003")
def test_clankerprof_ruby_rules_support_simplification_folding_and_attributables() -> (
    None
):
    profile = decode_profile_bytes(_ruby_profile_bytes())
    config = {
        "Target#render": {
            "Application": r"[/\\]app[/\\]",
            "Gem": r"[/\\]gems[/\\]",
        }
    }
    core = {"String", "Kernel", "Marshal"}

    simplified = analyze_targets(
        profile,
        config,
        TargetAnalysisOptions(runtime_rules=ruby_rules(core)),
    )["Target#render"]
    assert simplified["Ruby Core (Native)"].cpu_time == 1_000_000_000
    assert simplified["Instrumentation Overhead"].cpu_time == 2_000_000_000
    assert simplified["Serialization Overhead"].cpu_time == 3_000_000_000
    assert simplified["Template Engine Native"].cpu_time == 4_000_000_000

    folded = analyze_targets(
        profile,
        config,
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(core, verbose=True),
            fold_runtime_internals=True,
            track_semantic_callers=True,
        ),
    )["Target#render"]
    assert "Ruby Core (Native)" not in folded
    assert folded["Application"].cpu_time == 4_000_000_000
    assert folded["StatsD Gem"].cpu_time == 2_000_000_000
    assert folded["Template Engine Native"].cpu_time == 4_000_000_000
    assert folded["Application"].folded_from == {
        "String#gsub": 1_000_000_000,
        "Marshal.load": 3_000_000_000,
    }

    csv_output = render_target_csv(
        {"Target#render": folded},
        attributables={"request_count": {"Target#render": 10_000}},
        simplified=True,
    )
    assert "request_count" in csv_output
    assert "Application" in csv_output
    assert "4000.0" in csv_output


@covers("M9-003")
def test_clankerprof_loads_external_runtime_rule_packs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    app = builder.location(builder.function("App::Page#render", "/srv/app/page.py"))
    allocation = builder.location(builder.function("Object#new", "<native>"))
    trace = builder.location(
        builder.function("TraceLib::Span#finish", "/deps/trace-lib/span.py")
    )
    native_constructor = builder.location(
        builder.function("ReportBuilder::Row.new", "<native>")
    )
    view_constructor = builder.location(
        builder.function("ReceiptView::Line.new", "<native>")
    )
    builder.sample((allocation, app, target), 10_000_000)
    builder.sample((allocation, trace, target), 20_000_000)
    builder.sample((trace, target), 30_000_000)
    builder.sample((allocation, native_constructor, target), 5_000_000)
    builder.sample((allocation, view_constructor, target), 7_000_000)

    profile = decode_profile_bytes(builder.encode())
    rules_path = tmp_path / "runtime-rules.yml"
    rules_path.write_text(
        """
name: demo-runtime
core_native_default_category: Runtime Allocation
native_path_markers:
  - <native>
library_path_patterns:
  - regex:/deps/([^/]+)/
library_selector_path_patterns:
  plugin:
    - regex:/plugins/([^/]+)/
library_name_suffix_patterns:
  - -[0-9].*
semantic_rules:
  - category: Tracing Library
    name_contains:
      - TraceLib
native_name_category_rules:
  - category: Presentation Model
    name_patterns:
      - '^(?=.*View)[A-Z][A-Za-z0-9_]*(::[A-Z][A-Za-z0-9_]*)*\\.new$'
  - category: Application
    name_patterns:
      - '^[A-Z][A-Za-z0-9_]*(::[A-Z][A-Za-z0-9_]*)*\\.new$'
always_foldable_categories:
  - Runtime Allocation
""",
        encoding="utf-8",
    )
    runtime_rules = load_runtime_rules_file(
        rules_path,
        core_classes={"Object"},
    )
    assert match_path_pattern(
        "plugin:maps-extension",
        "/srv/plugins/maps-extension-2.1.0/lib/extension.py",
        runtime_rules,
    )

    result = analyze_targets(
        profile,
        {"Target#render": {"Application": "path:/srv/app/**"}},
        TargetAnalysisOptions(
            runtime_rules=runtime_rules,
            fold_runtime_internals=True,
            track_semantic_callers=True,
        ),
    )["Target#render"]

    assert result["Application"].cpu_time == 15_000_000
    assert result["Application"].folded_from == {"Object#new": 15_000_000}
    assert result["Presentation Model"].cpu_time == 7_000_000
    assert result["Presentation Model"].folded_from == {"Object#new": 7_000_000}
    assert result["Tracing Library"].cpu_time == 50_000_000
    assert result["Tracing Library"].folded_from == {"Object#new": 20_000_000}

    config_path = tmp_path / "target-config.json"
    profile_path = tmp_path / "profile.pb"
    attributables_path = tmp_path / "attributables.json"
    profile_path.write_bytes(builder.encode())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": "path:/srv/app/**"}}),
        encoding="utf-8",
    )
    attributables_path.write_text(
        json.dumps({"p90_ms": {"Target#render": 120.0}}),
        encoding="utf-8",
    )
    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime-rules",
                str(rules_path),
                "--core-classes",
                str(_write_core_csv(tmp_path)),
                "--fold-runtime-internals",
                "--format",
                "simple-csv",
                "--attributables",
                str(attributables_path),
            ]
        )
        == 0
    )
    rendered = capsys.readouterr().out
    assert "Tracing Library" in rendered
    assert "Presentation Model" in rendered
    assert "p90_ms" in rendered
    assert "83.3" in rendered

    assert runtime_rules_from_file(rules_path, core_classes={"Object"}).name == (
        "demo-runtime"
    )


@covers("M9-003")
def test_clankerprof_loads_packaged_ruby_core_classes_by_default(
    tmp_path: Path,
) -> None:
    assert len(load_default_ruby_core_classes()) > 1_000

    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/app/target.rb"))
    monitor = builder.location(builder.function("Monitor#synchronize", "<cfunc>"))
    builder.sample((monitor, target), 11_000_000)

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    output_path = tmp_path / "targets.json"
    profile_path.write_bytes(builder.encode())
    config_path.write_text(json.dumps({"Target#render": {}}), encoding="utf-8")

    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime",
                "ruby",
                "--format",
                "json",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    parent = cast(
        dict[str, object], cast(dict[str, object], payload["parents"])["Target#render"]
    )
    categories = cast(list[dict[str, object]], parent["categories"])
    assert categories[0]["name"] == "Ruby Core (Native)"
    assert categories[0]["time_ns"] == 11_000_000


@covers("M9-003")
def test_clankerprof_ruby_rule_pack_preserves_legacy_categorization_cases() -> None:
    rules = ruby_rules(_ruby_core_classes())

    def category(name: str, filename: str) -> str | None:
        return categorize_ruby_frame(
            Frame(
                location_id=1,
                function_id=1,
                name=name,
                filename=filename,
            ),
            rules,
        )

    cases = [
        ("String#gsub", "<cfunc>", "Ruby Core (Native)"),
        ("Array#map", "<cfunc>", "Ruby Core (Native)"),
        ("Hash#each", "<cfunc>", "Ruby Core (Native)"),
        ("Enumerator#next", "<cfunc>", "Ruby Core (Native)"),
        ("Integer#to_s", "<cfunc>", "Ruby Core (Native)"),
        ("Time.now", "<cfunc>", "Ruby Core (Native)"),
        ("File.read", "<cfunc>", "Ruby Core (Native)"),
        ("Dir.glob", "<cfunc>", "Ruby Core (Native)"),
        ("Class#new", "<cfunc>", "Ruby Core (Native)"),
        ("Module#include", "<cfunc>", "Ruby Core (Native)"),
        ("Object#send", "<cfunc>", "Ruby Core (Native)"),
        ("foo", "<internal:kernel>", "Ruby Internals"),
        ("bar", "<internal:marshal>", "Ruby Internals"),
        (
            "Enumerable#index_by",
            "/usr/local/lib/ruby/3.2.0/enumerable.rb",
            "Ruby Stdlib",
        ),
        ("Enumerable#map", "/usr/lib/ruby/3.2.0/enumerable.rb", "Ruby Stdlib"),
        ("Enumerable#index_by", "/app/vendor/ruby/3.2.0/lib/enumerable.rb", None),
        ("Digest::MD5#digest", "<cfunc>", "Digest (Native)"),
        ("BigDecimal#to_s", "<cfunc>", "Ruby Core (Native)"),
        ("CGI::escape", "<cfunc>", "Ruby Core (Native)"),
        ("Set#add", "<cfunc>", "Ruby Core (Native)"),
        ("StringScanner#scan", "<cfunc>", "Ruby Core (Native)"),
        ("Random.bytes", "<cfunc>", "Ruby Core (Native)"),
        ("TemplateEngine::Native#render", "<cfunc>", "Template Engine Native"),
        ("MessagePack::Unpacker#read", "<cfunc>", "Serialization (Native)"),
        ("Snappy.inflate", "<cfunc>", "Compression (Native)"),
        ("Zlib::Deflate.deflate", "<cfunc>", "Compression (Native)"),
        ("Trilogy#query", "<cfunc>", "Trilogy (Native)"),
        ("OpenSSL::Cipher#encrypt", "<cfunc>", "OpenSSL (Native)"),
        ("OpenSSL::Digest#initialize", "<cfunc>", "OpenSSL (Native)"),
        ("OpenSSL::HMAC#update", "<cfunc>", "OpenSSL (Native)"),
        ("OpenSSL::Cipher#decrypt", "<cfunc>", "OpenSSL (Native)"),
        ("OpenSSL::PKey::EC#initialize", "<cfunc>", "OpenSSL (Native)"),
        ("JSON::parse", "<cfunc>", "JSON (Native)"),
        ("JSON::State#initialize", "<cfunc>", "JSON (Native)"),
        ("JSONC.Parser#parse", "<cfunc>", "JSON (Native)"),
        ("Psych::load", "<cfunc>", "YAML (Native)"),
        ("Psych::Parser#parse", "<cfunc>", "YAML (Native)"),
        ("Thread::Mutex#lock", "<cfunc>", "Ruby Threading"),
        ("IO#read", "<cfunc>", "IO (Native)"),
        ("StatsD.increment", "<cfunc>", "StatsD (Native)"),
        (
            "StatsD.distribution",
            "/usr/local/lib/ruby/3.2.0/forwardable.rb",
            "StatsD Gem",
        ),
        (
            "OpenTelemetry::SDK::Internal#valid_simple_value?",
            "<cfunc>",
            "OpenTelemetry (Native)",
        ),
        (
            "OpenTelemetry::Context.stack",
            "/gems/opentelemetry-api/lib/context.rb",
            "OpenTelemetry Gems",
        ),
        (
            "ActiveSupport::Cache::Entry#value",
            "/gems/activesupport/lib/cache.rb",
            "ActiveSupport Gem",
        ),
        (
            "ActiveRecord::Base.connection",
            "/gems/activerecord/lib/base.rb",
            "ActiveRecord Gem",
        ),
        ("I18n.translate", "/gems/i18n/lib/i18n.rb", "I18n Gem"),
        (
            "I18n::Backend::Base#translate",
            "/gems/i18n/lib/i18n/backend/base.rb",
            "I18n Gem",
        ),
        (
            "CacheClient::Native#with_cache",
            "<cfunc>",
            "Cache Client (Native)",
        ),
        (
            "CacheClient::Memcached#read_multi",
            "/app/gems/cache-client/lib/cache_client/memcached.rb",
            "Cache Client Gem",
        ),
        (
            "Dalli::Protocol::ConnectionManager#flush",
            "/gems/dalli/lib/dalli.rb",
            "Dalli Gem",
        ),
        ("Dalli::Client#get", "<cfunc>", "Dalli (Native)"),
        ("Trilogy::Connection#query", "/gems/trilogy/lib/trilogy.rb", "Trilogy Gem"),
        (
            "MessageCodec::Types.time_unpack",
            "/gems/message-codec/lib/types.rb",
            "Message Codec Gem",
        ),
        ("MessageCodec::Native#load", "<cfunc>", "Serialization (Native)"),
        ("Net::HTTP#get", "<cfunc>", None),
        ("::Array#map", "<cfunc>", "Ruby Core (Native)"),
        ("Foo::String#process", "/app/lib/foo.rb", None),
        ("App::Array#custom", "/app/models/array.rb", None),
        (
            "Hash#stringify_keys",
            "/gems/activesupport/lib/active_support/core_ext/hash/keys.rb",
            None,
        ),
        (
            "Hash#deep_symbolize_keys",
            "/gems/activesupport/lib/active_support/core_ext/hash/keys.rb",
            None,
        ),
        (
            "Object#present?",
            "/gems/activesupport/lib/active_support/core_ext/object/blank.rb",
            None,
        ),
        ("Monitor#synchronize", "<cfunc>", "Ruby Core (Native)"),
        ("Ractor#[]", "<internal:ractor>", "Ruby Internals"),
        ("Ractor.current", "<internal:ractor>", "Ruby Internals"),
    ]
    for name, filename, expected in cases:
        assert category(name, filename) == expected


@covers("M9-003")
@pytest.mark.parametrize(
    ("verbose", "fold", "present", "absent"),
    [
        (
            False,
            False,
            {
                "Instrumentation Overhead",
                "Serialization Overhead",
                "Translation & HTML Sanitization",
                "Template Engine Native",
                "Ruby Core (Native)",
                "Ruby Internals",
            },
            {"StatsD Gem", "OpenTelemetry Gems"},
        ),
        (
            True,
            False,
            {
                "StatsD Gem",
                "OpenTelemetry Gems",
                "Serialization (Native)",
                "OpenSSL (Native)",
                "Message Codec Gem",
                "I18n Gem",
                "Template Engine Native",
                "Ruby Core (Native)",
                "Ruby Internals",
            },
            {
                "Instrumentation Overhead",
                "Serialization Overhead",
                "Translation & HTML Sanitization",
            },
        ),
        (
            False,
            True,
            {
                "Instrumentation Overhead",
                "Serialization Overhead",
                "Translation & HTML Sanitization",
                "Template Engine Native",
            },
            {
                "Ruby Core (Native)",
                "Ruby Internals",
                "StatsD Gem",
                "OpenTelemetry Gems",
            },
        ),
        (
            True,
            True,
            {
                "StatsD Gem",
                "OpenTelemetry Gems",
                "Message Codec Gem",
                "I18n Gem",
                "Template Engine Native",
            },
            {
                "Ruby Core (Native)",
                "Ruby Internals",
                "Serialization (Native)",
                "OpenSSL (Native)",
                "Instrumentation Overhead",
                "Serialization Overhead",
            },
        ),
    ],
)
def test_clankerprof_preserves_legacy_ruby_flag_combinations(
    verbose: bool,
    fold: bool,
    present: set[str],
    absent: set[str],
) -> None:
    profile = decode_profile_bytes(_legacy_flag_matrix_profile_bytes())
    categories = analyze_targets(
        profile,
        {"Example::HtmlResponder#render_template": {}},
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes(), verbose=verbose),
            fold_runtime_internals=fold,
        ),
    )["Example::HtmlResponder#render_template"]

    assert sum(item.cpu_time for item in categories.values()) == 9_400_000_000
    assert present <= set(categories)
    assert absent.isdisjoint(categories)


@covers("M9-003")
def test_clankerprof_preserves_main_simplified_category_totals_and_never_folds() -> (
    None
):
    profile = decode_profile_bytes(_main_category_profile_bytes())
    config = {
        "Example::HtmlResponder#render_template": {
            "Application": r"[/\\]app[/\\]",
            "OpenTelemetry Gems": r"[/\\]gems[/\\]opentelemetry",
            "StatsD Gem": r"[/\\]gems[/\\]statsd",
            "ActiveSupport Gem": r"[/\\]gems[/\\]activesupport",
            "I18n Gem": r"[/\\]gems[/\\]i18n",
        }
    }

    for fold in (False, True):
        categories = analyze_targets(
            profile,
            config,
            TargetAnalysisOptions(
                runtime_rules=ruby_rules(_ruby_core_classes()),
                fold_runtime_internals=fold,
            ),
        )["Example::HtmlResponder#render_template"]
        assert categories["Serialization Overhead"].cpu_time == 3_000_000_000
        assert categories["I/O Overhead"].cpu_time == 2_500_000_000
        assert categories["Instrumentation Overhead"].cpu_time == 1_500_000_000
        assert categories["Translation & HTML Sanitization"].cpu_time == 2_000_000_000


@covers("M9-003")
def test_clankerprof_exports_semantic_caller_csv(tmp_path: Path) -> None:
    profile = decode_profile_bytes(_ruby_profile_bytes())
    results = analyze_targets(
        profile,
        {"Target#render": {"Application": r"[/\\]app[/\\]", "Gem": r"[/\\]gems[/\\]"}},
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes(), verbose=True),
            fold_runtime_internals=True,
            track_semantic_callers=True,
        ),
    )
    rendered = render_semantic_callers_csv(
        results,
        runtime_rules=ruby_rules(_ruby_core_classes()),
    )
    assert (
        "Parent Function,Category,Leaf Function,Leaf Samples,Top Caller,"
        "Caller Samples,Caller File"
    ) in rendered
    assert "Kernel#clone" in rendered
    assert "StatsD::Instrument::Aggregator#increment" in rendered
    assert "deps/statsd-instrument/lib/statsd/instrument/aggregator.rb" in rendered

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    semantic_path = tmp_path / "semantic-callers.csv"
    profile_path.write_bytes(_ruby_profile_bytes())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": r"[/\\]app[/\\]"}}),
        encoding="utf-8",
    )
    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime",
                "ruby",
                "--ruby-core-classes",
                str(_write_core_csv(tmp_path)),
                "--track-semantic-callers",
                "--semantic-callers-csv",
                str(semantic_path),
            ]
        )
        == 0
    )
    assert "Kernel#clone" in semantic_path.read_text(encoding="utf-8")


@covers("M9-003")
def test_clankerprof_text_report_includes_folded_and_semantic_caller_sections() -> None:
    profile = decode_profile_bytes(_ruby_profile_bytes())
    results = analyze_targets(
        profile,
        {"Target#render": {"Application": r"[/\\]app[/\\]", "Gem": r"[/\\]gems[/\\]"}},
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes(), verbose=True),
            fold_runtime_internals=True,
            track_semantic_callers=True,
        ),
    )

    rendered = render_target_text(
        results,
        show_folded=True,
        show_semantic_callers=True,
    )

    assert "Total runtime internals folded into categories" in rendered
    assert "Runtime internals folded into categories" in rendered
    assert "Semantic callers for runtime internals" in rendered
    assert "Kernel#clone" in rendered
    assert "StatsD::Instrument::Aggregator#increment" in rendered


@covers("M9-003")
def test_clankerprof_target_csv_skips_runtime_stdlib_when_selecting_callsite() -> None:
    profile = decode_profile_bytes(_stdlib_caller_profile_bytes())
    results = analyze_targets(
        profile,
        {"Target#render": {"Application": r"[/\\]app[/\\]"}},
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes(), verbose=True),
            fold_runtime_internals=True,
        ),
    )
    csv_output = render_target_csv(results, simplified=True)
    assert "App::Presenter#call" in csv_output
    assert "Forwardable#_delegator_method" not in csv_output


@covers("M9-003")
def test_clankerprof_supports_legacy_no_enhanced_native_caller_fallback(
    tmp_path: Path,
) -> None:
    profile = decode_profile_bytes(_legacy_no_enhanced_profile_bytes())
    config = {"Target#render": {"Application": r"[/\\]app[/\\]"}}

    generic = analyze_targets(profile, config)["Target#render"]
    assert generic["Other"].cpu_time == 5_000_000

    no_enhanced = analyze_targets(
        profile,
        config,
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes()),
            enhanced_runtime_categorization=False,
            legacy_no_enhanced_caller_fallback=True,
        ),
    )["Target#render"]
    assert no_enhanced["Application"].cpu_time == 5_000_000

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    output_path = tmp_path / "targets.json"
    profile_path.write_bytes(_legacy_no_enhanced_profile_bytes())
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--no-enhanced",
                "--format",
                "json",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    parent = cast(
        dict[str, object], cast(dict[str, object], payload["parents"])["Target#render"]
    )
    categories = cast(list[dict[str, object]], parent["categories"])
    assert categories[0]["name"] == "Application"


@covers("M9-003")
def test_clankerprof_caller_fallback_prefixes_are_generic_rule_config(
    tmp_path: Path,
) -> None:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    app_caller = builder.location(
        builder.function("App::Renderer#render", "/srv/app/renderer.py")
    )
    delegated_leaf = builder.location(
        builder.function("Delegator.forward", "/opt/runtime/stdlib/delegator.py")
    )
    builder.sample((delegated_leaf, app_caller, target), 9_000_000)
    profile = decode_profile_bytes(builder.encode())
    config = {"Target#render": {"Application": "path:/srv/app/**"}}

    def load_rules(field_name: str) -> RuntimeRuleSet:
        path = tmp_path / f"{field_name}.yml"
        path.write_text(
            f"""
name: demo-runtime
stdlib_path_markers:
  - /opt/runtime/stdlib/
{field_name}:
  - Delegator.
""",
            encoding="utf-8",
        )
        return load_runtime_rules_file(path)

    generic_rules = load_rules("caller_fallback_name_prefixes")
    alias_rules = load_rules("legacy_caller_fallback_name_prefixes")
    assert generic_rules.caller_fallback_name_prefixes == ("Delegator.",)
    assert alias_rules.caller_fallback_name_prefixes == ("Delegator.",)

    for rules in (generic_rules, alias_rules):
        categories = analyze_targets(
            profile,
            config,
            TargetAnalysisOptions(
                runtime_rules=rules,
                enhanced_runtime_categorization=False,
                caller_fallback_when_uncategorized=True,
            ),
        )["Target#render"]
        assert categories["Application"].cpu_time == 9_000_000


@covers("M9-006")
def test_clankerprof_operator_skill_documents_portable_workflow() -> None:
    root = Path(__file__).resolve().parents[1]
    skill_dir = root / "skills" / "clankerprof-operator"
    skill = skill_dir / "SKILL.md"
    agent = skill_dir / "agents" / "openai.yaml"

    assert skill.exists()
    assert agent.exists()
    rendered = skill.read_text(encoding="utf-8")
    for phrase in [
        "docs/CLANKERPROF_SPEC.md",
        "clankerprof.sample_facts.v2",
        "clankerprof.sample_facts.v1",
        "runtime-rules.yml",
        "caller_fallback_name_prefixes",
        "check_real_profile_parity.py",
        "Cross-Language Port Checklist",
    ]:
        assert phrase in rendered

    def term(*codes: int) -> str:
        return "".join(chr(code) for code in codes)

    for forbidden in [
        term(112, 112, 114, 111, 102, 45, 114, 101, 97, 100, 101, 114),
        term(80, 104, 97, 115, 101, 114),
        term(83, 104, 111, 112, 105, 102, 121),
        term(83, 116, 111, 114, 101, 102, 114, 111, 110, 116),
        term(76, 105, 113, 117, 105, 100),
    ]:
        assert forbidden not in rendered


@covers("M9-003")
def test_clankerprof_can_emit_legacy_target_csv_artifact_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    attributables_path = tmp_path / "attributables.json"
    profile_path.write_bytes(_ruby_profile_bytes())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": r"[/\\]app[/\\]"}}),
        encoding="utf-8",
    )
    attributables_path.write_text(
        json.dumps({"request_count": {"Target#render": 10_000}}),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime",
                "ruby",
                "--fold-ruby-internals",
                "--track-semantic-callers",
                "--semantic-callers-csv",
                "semantic-callers.csv",
                "--format",
                "csv",
                "--cpu-attributables",
                str(attributables_path),
                "--output",
                "slices.csv",
                "--legacy-target-csv-layout",
            ]
        )
        == 0
    )

    simplified_path = tmp_path / "output" / "slices.csv"
    verbose_path = tmp_path / "output" / "verbose" / "slices.csv"
    assert simplified_path.exists()
    assert verbose_path.exists()
    simplified = simplified_path.read_text(encoding="utf-8")
    verbose = verbose_path.read_text(encoding="utf-8")
    assert simplified.startswith("Parent Function,Category,CPU %,request_count")
    assert "Top 3 Callsites,Top Leaf Functions" in simplified.splitlines()[0]
    assert verbose.startswith("Parent Function,Category,CPU Time (ns),CPU Time,%")
    assert "Top Caller\u2192Leaf Pair" in verbose.splitlines()[0]
    assert " \u2192 " in verbose
    semantic = (tmp_path / "semantic-callers.csv").read_text(encoding="utf-8")
    assert "gems/statsd-instrument/lib/statsd/instrument/aggregator.rb" in semantic

    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--runtime",
                "ruby",
                "--format",
                "csv",
                "--output",
                "generic-layout.csv",
                "--target-csv-layout",
                "compat",
            ]
        )
        == 0
    )
    assert (tmp_path / "output" / "generic-layout.csv").exists()


@covers("M9-004")
def test_clankerprof_slice_analysis_supports_filters_collapse_attributes_and_compare() -> (
    None
):
    profile = decode_profile_bytes(_target_profile_bytes())
    result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            slices=(
                SliceDefinition("app", ("app/**",)),
                SliceDefinition("default", (), is_default=True),
            ),
            collapse=("library:*",),
            attributes=(
                AttributionRule(
                    "name",
                    "TemplateEngine::Native",
                    "native-template",
                ),
            ),
            filters=("<name:Target#render",),
            no_collapse_native=True,
        ),
    )
    payload = render_slice_json(result)
    slices = {
        item["name"]: item for item in cast(list[dict[str, object]], payload["slices"])
    }

    assert slices["app"]["time_ns"] == 1_000_000
    assert slices["native-template"]["time_ns"] == 3_000_000
    assert slices["default"]["time_ns"] == 4_000_000

    collapsed = analyze_slices(
        profile,
        SliceAnalysisOptions(
            slices=(
                SliceDefinition("app", ("app/**",)),
                SliceDefinition("default", (), is_default=True),
            ),
            collapse=("library:*",),
            filters=("<name:Target#render",),
        ),
    )
    collapsed_payload = render_slice_json(collapsed)
    collapsed_slices = {
        item["name"]: item
        for item in cast(list[dict[str, object]], collapsed_payload["slices"])
    }
    assert collapsed_slices["app"]["time_ns"] == 4_000_000

    before = payload
    after = json.loads(json.dumps(payload))
    after_slices = cast(list[dict[str, object]], after["slices"])
    for item in after_slices:
        if item["name"] == "app":
            item["pct"] = 30.0
    compared = compare_slice_json(
        before,
        cast(dict[str, object], after),
        CompareOptions(
            threshold_abs=2.0, threshold_rel=15.0, focus_slices=frozenset({"app"})
        ),
    )
    assert compared["has_regression"] is True
    assert cast(list[dict[str, object]], compared["slices"])[0]["name"] == "app"


@covers("M9-005", "M9-007")
def test_clankerprof_compare_supports_boundary_outputs() -> None:
    before = render_boundary_json(
        analyze_boundaries(
            decode_profile_bytes(_boundary_decomposition_profile_bytes()),
            _generic_boundary_options(),
        )
    )
    after = json.loads(json.dumps(before))
    after_boundary = cast(list[dict[str, object]], after["boundaries"])[0]
    after_buckets = cast(list[dict[str, object]], after_boundary["buckets"])
    for bucket in after_buckets:
        if bucket["name"] == "Mechanics":
            bucket["pct"] = 90.0

    compared = compare_boundary_json(
        cast(dict[str, Any], before),
        cast(dict[str, Any], after),
        CompareOptions(threshold_abs=2.0, threshold_rel=15.0),
    )
    assert compared["has_regression"] is True
    rows = cast(list[dict[str, object]], compared["rows"])
    assert rows[0]["kind"] == "bucket"
    assert rows[0]["boundary"] == "Request render"
    assert rows[0]["name"] == "Mechanics"


def _write_core_csv(tmp_path: Path) -> Path:
    path = tmp_path / "ruby_core_classes.csv"
    path.write_text("\n".join(sorted(_ruby_core_classes())) + "\n", encoding="utf-8")
    return path


@covers("M9-004")
def test_clankerprof_slice_filters_apply_to_bottom_frame_after_native_collapse() -> (
    None
):
    profile = decode_profile_bytes(_slice_semantics_profile_bytes())
    result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=("name:ComponentRenderer#render",),
            slices=(SliceDefinition("components", ("app/components/**",)),),
        ),
    )
    assert result.matching_time_ns == 10_000_000
    payload = render_slice_json(result)
    slices = cast(list[dict[str, object]], payload["slices"])
    assert slices[0]["name"] == "components"
    assert slices[0]["time_ns"] == 10_000_000


@covers("M9-004")
def test_clankerprof_slice_paths_support_legacy_regex_and_simplified_patterns() -> None:
    for raw_pattern in (
        "app/components/**",
        "path:app/components/**",
        r"regex:[/\\]app[/\\]components[/\\]",
        "library:*",
        "gem:*",
    ):
        if raw_pattern in {"gem:*", "library:*"}:
            profile = decode_profile_bytes(_slice_semantics_profile_bytes())
            filters = ("name:Instrumentation#wrap",)
            expected_time_ns = 20_000_000
        else:
            profile = decode_profile_bytes(_slice_semantics_profile_bytes())
            filters = ("name:ComponentRenderer#render",)
            expected_time_ns = 10_000_000
        result = analyze_slices(
            profile,
            SliceAnalysisOptions(
                runtime_rules=(
                    ruby_rules(_ruby_core_classes())
                    if raw_pattern == "library:*"
                    else load_runtime_rules("generic")
                ),
                filters=filters,
                slices=(SliceDefinition("components", (raw_pattern,)),),
            ),
        )
        payload = render_slice_json(result)
        slices = cast(list[dict[str, object]], payload["slices"])
        assert slices[0]["name"] == "components"
        assert slices[0]["time_ns"] == expected_time_ns


@covers("M9-003", "M9-004")
def test_clankerprof_slice_native_collapse_uses_runtime_rules() -> None:
    profile = decode_profile_bytes(_rule_driven_native_path_profile_bytes())

    generic = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=("<name:RequestHandler#render_response",),
            slices=(
                SliceDefinition("runtime", ("path:opt/runtime/ruby/4.0.0/**",)),
                SliceDefinition("components", ("app/components/**",)),
            ),
        ),
    )
    generic_payload = render_slice_json(generic)
    generic_slices = cast(list[dict[str, object]], generic_payload["slices"])
    assert generic_slices[0]["name"] == "runtime"

    ruby = analyze_slices(
        profile,
        SliceAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes()),
            filters=("<name:RequestHandler#render_response",),
            slices=(
                SliceDefinition("runtime", ("path:opt/runtime/ruby/4.0.0/**",)),
                SliceDefinition("components", ("app/components/**",)),
            ),
        ),
    )
    ruby_payload = render_slice_json(ruby)
    ruby_slices = cast(list[dict[str, object]], ruby_payload["slices"])
    assert ruby_slices[0]["name"] == "components"
    assert ruby_slices[0]["time_ns"] == 15_000_000
    assert is_runtime_stdlib_path(
        "/usr/local/lib/ruby/3.2.0/forwardable.rb",
        ruby_rules(_ruby_core_classes()),
    )


@covers("M9-004")
def test_clankerprof_slice_descendant_filters_use_or_semantics() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())

    descendant_result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=(
                "<name:RequestHandler#render_response",
                "<name:BackgroundJob#perform",
            ),
        ),
    )
    assert descendant_result.matching_time_ns == 160_000_000

    mixed_result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=(
                "path:app/components/product_card.rb",
                "<name:RequestHandler#render_response",
                "<name:BackgroundJob#perform",
            ),
        ),
    )
    assert mixed_result.matching_time_ns == 60_000_000

    inverted_descendant_result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=(
                "<!name:DoesNotExist",
                "path:app/jobs/background_job.rb",
            ),
        ),
    )
    assert inverted_descendant_result.matching_time_ns == 60_000_000


@covers("M9-004")
def test_clankerprof_slice_filter_prefixes_can_repeat_in_any_order() -> None:
    profile = decode_profile_bytes(_request_rendering_profile_bytes())

    repeated_prefix_result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=(
                "!<!<name:DoesNotExist",
                "path:app/jobs/background_job.rb",
            ),
        ),
    )
    assert repeated_prefix_result.matching_time_ns == 60_000_000


@covers("M9-004")
def test_clankerprof_slice_filter_honors_descendant_attribute_slice_matches() -> None:
    profile = decode_profile_bytes(_descendant_attribute_profile_bytes())
    result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            filters=("slice:instrumentation",),
            slices=(
                SliceDefinition("components", ("app/components/**",)),
                SliceDefinition("instrumentation", ()),
            ),
            attributes=(
                AttributionRule(
                    "name",
                    "TelemetryWrapper#call",
                    "instrumentation",
                    descendant=True,
                ),
            ),
        ),
    )
    payload = render_slice_json(result)
    assert result.matching_time_ns == 70_000_000
    assert (
        cast(list[dict[str, object]], payload["slices"])[0]["name"] == "instrumentation"
    )


@covers("M9-004")
def test_clankerprof_slice_collapse_does_not_use_descendant_attribute_rescue() -> None:
    profile = decode_profile_bytes(_descendant_attribute_profile_bytes())
    options = SliceAnalysisOptions(
        filters=("slice:instrumentation",),
        collapse=("slice:instrumentation",),
        slices=(
            SliceDefinition("components", ("app/components/**",)),
            SliceDefinition("instrumentation", ()),
        ),
        attributes=(
            AttributionRule(
                "name",
                "TelemetryWrapper#call",
                "instrumentation",
                descendant=True,
            ),
        ),
    )
    result = analyze_slices(profile, options)
    payload = render_slice_json(result, options)

    assert result.matching_time_ns == 70_000_000
    slices = cast(list[dict[str, object]], payload["slices"])
    assert slices[0]["name"] == "instrumentation"
    frames = cast(list[dict[str, object]], slices[0]["frames"])
    assert frames[0]["function"] == "CacheClient#get"


@covers("M9-004")
def test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_libraries() -> (
    None
):
    profile = decode_profile_bytes(_slice_semantics_profile_bytes())
    result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes()),
            collapse=("library:*",),
            slices=(SliceDefinition("default", (), is_default=True),),
            unattributed_libraries=1,
        ),
    )
    payload = render_slice_json(
        result,
        SliceAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes()),
            collapse=("library:*",),
            slices=(SliceDefinition("default", (), is_default=True),),
            unattributed_libraries=1,
        ),
    )

    assert cast(dict[str, object], payload["gc"])["time_ns"] == 30_000_000
    uncollapsible = cast(dict[str, object], payload["uncollapsible"])
    assert uncollapsible["name"] == "(uncollapsible)"
    assert uncollapsible["time_ns"] == 20_000_000
    default_slice = cast(list[dict[str, object]], payload["slices"])[0]
    libraries = cast(list[dict[str, object]], default_slice["unattributed_libraries"])
    assert libraries[0]["name"] == "statsd-instrument"
    gems = cast(list[dict[str, object]], default_slice["unattributed_gems"])
    assert gems[0]["name"] == "statsd-instrument"


@covers("M9-004")
def test_clankerprof_uncollapsible_reports_root_eligible_frame() -> None:
    profile = decode_profile_bytes(_uncollapsible_profile_bytes())
    options = SliceAnalysisOptions(
        collapse=("name:Collapsed", "name:Root#call"),
        top=3,
    )
    result = analyze_slices(profile, options)
    payload = render_slice_json(result, options)

    uncollapsible = cast(dict[str, object], payload["uncollapsible"])
    frames = cast(list[dict[str, object]], uncollapsible["frames"])
    assert frames[0]["function"] == "Root#call"
    assert frames[0]["time_ns"] == 40_000_000


@covers("M9-004")
def test_clankerprof_slice_cli_supports_config_and_output_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "request-rendering.pb"
    slices_path = tmp_path / "slices.yml"
    config_path = tmp_path / "clankerprof-slices.yml"
    profile_path.write_bytes(_request_rendering_profile_bytes())
    slices_path.write_text(
        """
slices:
  - name: request-core
    paths:
      - app/http/**
      - app/view_models/**
  - name: rendering
    paths:
      - app/components/**
    metadata:
      owner: rendering-platform
      docs:
        - https://example.invalid/rendering
    contacts:
      - "#rendering-performance"
  - name: rendering-native
  - name: default
    default: true
""".strip(),
        encoding="utf-8",
    )
    config_path.write_text(
        f"""
slices: {slices_path}
filters:
  - <name:RequestHandler#render_response
collapse:
  - library:*
attribute:
  - name:TemplateEngine::Native,to:rendering-native
by_slice: 2
top: 1
show_paths: true
no_collapse_native: true
""".strip(),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "slices",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    slices = cast(list[dict[str, object]], payload["slices"])
    assert [item["name"] for item in slices] == ["rendering", "rendering-native"]
    assert slices[0]["time_ns"] == 50_000_000
    assert slices[0]["metadata"] == {
        "contacts": ["#rendering-performance"],
        "docs": ["https://example.invalid/rendering"],
        "owner": "rendering-platform",
    }
    assert slices[1]["time_ns"] == 40_000_000
    assert len(cast(list[dict[str, object]], slices[0]["frames"])) == 1


@covers("M9-004")
def test_clankerprof_slice_cli_validates_attribute_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb"
    slices_path = tmp_path / "slices.yml"
    profile_path.write_bytes(_request_rendering_profile_bytes())
    slices_path.write_text(
        """
slices:
  - name: default
    default: true
""".strip(),
        encoding="utf-8",
    )

    for raw_attribute, expected in [
        ("!library:cache-client,to:default", "do not support '!'"),
        ("slice:default,to:default", "do not support slice: filters"),
    ]:
        assert (
            clankerprof_main(
                [
                    "slices",
                    "--profile",
                    str(profile_path),
                    "--slices",
                    str(slices_path),
                    "--attribute",
                    raw_attribute,
                ]
            )
            == 2
        )
        assert expected in capsys.readouterr().err

    assert (
        clankerprof_main(
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--attribute",
                "library:cache-client,to:default",
                "--attribute",
                "<library:cache-client,to:default",
            ]
        )
        == 2
    )
    assert "Duplicate attribute rule filter" in capsys.readouterr().err

    assert (
        clankerprof_main(
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--attribute",
                "library:cache-client,to:virtual-cache",
                "--filter",
                "<name:RequestHandler#render_response",
            ]
        )
        == 2
    )
    assert "Unknown slice in --attribute: virtual-cache" in capsys.readouterr().err

    assert (
        clankerprof_main(
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--attribute",
                "library:cache-client,to:virtual-cache",
                "--allow-virtual-attribute-slices",
                "--filter",
                "<name:RequestHandler#render_response",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    slices = cast(list[dict[str, object]], payload["slices"])
    assert any(item["name"] == "virtual-cache" for item in slices)


@covers("M9-004")
def test_clankerprof_slice_cli_validates_filter_and_collapse_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb"
    slices_path = tmp_path / "slices.yml"
    profile_path.write_bytes(_request_rendering_profile_bytes())
    slices_path.write_text(
        """
slices:
  - name: default
    default: true
""".strip(),
        encoding="utf-8",
    )

    for flag, raw_filter, expected in [
        ("--filter", "category:unknown", "Unsupported filter key: category"),
        ("--filter", "name", "Filter must be '<key>:<value>'"),
        ("--collapse", "!name:Wrapper", "Collapse filters do not support prefixes"),
        ("--collapse", "<name:Wrapper", "Collapse filters do not support prefixes"),
    ]:
        assert (
            clankerprof_main(
                [
                    "slices",
                    "--profile",
                    str(profile_path),
                    "--slices",
                    str(slices_path),
                    flag,
                    raw_filter,
                ]
            )
            == 2
        )
        assert expected in capsys.readouterr().err


@covers("M9-004")
def test_clankerprof_slice_cli_rejects_duplicate_scalar_config_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "clankerprof-slices.toml"
    profile_path.write_bytes(_request_rendering_profile_bytes())
    config_path.write_text(
        f'''
profile = "{profile_path}"
show_paths = true
no_collapse_native = false
unattributed_libraries = 1
'''.strip(),
        encoding="utf-8",
    )

    for flag, expected in [
        ("--show-paths", "show_paths specified both"),
        ("--no-collapse-native", "no_collapse_native specified both"),
        ("--unattributed-libraries", "unattributed_libraries specified both"),
    ]:
        args = ["slices", "--config", str(config_path), flag]
        if flag == "--unattributed-libraries":
            args.append("2")
        assert clankerprof_main(args) == 2
        assert expected in capsys.readouterr().err


@covers("M9-004")
def test_clankerprof_slice_cli_supports_toml_config_default_slices_and_optional_by_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_request_rendering_profile_bytes())
    (tmp_path / "slices.yml").write_text(
        """
slices:
  - name: components
    paths:
      - app/components/**
  - name: default
    default: true
""".strip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "clankerprof-slices.toml"
    config_path.write_text(
        f'''
profile = "{profile_path}"
filter = ["<name:RequestHandler#render_response"]
top = 1
unattributed_libraries = 1
'''.strip(),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    assert (
        clankerprof_main(
            [
                "slices",
                "--config",
                str(config_path),
                "--by-slice",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    slices = cast(list[dict[str, object]], payload["slices"])
    assert [item["name"] for item in slices] == ["components", "default"]
    default_slice = next(item for item in slices if item["name"] == "default")
    libraries = cast(list[dict[str, object]], default_slice["unattributed_libraries"])
    assert len(libraries) == 1
    assert libraries[0]["name"] == "cache-client"


@covers("M9-005", "M9-006")
def test_clankerprof_cli_and_autoclanker_alias_generate_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb.gz"
    facts_path = tmp_path / "profile-facts.json"
    alias_facts_path = tmp_path / "alias-profile-facts.json"
    config_path = tmp_path / "target_config.json"
    output_path = tmp_path / "targets.csv"
    profile_path.write_bytes(_target_profile_bytes(gzipped=True))
    facts_path.write_text(
        dumps_sample_facts(load_profile(str(profile_path)).to_sample_facts()),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {"Target#render": {"App Templates": r"[/\\]app[/\\]templates[/\\]"}}
        ),
        encoding="utf-8",
    )

    assert (
        clankerprof_main(
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--format",
                "csv",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert "App Templates" in output_path.read_text(encoding="utf-8")

    assert (
        autoclanker_main(
            [
                "pprof",
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    alias_payload = json.loads(capsys.readouterr().out)
    assert alias_payload["tool"] == "clankerprof_targets"
    assert "Target#render" in alias_payload["parents"]

    assert (
        clankerprof_main(
            [
                "targets",
                "--facts",
                str(facts_path),
                "--config",
                str(config_path),
                "--format",
                "json",
            ]
        )
        == 0
    )
    facts_target_payload = json.loads(capsys.readouterr().out)
    assert facts_target_payload == alias_payload

    assert (
        clankerprof_main(
            [
                "facts",
                "--profile",
                str(profile_path),
            ]
        )
        == 0
    )
    facts_payload = json.loads(capsys.readouterr().out)
    assert facts_payload["tool"] == "clankerprof_facts"
    assert facts_payload["schema_version"] == SAMPLE_FACTS_SCHEMA_VERSION
    assert facts_payload["summary"]["sample_count"] == 3

    assert (
        autoclanker_main(
            [
                "pprof",
                "facts",
                "--profile",
                str(profile_path),
                "--output",
                str(alias_facts_path),
            ]
        )
        == 0
    )
    alias_facts_payload = json.loads(capsys.readouterr().out)
    assert alias_facts_payload["tool"] == "clankerprof_facts"
    alias_facts_file = json.loads(alias_facts_path.read_text(encoding="utf-8"))
    assert alias_facts_file["tool"] == "clankerprof_facts"
    assert alias_facts_file["schema_version"] == SAMPLE_FACTS_SCHEMA_VERSION
    assert "samples" in alias_facts_file


@covers("M9-004", "M9-005", "M9-006")
def test_clankerprof_cli_uses_sample_facts_for_slice_projection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb"
    facts_path = tmp_path / "profile-facts.json"
    slices_path = tmp_path / "slices.yml"
    config_path = tmp_path / "clankerprof-slices.yml"
    profile_path.write_bytes(_slice_semantics_profile_bytes())
    facts_path.write_text(
        dumps_sample_facts(load_profile(str(profile_path)).to_sample_facts()),
        encoding="utf-8",
    )
    slices_path.write_text(
        """
slices:
  - name: components
    paths:
      - app/components/**
  - name: default
    default: true
""".strip(),
        encoding="utf-8",
    )

    profile_args = [
        "slices",
        "--profile",
        str(profile_path),
        "--slices",
        str(slices_path),
        "--collapse",
        "library:*",
        "--unattributed-libraries",
        "1",
        "--runtime",
        "ruby",
    ]
    assert clankerprof_main(profile_args) == 0
    profile_payload = json.loads(capsys.readouterr().out)

    assert (
        autoclanker_main(
            [
                "pprof",
                "slices",
                "--facts",
                str(facts_path),
                "--slices",
                str(slices_path),
                "--collapse",
                "library:*",
                "--unattributed-libraries",
                "1",
                "--runtime",
                "ruby",
            ]
        )
        == 0
    )
    facts_payload = json.loads(capsys.readouterr().out)
    assert facts_payload == profile_payload

    config_path.write_text(
        f"""
facts: {facts_path}
slices: {slices_path}
collapse:
  - library:*
unattributed_libraries: 1
""".strip(),
        encoding="utf-8",
    )
    assert (
        clankerprof_main(["slices", "--config", str(config_path), "--runtime", "ruby"])
        == 0
    )
    config_payload = json.loads(capsys.readouterr().out)
    assert config_payload == profile_payload


@covers("M9-005")
def test_clankerprof_compare_exits_nonzero_for_regression_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100, "matching_time_ns": 100},
        "slices": [{"name": "rendering", "pct": 10.0, "frames": []}],
    }
    after: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100, "matching_time_ns": 100},
        "slices": [{"name": "rendering", "pct": 15.0, "frames": []}],
    }
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    assert (
        clankerprof_main(
            [
                "compare",
                "--before",
                str(before_path),
                "--after",
                str(after_path),
                "--threshold-abs",
                "2",
                "--threshold-rel",
                "15",
                "--focus-slices",
                "rendering",
            ]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["has_regression"] is True


@covers("M9-005")
def test_clankerprof_loads_real_profile_file_from_disk(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_target_profile_bytes())
    profile = load_profile(str(profile_path))
    assert len(profile.functions) == 5


@covers("M9-006")
def test_clankerprof_real_profile_parity_helper_requires_opt_in(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from clankerprof.parity import main as parity_main

    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_target_profile_bytes())
    monkeypatch.delenv("CLANKERPROF_REAL_PROFILE_PARITY", raising=False)

    assert parity_main(["--profile", str(profile_path)]) == 2
    error_payload = json.loads(capsys.readouterr().err)
    assert "CLANKERPROF_REAL_PROFILE_PARITY" in error_payload["error"]


@covers("M9-006")
def test_clankerprof_real_profile_parity_helper_covers_legacy_target_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.parity import main as parity_main

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "target-config.json"
    attributables_path = tmp_path / "attributables.json"
    expected_simple_path = tmp_path / "expected-simple.csv"
    expected_verbose_path = tmp_path / "expected-verbose.csv"
    expected_semantic_path = tmp_path / "expected-semantic.csv"
    profile_path.write_bytes(_ruby_profile_bytes())
    config_path.write_text(
        json.dumps({"Target#render": {"Application": r"[/\\]app[/\\]"}}),
        encoding="utf-8",
    )
    attributables_path.write_text(
        json.dumps({"request_count": {"Target#render": 10_000}}),
        encoding="utf-8",
    )
    runtime_rules = ruby_rules(_ruby_core_classes())
    results = analyze_targets(
        decode_profile_bytes(_ruby_profile_bytes()),
        {"Target#render": {"Application": r"[/\\]app[/\\]"}},
        TargetAnalysisOptions(
            runtime_rules=runtime_rules,
            fold_runtime_internals=True,
            track_semantic_callers=True,
            attributables={"request_count": {"Target#render": 10_000}},
        ),
    )
    expected_simple_path.write_text(
        render_target_csv(
            results,
            attributables={"request_count": {"Target#render": 10_000}},
            simplified=True,
            legacy_layout=True,
        ),
        encoding="utf-8",
    )
    expected_verbose_path.write_text(
        render_target_csv(
            results,
            attributables={"request_count": {"Target#render": 10_000}},
            legacy_layout=True,
        ),
        encoding="utf-8",
    )
    expected_semantic_path.write_text(
        render_semantic_callers_csv(
            results,
            runtime_rules=runtime_rules,
            dependency_prefix="gems",
            legacy_layout=True,
        ),
        encoding="utf-8",
    )

    assert (
        parity_main(
            [
                "--allow-local-inputs",
                "--profile",
                str(profile_path),
                "--target-config",
                str(config_path),
                "--runtime",
                "ruby",
                "--core-classes",
                str(_write_core_csv(tmp_path)),
                "--fold-runtime-internals",
                "--track-semantic-callers",
                "--attributables",
                str(attributables_path),
                "--legacy-target-csv-layout",
                "--expected-target-csv",
                str(expected_simple_path),
                "--expected-verbose-target-csv",
                str(expected_verbose_path),
                "--expected-semantic-callers-csv",
                str(expected_semantic_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"checked": ["targets"], "ok": True}


@covers("M9-006", "M9-007")
def test_clankerprof_real_profile_parity_helper_covers_boundary_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.parity import main as parity_main

    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "boundaries.toml"
    expected_path = tmp_path / "expected-boundaries.json"
    profile_path.write_bytes(_boundary_decomposition_profile_bytes())
    config_path.write_text(
        """
[category]
"View Models" = "path:app/view_models/**"
"Components" = "path:app/components/**"
"Cache Client" = "library:cache-client"
"Serialization" = "name:JSON.generate"

[domain]
"Component rendering" = "category:Components"
"View models" = "category:View Models"
"Rendering fallback" = { patterns = ["path:app/rendering/**"], fallback = true }

[[boundary]]
label = "Request render"
function = "RequestHandler#render_response"

[boundary.attributables]
p90_ms = 200.0

[boundary.bucket]
"Application code" = ["View Models", "Components"]
"Mechanics" = ["Cache Client", "Serialization"]
""".strip(),
        encoding="utf-8",
    )
    expected = render_boundary_json(
        analyze_boundaries(
            decode_profile_bytes(_boundary_decomposition_profile_bytes()),
            BoundaryAnalysisOptions(
                categories=(
                    BoundaryCategoryDefinition(
                        "View Models",
                        parse_frame_predicates(("path:app/view_models/**",)),
                    ),
                    BoundaryCategoryDefinition(
                        "Components",
                        parse_frame_predicates(("path:app/components/**",)),
                    ),
                    BoundaryCategoryDefinition(
                        "Cache Client",
                        parse_frame_predicates(("library:cache-client",)),
                    ),
                    BoundaryCategoryDefinition(
                        "Serialization",
                        parse_frame_predicates(("name:JSON.generate",)),
                    ),
                ),
                domains=(
                    BoundaryDomainDefinition(
                        "Component rendering",
                        parse_frame_predicates(("category:Components",)),
                    ),
                    BoundaryDomainDefinition(
                        "View models",
                        parse_frame_predicates(("category:View Models",)),
                    ),
                    BoundaryDomainDefinition(
                        "Rendering fallback",
                        parse_frame_predicates(("path:app/rendering/**",)),
                        fallback=True,
                    ),
                ),
                boundaries=(
                    BoundaryDefinition(
                        name="Request render",
                        predicates=parse_frame_predicates(
                            ("name_eq:RequestHandler#render_response",),
                            default_key="name_eq",
                        ),
                        buckets={
                            "Application code": ("View Models", "Components"),
                            "Mechanics": ("Cache Client", "Serialization"),
                        },
                        attributables={"p90_ms": 200.0},
                    ),
                ),
            ),
        )
    )
    expected_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    assert (
        parity_main(
            [
                "--allow-local-inputs",
                "--profile",
                str(profile_path),
                "--boundary-config",
                str(config_path),
                "--expected-boundary-json",
                str(expected_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"checked": ["boundaries"], "ok": True}

    assert (
        parity_main(
            [
                "--allow-local-inputs",
                "--profile",
                str(profile_path),
                "--scope-config",
                str(config_path),
                "--expected-boundary-json",
                str(expected_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"checked": ["boundaries"], "ok": True}
