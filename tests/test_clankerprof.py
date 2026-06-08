from __future__ import annotations

import json

from pathlib import Path
from typing import Any, cast

import pytest

from autoclanker.cli import main as autoclanker_main
from clankerprof.analysis import (
    AttributionRule,
    SliceAnalysisOptions,
    SliceDefinition,
    TargetAnalysisOptions,
    analyze_slices,
    analyze_targets,
    categorize_ruby_frame,
    load_default_ruby_core_classes,
    ruby_rules,
)
from clankerprof.cli import main as clankerprof_main
from clankerprof.compare import CompareOptions, compare_slice_json
from clankerprof.model import Frame
from clankerprof.proto import decode_profile_bytes, load_profile
from clankerprof.render import (
    render_semantic_callers_csv,
    render_slice_json,
    render_target_csv,
    render_target_json,
    render_target_text,
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
            "Responders::BaseHtmlResponder#render_template",
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
            "Responders::BaseHtmlResponder#render_template",
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
        builder.function("CatalogView#build", "/app/view_models/catalog_view.rb")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/app/components/product_card.rb")
    )
    cache = builder.location(
        builder.function("CacheClient#get_multi", "/gems/cache-client/lib/client.rb")
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
        builder.function("CacheClient#get", "/gems/cache-client/lib/client.rb")
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
                "Cache Client": r"[/\\]gems[/\\]cache-client[/\\]",
            }
        },
    )["RequestHandler#render_response"]

    assert results["View Model"].cpu_time == 10_000_000
    assert results["Components"].cpu_time == 20_000_000
    assert results["Cache Client"].cpu_time == 30_000_000
    assert results["Other"].cpu_time == 40_000_000
    assert sum(item.cpu_time for item in results.values()) == 100_000_000


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
        {"Responders::BaseHtmlResponder#render_template": {}},
        TargetAnalysisOptions(
            runtime_rules=ruby_rules(_ruby_core_classes(), verbose=verbose),
            fold_runtime_internals=fold,
        ),
    )["Responders::BaseHtmlResponder#render_template"]

    assert sum(item.cpu_time for item in categories.values()) == 9_400_000_000
    assert present <= set(categories)
    assert absent.isdisjoint(categories)


@covers("M9-003")
def test_clankerprof_preserves_main_simplified_category_totals_and_never_folds() -> (
    None
):
    profile = decode_profile_bytes(_main_category_profile_bytes())
    config = {
        "Responders::BaseHtmlResponder#render_template": {
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
        )["Responders::BaseHtmlResponder#render_template"]
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
    rendered = render_semantic_callers_csv(results)
    assert (
        "Parent Function,Category,Leaf Function,Leaf Samples,Top Caller,"
        "Caller Samples,Caller File"
    ) in rendered
    assert "Kernel#clone" in rendered
    assert "StatsD::Instrument::Aggregator#increment" in rendered

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
    assert "Top Caller->Leaf Pair" in verbose.splitlines()[0]


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
            collapse=("gem:*",),
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
            collapse=("gem:*",),
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
def test_clankerprof_slice_outputs_gc_uncollapsible_and_unattributed_gems() -> None:
    profile = decode_profile_bytes(_slice_semantics_profile_bytes())
    result = analyze_slices(
        profile,
        SliceAnalysisOptions(
            collapse=("gem:*",),
            slices=(SliceDefinition("default", (), is_default=True),),
            unattributed_gems=1,
        ),
    )
    payload = render_slice_json(
        result,
        SliceAnalysisOptions(
            collapse=("gem:*",),
            slices=(SliceDefinition("default", (), is_default=True),),
            unattributed_gems=1,
        ),
    )

    assert cast(dict[str, object], payload["gc"])["time_ns"] == 30_000_000
    uncollapsible = cast(dict[str, object], payload["uncollapsible"])
    assert uncollapsible["name"] == "(uncollapsible)"
    assert uncollapsible["time_ns"] == 20_000_000
    default_slice = cast(list[dict[str, object]], payload["slices"])[0]
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
  - gem:*
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
        ("!gem:cache-client,to:default", "do not support '!'"),
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
                "gem:cache-client,to:default",
                "--attribute",
                "<gem:cache-client,to:default",
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
                "gem:cache-client,to:virtual-cache",
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
                "gem:cache-client,to:virtual-cache",
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
unattributed_gems = 1
'''.strip(),
        encoding="utf-8",
    )

    for flag, expected in [
        ("--show-paths", "show_paths specified both"),
        ("--no-collapse-native", "no_collapse_native specified both"),
        ("--unattributed-gems", "unattributed_gems specified both"),
    ]:
        args = ["slices", "--config", str(config_path), flag]
        if flag == "--unattributed-gems":
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
unattributed_gems = 1
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
    gems = cast(list[dict[str, object]], default_slice["unattributed_gems"])
    assert len(gems) == 1
    assert gems[0]["name"] == "cache-client"


@covers("M9-005")
def test_clankerprof_cli_and_autoclanker_alias_generate_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile_path = tmp_path / "profile.pb.gz"
    config_path = tmp_path / "target_config.json"
    output_path = tmp_path / "targets.csv"
    profile_path.write_bytes(_target_profile_bytes(gzipped=True))
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


@covers("M9-005")
def test_clankerprof_compare_exits_nonzero_for_regression_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before: dict[str, Any] = {
        "summary": {"total_time_ns": 100, "matching_time_ns": 100},
        "slices": [{"name": "rendering", "pct": 10.0, "frames": []}],
    }
    after: dict[str, Any] = {
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
