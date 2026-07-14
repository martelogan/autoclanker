from __future__ import annotations

import json
import shutil
import subprocess

from pathlib import Path
from typing import Any

import pytest

from clankerprof.analysis import (
    SliceAnalysisOptions,
    SliceDefinition,
    analyze_slices,
    analyze_targets,
)
from clankerprof.compare import (
    CompareOptions,
    compare_boundary_json,
    compare_slice_json,
)
from clankerprof.facts import sample_facts_to_jsonable
from clankerprof.proto import decode_profile_bytes
from clankerprof.render import render_slice_json, render_target_json
from tests.fixtures.pprof_builder import PprofFixtureBuilder


def _fixture_matrix() -> dict[str, bytes]:
    return {
        "raw": _profile_bytes(gzipped=False),
        "gzipped": _profile_bytes(gzipped=True),
        "inline": _inline_profile_bytes(),
        "folded": _folded_profile_bytes(),
        "sparse": _sparse_profile_bytes(),
        "multi_value": _multi_value_profile_bytes(),
        "multi_value_default": _multi_value_profile_bytes(
            default_sample_type="samples"
        ),
        "packed": _multi_value_profile_bytes(packed_samples=True),
    }


def _profile_bytes(*, gzipped: bool) -> bytes:
    builder = PprofFixtureBuilder.create()
    request = builder.location(
        builder.function("RequestHandler#render_response", "/srv/app/request.py")
    )
    component = builder.location(
        builder.function("ComponentRenderer#render", "/srv/app/components/card.py")
    )
    cache = builder.location(
        builder.function("CacheClient#get_multi", "/srv/vendor/cache-client/lib.py")
    )
    builder.sample((component, request), 10_000_000)
    builder.sample((cache, component, request), 20_000_000)
    return builder.encode(gzipped=gzipped)


def _inline_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    leaf = builder.function("InlineLeaf#work", "/srv/app/inline_leaf.py")
    target = builder.function("Target#render", "/srv/app/target.py")
    inline_location = builder.inline_location((leaf, target))
    builder.sample((inline_location,), 9_000_000)
    return builder.encode()


def _folded_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    folded_leaf = builder.folded_location(
        builder.function("FoldedLeaf#work", "/srv/app/folded_leaf.py")
    )
    builder.sample((folded_leaf, target), 6_000_000)
    return builder.encode()


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


def _sparse_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.function("Target#render", "/srv/app/target.py")
    leaf = builder.function("Leaf#work", "/srv/app/leaf.py")
    target_entry = builder.functions[target - 1]
    leaf_entry = builder.functions[leaf - 1]
    builder.functions[target - 1] = (100, target_entry[1], target_entry[2])
    builder.functions[leaf - 1] = (250, leaf_entry[1], leaf_entry[2])
    leaf_location = builder.location(250)
    target_location = builder.location(100)
    builder.sample((leaf_location, target_location), 13_000_000)
    return builder.encode()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_facts_match_python_sample_facts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for name, profile_bytes in _fixture_matrix().items():
        profile_path = tmp_path / f"{name}.pb"
        profile_path.write_bytes(profile_bytes)

        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "clankerprof-core",
                "--bin",
                "clankerprof-rs",
                "--",
                "facts",
                "--profile",
                str(profile_path),
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        rust_payload = json.loads(completed.stdout)
        python_payload: dict[str, Any] = sample_facts_to_jsonable(
            decode_profile_bytes(profile_bytes).to_sample_facts()
        )
        assert rust_payload == python_payload


def _recursive_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    helper = builder.location(builder.function("Helper#step", "/srv/app/helper.py"))
    leaf = builder.location(builder.function("Leaf#work", "/srv/app/leaf.py"))
    builder.sample((leaf, target, helper, target), 8_000_000)
    builder.sample((leaf, target), 2_000_000)
    return builder.encode()


def _precedence_ties_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    zebra = builder.location(builder.function("ZebraWork#run", "/srv/app/zone/z.py"))
    alpha = builder.location(builder.function("AlphaWork#run", "/srv/app/azone/a.py"))
    shared = builder.location(builder.function("BothWork#run", "/srv/app/shared/b.py"))
    builder.sample((zebra, target), 5_000_000)
    builder.sample((alpha, target), 5_000_000)
    builder.sample((shared, target), 2_000_000)
    return builder.encode()


_PRECEDENCE_TIES_CONFIG = {
    "Target#render": {
        "Zebra": "path:srv/app/zone",
        "Alpha": "path:srv/app/azone",
        "Shared Z": "path:srv/app/shared",
        "Shared A": "path:srv/app/shared",
    }
}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_targets_match_python_generic_projection(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases: dict[str, tuple[bytes, dict[str, dict[str, str]]]] = {
        "standard": (
            _profile_bytes(gzipped=False),
            {
                "RequestHandler#render_response": {
                    "Components": "path:srv/app/components",
                    "Cache Client": "library:cache-client",
                }
            },
        ),
        "recursive": (
            _recursive_profile_bytes(),
            {"Target#render": {"App": "path:srv/app"}},
        ),
        "precedence_ties": (
            _precedence_ties_profile_bytes(),
            _PRECEDENCE_TIES_CONFIG,
        ),
    }
    for name, (profile_bytes, config) in cases.items():
        profile_path = tmp_path / f"{name}.pb"
        config_path = tmp_path / f"{name}-targets.json"
        profile_path.write_bytes(profile_bytes)
        config_path.write_text(json.dumps(config), encoding="utf-8")

        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "clankerprof-core",
                "--bin",
                "clankerprof-rs",
                "--",
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        rust_payload = json.loads(completed.stdout)
        python_payload = render_target_json(
            analyze_targets(decode_profile_bytes(profile_bytes), config)
        )
        assert rust_payload == python_payload, f"targets parity diverged for {name}"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_slices_match_python_generic_projection(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile_bytes = _profile_bytes(gzipped=False)
    profile_path = tmp_path / "profile.pb"
    slices_path = tmp_path / "slices.yml"
    profile_path.write_bytes(profile_bytes)
    slices_path.write_text(
        """
slices:
  - name: components
    paths:
      - srv/app/components
  - name: default
    default: true
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "slices",
            "--profile",
            str(profile_path),
            "--slices",
            str(slices_path),
            "--unattributed-libraries",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    options = SliceAnalysisOptions(
        slices=(
            SliceDefinition("components", ("srv/app/components",)),
            SliceDefinition("default", is_default=True),
        ),
        unattributed_libraries=2**63 - 1,
    )
    rust_payload = json.loads(completed.stdout)
    python_payload = render_slice_json(
        analyze_slices(decode_profile_bytes(profile_bytes), options), options
    )
    assert rust_payload == python_payload


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_matches_python_slice_compare(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    before = {
        "tool": "clankerprof_slices",
        "summary": {
            "matching_time_ns": 100,
            "total_time_ns": 100,
            "matching_pct": 100.0,
        },
        "slices": [
            {
                "name": "rendering",
                "time_ns": 40,
                "pct": 40.0,
                "is_default": False,
                "frames": [
                    {
                        "function": "Renderer#call",
                        "filename": "/srv/app/render.py",
                        "line": 1,
                        "time_ns": 40,
                        "pct": 40.0,
                    }
                ],
                "unattributed_gems": [],
                "unattributed_libraries": [],
            },
            {
                "name": "default",
                "time_ns": 60,
                "pct": 60.0,
                "is_default": True,
                "frames": [
                    {
                        "function": "CacheClient#get",
                        "filename": "/srv/vendor/cache-client/lib.py",
                        "line": 1,
                        "time_ns": 60,
                        "pct": 60.0,
                    }
                ],
                "unattributed_gems": [],
                "unattributed_libraries": [],
            },
        ],
    }
    after = {
        "tool": "clankerprof_slices",
        "summary": {
            "matching_time_ns": 100,
            "total_time_ns": 100,
            "matching_pct": 100.0,
        },
        "slices": [
            {
                "name": "rendering",
                "time_ns": 45,
                "pct": 45.0,
                "is_default": False,
                "frames": [
                    {
                        "function": "Renderer#call",
                        "filename": "/srv/app/render.py",
                        "line": 1,
                        "time_ns": 45,
                        "pct": 45.0,
                    }
                ],
                "unattributed_gems": [],
                "unattributed_libraries": [],
            },
            {
                "name": "default",
                "time_ns": 55,
                "pct": 55.0,
                "is_default": True,
                "frames": [
                    {
                        "function": "CacheClient#get",
                        "filename": "/srv/vendor/cache-client/lib.py",
                        "line": 1,
                        "time_ns": 55,
                        "pct": 55.0,
                    }
                ],
                "unattributed_gems": [],
                "unattributed_libraries": [],
            },
        ],
    }
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--threshold-abs",
            "10",
            "--threshold-rel",
            "1000",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rust_payload = json.loads(completed.stdout)
    python_payload = compare_slice_json(
        before,
        after,
        CompareOptions(threshold_abs=10.0, threshold_rel=1000.0),
    )
    assert rust_payload == python_payload


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_matches_python_for_new_and_removed_rows(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    before: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"matching_time_ns": 100, "total_time_ns": 100},
        "slices": [{"name": "removed", "pct": 30.0, "frames": []}],
    }
    after: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"matching_time_ns": 100, "total_time_ns": 100},
        "slices": [{"name": "added", "pct": 30.0, "frames": []}],
    }
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--threshold-abs",
            "1000",
            "--threshold-rel",
            "1000",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Infinity" not in completed.stdout
    rust_payload = json.loads(completed.stdout)
    python_payload = compare_slice_json(
        before,
        after,
        CompareOptions(threshold_abs=1000.0, threshold_rel=1000.0),
    )
    assert rust_payload == python_payload
    added_row = next(row for row in rust_payload["slices"] if row["name"] == "added")
    assert added_row["delta_rel"] is None


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_matches_python_boundary_compare(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    before: dict[str, Any] = {
        "tool": "clankerprof_boundaries",
        "summary": {"total_time_ns": 1_000},
        "boundaries": [
            {
                "name": "render",
                "pct_of_profile": 40.0,
                "buckets": [
                    {
                        "name": "components",
                        "pct": 25.0,
                        "categories": [{"name": "Application", "pct": 20.0}],
                    }
                ],
                "domains": [{"name": "cards", "pct": 15.0}],
            }
        ],
    }
    after: dict[str, Any] = {
        "tool": "clankerprof_boundaries",
        "summary": {"total_time_ns": 1_000},
        "boundaries": [
            {
                "name": "render",
                "pct_of_profile": 9.0,
                "buckets": [
                    {
                        "name": "components",
                        "pct": 24.5,
                        "categories": [{"name": "Application", "pct": 20.0}],
                    }
                ],
                "domains": [
                    {"name": "cards", "pct": 15.0},
                    {"name": "banners", "pct": 3.0},
                ],
            }
        ],
    }
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--threshold-abs",
            "1000",
            "--threshold-rel",
            "1000",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rust_payload = json.loads(completed.stdout)
    python_payload = compare_boundary_json(
        before,
        after,
        CompareOptions(threshold_abs=1000.0, threshold_rel=1000.0),
    )
    assert rust_payload == python_payload
    improvements = rust_payload["top_improvements"]
    assert improvements[0]["name"] == "render"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_cli_rejects_malformed_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile_path = tmp_path / "profile.pb"
    profile_path.write_bytes(_profile_bytes(gzipped=False))
    cases: list[list[str]] = [
        ["targets", "--profile", str(profile_path), "--bogus-flag"],
        ["targets", "--profile"],
        ["slices", "--profile", str(profile_path), "--top", "not-an-int"],
        [
            "compare",
            "--before",
            str(profile_path),
            "--after",
            str(profile_path),
            "--threshold-abs",
            "not-a-number",
        ],
        ["unknown-subcommand"],
    ]
    for argv in cases:
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "clankerprof-core",
                "--bin",
                "clankerprof-rs",
                "--",
                *argv,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2, argv
        assert completed.stdout == "", argv


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_decoder_rejects_malformed_profiles(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cases = {
        "overlong-varint.pb": bytes([0x48]) + b"\xff" * 10 + b"\x01",
        "truncated-fixed64.pb": bytes([0x79, 0x01, 0x02]),
        "truncated-fixed32.pb": bytes([0x7D, 0x01]),
    }
    for name, data in cases.items():
        profile_path = tmp_path / name
        profile_path.write_bytes(data)
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "clankerprof-core",
                "--bin",
                "clankerprof-rs",
                "--",
                "facts",
                "--profile",
                str(profile_path),
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2, name
        assert completed.stdout == "", name


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_rejects_mismatched_tools(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps({"tool": "clankerprof_slices", "slices": []}),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps({"tool": "clankerprof_facts"}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "same clankerprof projection" in completed.stderr

    after_path.write_text(
        json.dumps({"tool": "clankerprof_facts"}),
        encoding="utf-8",
    )
    before_path.write_text(
        json.dumps({"tool": "clankerprof_facts"}),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "clankerprof_slices or clankerprof_boundaries" in completed.stderr


def _ruby_runtime_profile_bytes() -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(
        builder.function(
            "Example::HtmlResponder#render_template",
            "/app/responders/base_html_responder.rb",
        )
    )
    marshal_load = builder.location(builder.function("Marshal.load", "/app/file.rb"))
    io_read = builder.location(builder.function("IO.read", "<cfunc>"))
    trilogy = builder.location(builder.function("Trilogy#query", "<cfunc>"))
    statsd = builder.location(builder.function("StatsD.increment", "<cfunc>"))
    otel = builder.location(
        builder.function(
            "OpenTelemetry::Trace#span", "/gems/opentelemetry/lib/trace.rb"
        )
    )
    i18n = builder.location(
        builder.function("I18n.translate", "/gems/i18n/lib/i18n.rb")
    )
    array_map = builder.location(builder.function("Array#map", "<cfunc>"))
    zlib_bare = builder.location(builder.function("Zlib.inflate", "<cfunc>"))
    app_leaf = builder.location(
        builder.function("Card#render", "/app/components/card.rb")
    )

    builder.sample((marshal_load, target), 2_000_000_000)
    builder.sample((io_read, target), 1_500_000_000)
    builder.sample((trilogy, target), 1_000_000_000)
    builder.sample((statsd, target), 800_000_000)
    builder.sample((otel, target), 700_000_000)
    builder.sample((i18n, target), 800_000_000)
    builder.sample((array_map, app_leaf, target), 600_000_000)
    builder.sample((zlib_bare, target), 400_000_000)
    builder.sample((app_leaf, target), 900_000_000)
    return builder.encode()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_targets_match_python_ruby_runtime(tmp_path: Path) -> None:
    from clankerprof.categorize import load_default_ruby_core_classes, ruby_rules
    from clankerprof.targets import TargetAnalysisOptions, analyze_target_facts

    repo_root = Path(__file__).resolve().parents[1]
    profile_bytes = _ruby_runtime_profile_bytes()
    profile_path = tmp_path / "ruby.pb"
    config_path = tmp_path / "targets.json"
    profile_path.write_bytes(profile_bytes)
    config = {
        "Example::HtmlResponder#render_template": {
            "Components": "path:app/components",
            "Application": "path:app/**",
        }
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")

    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--runtime",
            "ruby",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    rust_payload = json.loads(completed.stdout)
    python_payload = render_target_json(
        analyze_target_facts(
            decode_profile_bytes(profile_bytes).to_sample_facts(),
            config,
            TargetAnalysisOptions(
                runtime_rules=ruby_rules(load_default_ruby_core_classes())
            ),
        )
    )
    assert rust_payload == python_payload
    categories = {
        item["name"]
        for item in rust_payload["parents"]["Example::HtmlResponder#render_template"][
            "categories"
        ]
    }
    assert "Serialization Overhead" in categories
    assert "Instrumentation Overhead" in categories


def _scopes_decomposition_profile_bytes() -> bytes:
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
    builder.sample((view_model, template, request, middleware), 10_000_000)
    builder.sample((component, template, request, middleware), 20_000_000)
    builder.sample((cache, component, template, request, middleware), 30_000_000)
    builder.sample(
        (json_generate, view_model, template, request, middleware),
        40_000_000,
    )
    builder.sample((worker,), 50_000_000)
    # recursion into the boundary for once_per_sample coverage
    builder.sample((component, request, template, request, middleware), 12_000_000)
    return builder.encode()


_SCOPES_PREFERRED_CONFIG = """
[cost_kind]
"View Models" = "path:app/view_models/**"
"Components" = "path:app/components/**"
"Cache" = "library:cache-client"

[owner]
"Rendering" = "path:app/rendering/**"
"HTTP" = { match = "path:app/http/**", fallback = true }

[[scope]]
label = "Request render"
match = "name_eq:RequestHandler#render_response"

[scope.rollup]
"Application code" = ["View Models", "Components"]
"Dependencies" = ["Cache"]

[scope.attributables]
db_queries = 12.0

[[scope]]
label = "Request once"
match = "name_eq:RequestHandler#render_response"
count = "once_per_sample"

[[scope]]
label = "Render outside components"
match = "name_eq:TemplateRenderer#render"
exclude_descendants = "name_eq:ComponentRenderer#render"
"""

_SCOPES_LEGACY_CONFIG = (
    _SCOPES_PREFERRED_CONFIG.replace("[cost_kind]", "[category]")
    .replace("[owner]", "[domain]")
    .replace("[[scope]]", "[[boundary]]")
    .replace("[scope.rollup]", "[boundary.bucket]")
    .replace("[scope.attributables]", "[boundary.attributables]")
)


def _python_scopes_payload(
    profile_bytes: bytes, config_text: str, tmp_path: Path
) -> Any:
    import clankerprof.cli as clankerprof_cli_module

    from clankerprof.render import render_boundary_json
    from clankerprof.scopes import analyze_boundary_facts

    config_path = tmp_path / "scopes-ref.toml"
    config_path.write_text(config_text, encoding="utf-8")
    options = clankerprof_cli_module._load_boundary_options(  # pyright: ignore[reportPrivateUsage]
        config_path,
        runtime_rules=clankerprof_cli_module.DEFAULT_RUNTIME_RULES,
    )
    return render_boundary_json(
        analyze_boundary_facts(
            decode_profile_bytes(profile_bytes).to_sample_facts(), options
        ),
        top=None,
    )


def _run_rust_scopes(
    tmp_path: Path,
    profile_args: list[str],
    config_text: str,
    subcommand: str = "scopes",
) -> Any:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / f"scopes-{subcommand}.toml"
    config_path.write_text(config_text, encoding="utf-8")
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            subcommand,
            *profile_args,
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_scopes_match_python_boundary_decomposition(
    tmp_path: Path,
) -> None:
    profile_bytes = _scopes_decomposition_profile_bytes()
    profile_path = tmp_path / "scopes.pb"
    profile_path.write_bytes(profile_bytes)

    python_payload = _python_scopes_payload(
        profile_bytes, _SCOPES_PREFERRED_CONFIG, tmp_path
    )
    rust_payload = _run_rust_scopes(
        tmp_path, ["--profile", str(profile_path)], _SCOPES_PREFERRED_CONFIG
    )
    assert rust_payload == python_payload

    boundaries = {item["name"]: item for item in rust_payload["boundaries"]}
    assert boundaries["Request render"]["total_time_ns"] == 112_000_000 + 12_000_000
    assert (
        boundaries["Request once"]["total_time_ns"]
        < (boundaries["Request render"]["total_time_ns"])
    )
    assert (
        boundaries["Render outside components"]["total_time_ns"]
        < boundaries["Request render"]["total_time_ns"]
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_scopes_legacy_aliases_and_boundaries_subcommand(
    tmp_path: Path,
) -> None:
    profile_bytes = _scopes_decomposition_profile_bytes()
    profile_path = tmp_path / "scopes.pb"
    profile_path.write_bytes(profile_bytes)

    python_payload = _python_scopes_payload(
        profile_bytes, _SCOPES_LEGACY_CONFIG, tmp_path
    )
    rust_payload = _run_rust_scopes(
        tmp_path,
        ["--profile", str(profile_path)],
        _SCOPES_LEGACY_CONFIG,
        subcommand="boundaries",
    )
    assert rust_payload == python_payload


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_scopes_replay_facts_identically(tmp_path: Path) -> None:
    from clankerprof.facts import dumps_sample_facts

    profile_bytes = _scopes_decomposition_profile_bytes()
    profile_path = tmp_path / "scopes.pb"
    profile_path.write_bytes(profile_bytes)
    facts_path = tmp_path / "scopes-facts.json"
    facts_path.write_text(
        dumps_sample_facts(decode_profile_bytes(profile_bytes).to_sample_facts())
        + "\n",
        encoding="utf-8",
    )

    from_profile = _run_rust_scopes(
        tmp_path, ["--profile", str(profile_path)], _SCOPES_PREFERRED_CONFIG
    )
    from_facts = _run_rust_scopes(
        tmp_path, ["--facts", str(facts_path)], _SCOPES_PREFERRED_CONFIG
    )
    assert from_facts == from_profile
    python_payload = _python_scopes_payload(
        profile_bytes, _SCOPES_PREFERRED_CONFIG, tmp_path
    )
    assert from_facts == python_payload


def _run_python_cli(argv: list[str]) -> None:
    from clankerprof.cli import main as clankerprof_main

    assert clankerprof_main(argv) == 0, argv


def _run_rust_cli(argv: list[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "clankerprof-core",
            "--bin",
            "clankerprof-rs",
            "--",
            *argv,
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (argv, completed.stderr)


def _flag_matrix_cases(tmp_path: Path) -> list[tuple[str, list[str], str]]:
    """(case name, shared argv after subcommand, compare mode)."""
    profile_path = tmp_path / "matrix.pb"
    profile_path.write_bytes(_ruby_runtime_profile_bytes())
    config_path = tmp_path / "matrix-targets.json"
    config_path.write_text(
        json.dumps(
            {
                "Example::HtmlResponder#render_template": {
                    "Components": "path:app/components",
                    "Application": "path:app/**",
                }
            }
        ),
        encoding="utf-8",
    )
    attributables_path = tmp_path / "attributables.json"
    attributables_path.write_text(
        json.dumps(
            {"db_queries": {"Example::HtmlResponder#render_template": 40.0}}
        ),
        encoding="utf-8",
    )
    slices_path = tmp_path / "matrix-slices.yml"
    slices_path.write_text(
        """
slices:
  - name: components
    paths:
      - app/components
    owner: web-platform
  - name: default
    default: true
""",
        encoding="utf-8",
    )
    gc_profile_path = tmp_path / "gc.pb"
    gc_builder = PprofFixtureBuilder.create()
    gc_target = gc_builder.location(
        gc_builder.function("Target#render", "/app/target.rb")
    )
    gc_leaf = gc_builder.location(gc_builder.function("(marking)", "<gc>"))
    app_leaf = gc_builder.location(gc_builder.function("Card#render", "/app/card.rb"))
    gc_builder.sample((gc_leaf, gc_target), 4_000_000)
    gc_builder.sample((app_leaf, gc_target), 6_000_000)
    gc_profile_path.write_bytes(gc_builder.encode())

    target_base = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
        "--runtime",
        "ruby",
    ]
    return [
        (
            "targets-fold-json",
            [*target_base, "--fold-runtime-internals"],
            "json",
        ),
        (
            "targets-fold-track-csv",
            [
                *target_base,
                "--fold-runtime-internals",
                "--track-semantic-callers",
                "--format",
                "csv",
            ],
            "bytes",
        ),
        (
            "targets-simple-csv-attributables",
            [
                *target_base,
                "--cpu-attributables",
                str(attributables_path),
                "--format",
                "simple-csv",
            ],
            "bytes",
        ),
        (
            "targets-text-fold-track",
            [
                *target_base,
                "--fold-runtime-internals",
                "--track-semantic-callers",
                "--format",
                "text",
            ],
            "bytes",
        ),
        (
            "targets-no-enhanced",
            [*target_base, "--no-enhanced"],
            "json",
        ),
        (
            "targets-minimal-target-mode",
            [
                "targets",
                "--profile",
                str(profile_path),
                "--target",
                "Example::HtmlResponder#render_template",
                "--runtime",
                "ruby",
            ],
            "json",
        ),
        (
            "slices-attribute-metadata-byslice",
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--attribute",
                "name:StatsD,to:components",
                "--by-slice",
                "5",
                "--unattributed-libraries",
                "1",
                "--runtime",
                "ruby",
            ],
            "json",
        ),
        (
            "slices-gc-pseudo",
            [
                "slices",
                "--profile",
                str(gc_profile_path),
                "--slices",
                str(slices_path),
            ],
            "json",
        ),
    ]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_cli_flag_matrix_matches_python(tmp_path: Path) -> None:
    for name, argv, mode in _flag_matrix_cases(tmp_path):
        python_out = tmp_path / f"{name}-python.out"
        rust_out = tmp_path / f"{name}-rust.out"
        _run_python_cli([*argv, "--output", str(python_out)])
        _run_rust_cli([*argv, "--output", str(rust_out)])
        python_text = python_out.read_text(encoding="utf-8")
        rust_text = rust_out.read_text(encoding="utf-8")
        if mode == "json":
            assert json.loads(rust_text) == json.loads(python_text), name
        else:
            assert rust_text == python_text, name


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_semantic_callers_csv_matches_python(tmp_path: Path) -> None:
    profile_path = tmp_path / "sem.pb"
    profile_path.write_bytes(_ruby_runtime_profile_bytes())
    config_path = tmp_path / "sem-targets.json"
    config_path.write_text(
        json.dumps({"Example::HtmlResponder#render_template": {"App": "path:app/**"}}),
        encoding="utf-8",
    )
    base = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
        "--runtime",
        "ruby",
        "--track-semantic-callers",
    ]
    python_csv = tmp_path / "sem-python.csv"
    rust_csv = tmp_path / "sem-rust.csv"
    python_json = tmp_path / "sem-python.json"
    rust_json = tmp_path / "sem-rust.json"
    _run_python_cli(
        [*base, "--semantic-callers-csv", str(python_csv), "--output", str(python_json)]
    )
    _run_rust_cli(
        [*base, "--semantic-callers-csv", str(rust_csv), "--output", str(rust_json)]
    )
    assert rust_csv.read_text(encoding="utf-8") == python_csv.read_text(
        encoding="utf-8"
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compat_csv_artifacts_match_python(tmp_path: Path) -> None:
    import os

    profile_path = tmp_path / "compat.pb"
    profile_path.write_bytes(_ruby_runtime_profile_bytes())
    config_path = tmp_path / "compat-targets.json"
    config_path.write_text(
        json.dumps({"Example::HtmlResponder#render_template": {"App": "path:app/**"}}),
        encoding="utf-8",
    )
    base = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
        "--runtime",
        "ruby",
        "--format",
        "csv",
        "--target-csv-layout",
        "compat",
    ]
    previous_cwd = Path.cwd()
    python_dir = tmp_path / "python-compat"
    rust_dir = tmp_path / "rust-compat"
    python_dir.mkdir()
    rust_dir.mkdir()
    try:
        os.chdir(python_dir)
        _run_python_cli([*base, "--output", "report.csv"])
    finally:
        os.chdir(previous_cwd)

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(repo_root / "crates" / "clankerprof-core" / "Cargo.toml"),
            "--bin",
            "clankerprof-rs",
            "--",
            *base,
            "--output",
            "report.csv",
        ],
        cwd=rust_dir,
        env={**os.environ, "CARGO_TARGET_DIR": str(repo_root / "target")},
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0

    for relative in ("output/report.csv", "output/verbose/report.csv"):
        python_text = (python_dir / relative).read_text(encoding="utf-8")
        rust_text = (rust_dir / relative).read_text(encoding="utf-8")
        assert rust_text == python_text, relative
