from __future__ import annotations

import gzip
import json
import re
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Any, cast

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
from clankerprof.render import (
    render_json_payload,
    render_slice_json,
    render_target_json,
)
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
        envelope = json.loads(completed.stderr)
        assert envelope["ok"] is False, argv
        assert envelope["error"], argv


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_output_receipts_and_usage_envelopes_match_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    repo_root = Path(__file__).resolve().parents[1]
    profile_path = tmp_path / "receipts.pb"
    profile_path.write_bytes(_profile_bytes(gzipped=False))
    config_path = tmp_path / "targets.json"
    config_path.write_text(
        json.dumps({"Target#render": {"App": "path:app/**"}}),
        encoding="utf-8",
    )

    def run_rust(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
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

    # facts stdout carries the artifact bytes: compact default, --pretty opt-in.
    for extra in ([], ["--pretty"]):
        assert clankerprof_main(["facts", "--profile", str(profile_path), *extra]) == 0
        python_stdout = capsys.readouterr().out
        completed = run_rust(["facts", "--profile", str(profile_path), *extra])
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == python_stdout, extra

    # --output prints the same receipt bytes (modulo the path value) and
    # writes identical artifacts.
    python_targets = tmp_path / "targets-python.json"
    rust_targets = tmp_path / "targets-rust.json"
    targets_argv = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
    ]
    assert clankerprof_main([*targets_argv, "--output", str(python_targets)]) == 0
    python_receipt = capsys.readouterr().out
    completed = run_rust([*targets_argv, "--output", str(rust_targets)])
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.replace(
        str(rust_targets), "<path>"
    ) == python_receipt.replace(str(python_targets), "<path>")
    assert rust_targets.read_bytes() == python_targets.read_bytes()

    # compare accepts --output through the global placement in both CLIs,
    # prints a has_regression receipt, and keeps the regression exit code.
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(
        json.dumps(
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [{"name": "A", "pct": 10.0}],
            }
        ),
        encoding="utf-8",
    )
    after_path.write_text(
        json.dumps(
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [{"name": "A", "pct": 50.0}],
            }
        ),
        encoding="utf-8",
    )
    python_compare = tmp_path / "compare-python.json"
    rust_compare = tmp_path / "compare-rust.json"
    assert (
        clankerprof_main(
            [
                "--output",
                str(python_compare),
                "compare",
                "--before",
                str(before_path),
                "--after",
                str(after_path),
            ]
        )
        == 2
    )
    python_receipt = capsys.readouterr().out
    completed = run_rust(
        [
            "--output",
            str(rust_compare),
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
        ]
    )
    assert completed.returncode == 2, completed.stderr
    assert completed.stdout.replace(
        str(rust_compare), "<path>"
    ) == python_receipt.replace(str(python_compare), "<path>")
    assert rust_compare.read_bytes() == python_compare.read_bytes()

    # Usage errors exit 2 with an empty stdout and a JSON envelope on stderr.
    assert clankerprof_main(["targets", "--bogus"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    python_envelope = json.loads(captured.err)
    assert python_envelope["ok"] is False
    completed = run_rust(["targets", "--bogus"])
    assert completed.returncode == 2
    assert completed.stdout == ""
    rust_envelope = json.loads(completed.stderr)
    assert rust_envelope["ok"] is False


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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_rejects_malformed_reports_like_python(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    good: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100},
        "slices": [{"name": "A", "pct": 10.0, "frames": []}],
    }
    cases: list[tuple[dict[str, Any], dict[str, Any], list[str], str]] = [
        (
            {"tool": "clankerprof_slices"},
            good,
            [],
            "Profile comparison input must contain a slices array.",
        ),
        (
            good,
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [{"name": "A", "pct": "not-a-number"}],
            },
            [],
            "Slice field 'pct' must be a number.",
        ),
        (
            good,
            good,
            ["--threshold-abs", "NaN"],
            "Compare thresholds must be finite, non-negative numbers.",
        ),
    ]
    for index, (before, after, extra_argv, message) in enumerate(cases):
        with pytest.raises(ValueError, match=re.escape(message)):
            compare_slice_json(
                before,
                after,
                CompareOptions(threshold_abs=float("nan"))
                if extra_argv
                else CompareOptions(),
            )
        before_path = tmp_path / f"malformed-{index}-before.json"
        after_path = tmp_path / f"malformed-{index}-after.json"
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
                *extra_argv,
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2, (message, completed.stderr)
        assert message in completed.stderr, completed.stderr
        assert '"ok": false' in completed.stderr, completed.stderr


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_aggregates_duplicate_functions_like_python(
    tmp_path: Path,
) -> None:
    before = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100},
        "slices": [
            {
                "name": "A",
                "pct": 30.0,
                "frames": [
                    {"function": "f", "filename": "/one", "pct": 10.0},
                    {"function": "f", "filename": "/two", "pct": 15.0},
                    {"function": "g", "filename": "/g", "pct": 5.0},
                ],
            }
        ],
    }
    after = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100},
        "slices": [
            {
                "name": "A",
                "pct": 30.0,
                "frames": [
                    {"function": "f", "filename": "/one", "pct": 15.0},
                    {"function": "f", "filename": "/two", "pct": 15.0},
                    {"function": "g", "filename": "/g", "pct": 0.0},
                ],
            }
        ],
    }
    before_path = tmp_path / "dup-before.json"
    after_path = tmp_path / "dup-after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")
    rust_out = tmp_path / "dup-rust.out"
    _run_rust_cli(
        [
            "compare",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--output",
            str(rust_out),
        ]
    )
    python_payload = compare_slice_json(before, after)
    frame_deltas = cast(
        list[dict[str, Any]],
        cast(list[dict[str, Any]], python_payload["slices"])[0]["frame_deltas"],
    )
    f_row = next(row for row in frame_deltas if row["function"] == "f")
    assert f_row["before_pct"] == 25.0
    assert f_row["after_pct"] == 30.0
    assert f_row["delta_abs"] == 5.0
    assert [
        row["function"]
        for row in cast(list[dict[str, Any]], python_payload["top_regressions"])
    ] == ["f"]
    rust_text = rust_out.read_text(encoding="utf-8")
    assert json.loads(rust_text) == python_payload
    assert rust_text == render_json_payload(python_payload) + "\n"


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
        json.dumps({"db_queries": {"Example::HtmlResponder#render_template": 40.0}}),
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
    scopes_config_path = tmp_path / "matrix-scopes.toml"
    scopes_config_path.write_text(_SCOPES_PREFERRED_CONFIG, encoding="utf-8")
    scopes_profile_path = tmp_path / "matrix-scopes.pb"
    scopes_profile_path.write_bytes(_scopes_decomposition_profile_bytes())

    slices_config_toml = tmp_path / "matrix-slices-config.toml"
    slices_config_toml.write_text(
        f'profile = "{profile_path}"\n'
        f'slices = "{slices_path}"\n'
        "top = 2\n"
        "by_slice = 5\n"
        "show_paths = true\n"
        'filter = ["!name:NoSuchThing"]\n',
        encoding="utf-8",
    )
    slices_config_yaml = tmp_path / "matrix-slices-config.yaml"
    slices_config_yaml.write_text(
        f"profile: {gc_profile_path}\n"
        f"slices: {slices_path}\n"
        "unattributed_libraries: true\n"
        "no_collapse_native: true\n"
        "top: -1\n",
        encoding="utf-8",
    )
    bracket_profile_path = tmp_path / "matrix-bracket.pb"
    bracket_builder = PprofFixtureBuilder.create()
    bracket_target = bracket_builder.location(
        bracket_builder.function("T#render", "/app/t.rb")
    )
    bracket_a = bracket_builder.location(
        bracket_builder.function("A#call", "/app/a.rb")
    )
    bracket_c = bracket_builder.location(
        bracket_builder.function("C#call", "/app/c.rb")
    )
    bracket_literal = bracket_builder.location(
        bracket_builder.function("L#call", "/app/[x.rb")
    )
    bracket_builder.sample((bracket_a, bracket_target), 5_000_000)
    bracket_builder.sample((bracket_c, bracket_target), 3_000_000)
    bracket_builder.sample((bracket_literal, bracket_target), 2_000_000)
    bracket_profile_path.write_bytes(bracket_builder.encode())
    bracket_config_path = tmp_path / "matrix-bracket-targets.json"
    bracket_config_path.write_text(
        json.dumps(
            {
                "T#render": {
                    "Bracketed": "path:/app/[ab].rb",
                    "Negated": "path:/app/[!ab].rb",
                    "Literal": "path:/app/[x.rb",
                }
            }
        ),
        encoding="utf-8",
    )
    multigz_path = tmp_path / "matrix-multi.pb.gz"
    raw_profile_bytes = _profile_bytes(gzipped=False)
    multigz_path.write_bytes(
        gzip.compress(raw_profile_bytes[:16]) + gzip.compress(raw_profile_bytes[16:])
    )
    boundary_report_path = tmp_path / "matrix-focus-scopes-report.json"
    _run_python_cli(
        [
            "scopes",
            "--profile",
            str(scopes_profile_path),
            "--config",
            str(scopes_config_path),
            "--output",
            str(boundary_report_path),
        ]
    )

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
        (
            "facts-compact",
            ["facts", "--profile", str(profile_path)],
            "bytes",
        ),
        (
            "facts-pretty",
            ["facts", "--profile", str(profile_path), "--pretty"],
            "bytes",
        ),
        (
            "targets-generic-default",
            [
                "targets",
                "--profile",
                str(profile_path),
                "--config",
                str(config_path),
            ],
            "json",
        ),
        (
            "scopes-top",
            [
                "scopes",
                "--profile",
                str(scopes_profile_path),
                "--config",
                str(scopes_config_path),
                "--top",
                "2",
            ],
            "json",
        ),
        (
            "slices-filter-collapse-top-paths",
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--filter",
                "!name:NoSuchThing",
                "--collapse",
                "library:*",
                "--top",
                "3",
                "--show-paths",
                "--runtime",
                "ruby",
            ],
            "json",
        ),
        (
            "slices-config-toml",
            ["slices", "--config", str(slices_config_toml), "--runtime", "ruby"],
            "json",
        ),
        (
            "slices-config-yaml-negative-top",
            ["slices", "--config", str(slices_config_yaml)],
            "json",
        ),
        (
            "slices-by-slice-negative",
            [
                "slices",
                "--profile",
                str(profile_path),
                "--slices",
                str(slices_path),
                "--by-slice",
                "-1",
                "--runtime",
                "ruby",
            ],
            "json",
        ),
        (
            "targets-bracket-class-globs",
            [
                "targets",
                "--profile",
                str(bracket_profile_path),
                "--config",
                str(bracket_config_path),
            ],
            "json",
        ),
        (
            "facts-multimember-gzip",
            ["facts", "--profile", str(multigz_path)],
            "bytes",
        ),
        (
            "compare-focus-scopes-alias",
            [
                "compare",
                "--before",
                str(boundary_report_path),
                "--after",
                str(boundary_report_path),
                "--focus-scopes",
                "x",
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
        # JSON artifacts are byte-identical across emitters (sorted keys,
        # matching indentation); everything compares at the byte level, with
        # a parsed comparison first for a readable diff on JSON regressions.
        if mode == "json":
            assert json.loads(rust_text) == json.loads(python_text), name
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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_report_sections_match_individual_subcommands(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "report.pb"
    profile_path.write_bytes(_scopes_decomposition_profile_bytes())
    config_path = tmp_path / "report-targets.json"
    config_path.write_text(
        json.dumps({"RequestHandler#render_response": {"App": "path:app/**"}}),
        encoding="utf-8",
    )
    slices_path = tmp_path / "report-slices.yml"
    slices_path.write_text(
        "slices:\n  - name: app\n    paths:\n      - app\n  - name: default\n    default: true\n",
        encoding="utf-8",
    )
    scopes_config_path = tmp_path / "report-scopes.toml"
    scopes_config_path.write_text(_SCOPES_PREFERRED_CONFIG, encoding="utf-8")

    report_out = tmp_path / "report.json"
    _run_rust_cli(
        [
            "report",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--slices",
            str(slices_path),
            "--scopes-config",
            str(scopes_config_path),
            "--include-facts",
            "--output",
            str(report_out),
        ]
    )
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["tool"] == "clankerprof_report"

    for section, argv in {
        "facts": ["facts", "--profile", str(profile_path)],
        "targets": [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
        ],
        "slices": [
            "slices",
            "--profile",
            str(profile_path),
            "--slices",
            str(slices_path),
        ],
        "scopes": [
            "scopes",
            "--profile",
            str(profile_path),
            "--config",
            str(scopes_config_path),
        ],
    }.items():
        individual_out = tmp_path / f"report-{section}.json"
        _run_rust_cli([*argv, "--output", str(individual_out)])
        individual = json.loads(individual_out.read_text(encoding="utf-8"))
        assert report[section] == individual, section


def test_clankerprof_sample_facts_schema_version_is_symmetric() -> None:
    from clankerprof.facts import (
        SAMPLE_FACTS_SCHEMA_VERSION,
        SAMPLE_FACTS_SCHEMA_VERSION_V1,
    )

    rust_source = (
        Path(__file__).resolve().parents[1]
        / "crates"
        / "clankerprof-core"
        / "src"
        / "facts.rs"
    ).read_text(encoding="utf-8")
    assert (
        f'pub const SAMPLE_FACTS_SCHEMA_VERSION: &str = "{SAMPLE_FACTS_SCHEMA_VERSION}";'
        in rust_source
    )
    assert (
        f'pub const SAMPLE_FACTS_SCHEMA_VERSION_V1: &str = "{SAMPLE_FACTS_SCHEMA_VERSION_V1}";'
        in rust_source
    )


def _uint64_id_profile_bytes() -> bytes:
    """A raw pprof profile whose location/function IDs are 2**63 (valid uint64)."""
    from tests.fixtures.pprof_builder import (
        field_bytes,
        field_string,
        field_varint,
    )

    big_id = 2**63
    payload = bytearray()
    payload.extend(field_bytes(1, field_varint(1, 1) + field_varint(2, 2)))
    payload.extend(field_bytes(2, field_varint(1, big_id) + field_varint(2, 7)))
    line_message = field_varint(1, big_id) + field_varint(2, 1)
    payload.extend(
        field_bytes(4, field_varint(1, big_id) + field_bytes(4, line_message))
    )
    payload.extend(
        field_bytes(
            5,
            field_varint(1, big_id) + field_varint(2, 3) + field_varint(4, 4),
        )
    )
    for value in ("", "cpu", "nanoseconds", "Target#render", "/app/target.rb"):
        payload.extend(field_string(6, value))
    return bytes(payload)


def _big_aggregate_profile_bytes(sample_count: int) -> bytes:
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("Target#render", "/srv/app/target.py"))
    for _ in range(sample_count):
        builder.sample((target,), 2**63 - 1)
    return builder.encode()


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_facts_uint64_ids_round_trip(tmp_path: Path) -> None:
    profile_path = tmp_path / "bigid.pb"
    profile_path.write_bytes(_uint64_id_profile_bytes())
    config_path = tmp_path / "targets.json"
    config_path.write_text(
        json.dumps({"Target#render": {"Application": "path:/app/**"}}),
        encoding="utf-8",
    )

    python_facts = tmp_path / "python-facts.json"
    rust_facts = tmp_path / "rust-facts.json"
    _run_python_cli(
        ["facts", "--profile", str(profile_path), "--output", str(python_facts)]
    )
    _run_rust_cli(
        ["facts", "--profile", str(profile_path), "--output", str(rust_facts)]
    )
    assert python_facts.read_bytes() == rust_facts.read_bytes()
    assert "9223372036854775808" in python_facts.read_text(encoding="utf-8")

    python_replay = tmp_path / "python-replay.json"
    rust_replay = tmp_path / "rust-replay.json"
    shared = [
        "targets",
        "--facts",
        str(python_facts),
        "--config",
        str(config_path),
        "--format",
        "json",
    ]
    _run_python_cli([*shared, "--output", str(python_replay)])
    _run_rust_cli([*shared, "--output", str(rust_replay)])
    assert python_replay.read_bytes() == rust_replay.read_bytes()
    payload = json.loads(python_replay.read_text(encoding="utf-8"))
    assert payload["parents"]["Target#render"]["total_time_ns"] == 7


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_big_aggregates_match_python_exactly(tmp_path: Path) -> None:
    profile_path = tmp_path / "big-aggregate.pb"
    profile_path.write_bytes(_big_aggregate_profile_bytes(2))
    config_path = tmp_path / "targets.json"
    config_path.write_text(
        json.dumps({"Target#render": {"Application": "path:/srv/app/**"}}),
        encoding="utf-8",
    )

    python_report = tmp_path / "python-report.json"
    rust_report = tmp_path / "rust-report.json"
    shared = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
        "--format",
        "json",
    ]
    _run_python_cli([*shared, "--output", str(python_report)])
    _run_rust_cli([*shared, "--output", str(rust_report)])
    assert python_report.read_bytes() == rust_report.read_bytes()
    # 2 * i64::MAX is exact in both languages: no panic, no float rounding.
    assert "18446744073709551614" in python_report.read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_facts_numeric_contract_matches_python(
    tmp_path: Path,
) -> None:
    from clankerprof.facts import dumps_sample_facts, loads_sample_facts

    repo_root = Path(__file__).resolve().parents[1]
    valid = json.loads(
        dumps_sample_facts(
            decode_profile_bytes(_big_aggregate_profile_bytes(1)).to_sample_facts()
        )
    )

    float_values = json.loads(json.dumps(valid))
    float_values["samples"][0]["values"] = [7.9]
    float_values.pop("summary", None)
    beyond_bound = json.loads(json.dumps(valid))
    template = beyond_bound["samples"][0]
    beyond_bound["samples"] = [dict(template, sample_index=index) for index in range(3)]
    beyond_bound.pop("summary", None)
    cases = [
        (
            float_values,
            "Sample fact values entries must be signed 64-bit integers.",
        ),
        (
            beyond_bound,
            "Aggregate sample values exceed the supported integer range.",
        ),
    ]
    for index, (payload, message) in enumerate(cases):
        with pytest.raises(ValueError, match=re.escape(message)):
            loads_sample_facts(json.dumps(payload))
        facts_path = tmp_path / f"invalid-{index}.json"
        facts_path.write_text(json.dumps(payload), encoding="utf-8")
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
                "--facts",
                str(facts_path),
                "--target",
                "Target#render",
            ],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2, (message, completed.stderr)
        assert message in completed.stderr, completed.stderr
        assert '"ok": false' in completed.stderr, completed.stderr

    # Non-finite JSON tokens fail closed in both languages (RFC 8259 strict).
    infinity_path = tmp_path / "infinity.json"
    infinity_path.write_text(
        '{"schema_version": "clankerprof.sample_facts.v2", "samples": [], '
        '"strings": [], "frames": [], "profile": {"period": Infinity}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite token 'Infinity'"):
        loads_sample_facts(infinity_path.read_text(encoding="utf-8"))
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
            "--facts",
            str(infinity_path),
            "--target",
            "Target#render",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2, completed.stderr
    assert '"ok": false' in completed.stderr, completed.stderr


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_slices_validation_envelopes_match_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    repo_root = Path(__file__).resolve().parents[1]
    profile_path = tmp_path / "validation.pb"
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("T#render", "/srv/app/t.rb"))
    leaf = builder.location(builder.function("Leaf#call", "/srv/app/leaf.rb"))
    builder.sample((leaf, target), 7_000_000)
    profile_path.write_bytes(builder.encode())
    slices_path = tmp_path / "validation-slices.yml"
    slices_path.write_text(
        "slices:\n  - name: app\n    paths:\n      - /srv/app\n",
        encoding="utf-8",
    )
    duplicate_config = tmp_path / "duplicate-top.yaml"
    duplicate_config.write_text(
        f"profile: {profile_path}\ntop: 1\n",
        encoding="utf-8",
    )
    fractional_config = tmp_path / "fractional-top.yaml"
    fractional_config.write_text(
        f"profile: {profile_path}\ntop: 2.5\n",
        encoding="utf-8",
    )
    list_config = tmp_path / "list-config.yaml"
    list_config.write_text("- not-a-mapping\n", encoding="utf-8")
    bogus_scopes_config = tmp_path / "bogus-scopes.yml"
    bogus_scopes_config.write_text(
        'cost_kind:\n  Invalid: "bogus:value"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )

    slice_base = [
        "slices",
        "--profile",
        str(profile_path),
        "--slices",
        str(slices_path),
    ]
    cases: list[tuple[list[str], str]] = [
        (
            ["slices", "--config", str(duplicate_config), "--top", "2"],
            "top specified both on command line and in config file.",
        ),
        (
            ["slices", "--config", str(fractional_config)],
            "top in slice config must be an integer.",
        ),
        (
            ["slices", "--config", str(list_config)],
            "Slice config file must be a YAML object.",
        ),
        (
            [*slice_base, "--by-slice", "garbage"],
            "--by-slice values must be integers.",
        ),
        (
            [*slice_base, "--by-slice", "garbage%"],
            "--by-slice percentage thresholds must be finite numbers.",
        ),
        (
            [*slice_base, "--by-slice", "inf%"],
            "--by-slice percentage thresholds must be finite numbers.",
        ),
        (
            [
                "scopes",
                "--profile",
                str(profile_path),
                "--config",
                str(bogus_scopes_config),
            ],
            "Unsupported predicate key: bogus",
        ),
    ]
    for argv, message in cases:
        assert clankerprof_main(argv) == 2, argv
        python_error = capsys.readouterr().err
        python_payload = json.loads(python_error)
        assert python_payload == {"ok": False, "error": message}, argv
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
        assert completed.returncode == 2, (argv, completed.stderr)
        assert json.loads(completed.stderr) == {"ok": False, "error": message}, argv


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_numeric_edge_semantics_match_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    repo_root = Path(__file__).resolve().parents[1]

    def rust(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
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

    def assert_stdout_parity(argv: list[str]) -> str:
        assert clankerprof_main(argv) == 0, argv
        python_stdout = capsys.readouterr().out
        completed = rust(argv)
        assert completed.returncode == 0, (argv, completed.stderr)
        assert completed.stdout == python_stdout, argv
        return python_stdout

    i64_min = "-9223372036854775808"
    i64_max = 2**63 - 1
    two_i64_max = 18_446_744_073_709_551_614

    # Signed-minimum tail limits (--by-slice / --top reach the same helper).
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("T", "/srv/app/t.py"))
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    builder.sample((leaf, target), 7)
    small_profile = tmp_path / "numeric-small.pb"
    small_profile.write_bytes(builder.encode())
    slices_path = tmp_path / "numeric-slices.yml"
    slices_path.write_text(
        "slices:\n  - name: app\n    paths:\n      - /srv/app\n",
        encoding="utf-8",
    )
    slice_base = [
        "slices",
        "--profile",
        str(small_profile),
        "--slices",
        str(slices_path),
    ]
    for flag in ("--by-slice", "--top"):
        assert_stdout_parity([*slice_base, flag, i64_min])

    # Scope buckets above i64::MAX render (and sort) identically.
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("T", "/srv/app/t.py"))
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    builder.sample((leaf, target), i64_max)
    builder.sample((leaf, target), i64_max)
    big_profile = tmp_path / "numeric-big.pb"
    big_profile.write_bytes(builder.encode())
    scopes_config = tmp_path / "numeric-scopes.yml"
    scopes_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    scopes_base = [
        "scopes",
        "--profile",
        str(big_profile),
        "--config",
        str(scopes_config),
    ]
    big_stdout = assert_stdout_parity(scopes_base)
    big_boundary = json.loads(big_stdout)["boundaries"][0]
    assert big_boundary["total_time_ns"] == two_i64_max
    assert [bucket["time_ns"] for bucket in big_boundary["buckets"]] == [two_i64_max]

    # Occurrence-weighted aggregates beyond u64::MAX fail closed identically.
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("T", "/srv/app/t.py"))
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    builder.sample((leaf, target, target, target), i64_max)
    overflow_profile = tmp_path / "numeric-overflow.pb"
    overflow_profile.write_bytes(builder.encode())
    overflow_argv = [
        "scopes",
        "--profile",
        str(overflow_profile),
        "--config",
        str(scopes_config),
    ]
    assert clankerprof_main(overflow_argv) == 2
    python_error = json.loads(capsys.readouterr().err)
    assert python_error == {
        "ok": False,
        "error": "Aggregate sample values exceed the supported integer range.",
    }
    completed = rust(overflow_argv)
    assert completed.returncode == 2, completed.stderr
    assert json.loads(completed.stderr) == python_error

    # Compare accepts report totals across the full aggregate range.
    report: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": two_i64_max},
        "slices": [{"name": "A", "pct": 50.0, "frames": []}],
    }
    before_path = tmp_path / "numeric-before.json"
    after_path = tmp_path / "numeric-after.json"
    before_path.write_text(json.dumps(report), encoding="utf-8")
    after_path.write_text(json.dumps(report), encoding="utf-8")
    compare_stdout = assert_stdout_parity(
        ["compare", "--before", str(before_path), "--after", str(after_path)]
    )
    compare_payload = json.loads(compare_stdout)
    assert compare_payload["before_total_ns"] == two_i64_max
    assert compare_payload["after_total_ns"] == two_i64_max

    # Mixed-sign costs stay additive and byte-identical.
    builder = PprofFixtureBuilder.create()
    target = builder.location(builder.function("T", "/srv/app/t.py"))
    pos = builder.location(builder.function("Pos", "/srv/app/pos.py"))
    neg = builder.location(builder.function("Neg", "/srv/app/neg.py"))
    builder.sample((pos, target), 10)
    builder.sample((neg, target), -5)
    mixed_profile = tmp_path / "numeric-mixed.pb"
    mixed_profile.write_bytes(builder.encode())
    mixed_config = tmp_path / "numeric-mixed.yml"
    mixed_config.write_text(
        'cost_kind:\n  Pos: "name:Pos"\n  Neg: "name:Neg"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    mixed_stdout = assert_stdout_parity(
        ["scopes", "--profile", str(mixed_profile), "--config", str(mixed_config)]
    )
    mixed_boundary = json.loads(mixed_stdout)["boundaries"][0]
    assert mixed_boundary["total_time_ns"] == 5
    assert (
        sum(bucket["time_ns"] for bucket in mixed_boundary["buckets"])
        == mixed_boundary["total_time_ns"]
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_row_strictness_matches_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    good: dict[str, Any] = {
        "tool": "clankerprof_slices",
        "summary": {"total_time_ns": 100},
        "slices": [{"name": "hot", "pct": 10.0, "frames": []}],
    }
    malformed: list[tuple[dict[str, Any], str]] = [
        (
            {**good, "slices": [{"name": "hot"}]},
            "Slice field 'pct' must be a number.",
        ),
        (
            {**good, "slices": [{"pct": 10.0}]},
            "Slice rows must carry a string 'name'.",
        ),
        ({**good, "slices": ["hot"]}, "Slice rows must be objects."),
        (
            {**good, "slices": [{"name": "hot", "pct": 10.0, "frames": "junk"}]},
            "Slice field 'frames' must be an array.",
        ),
        (
            {
                "tool": "clankerprof_slices",
                "slices": [{"name": "hot", "pct": 10.0}],
            },
            "Report summary must be an object.",
        ),
        (
            {
                "tool": "clankerprof_slices",
                "summary": {},
                "slices": [{"name": "hot", "pct": 10.0}],
            },
            "Report summary field 'total_time_ns' must be an integer.",
        ),
    ]
    good_path = tmp_path / "row-good.json"
    good_path.write_text(json.dumps(good), encoding="utf-8")
    for index, (after, message) in enumerate(malformed):
        after_path = tmp_path / f"row-bad-{index}.json"
        after_path.write_text(json.dumps(after), encoding="utf-8")
        argv = [
            "compare",
            "--before",
            str(good_path),
            "--after",
            str(after_path),
        ]
        assert clankerprof_main(argv) == 2, message
        python_error = json.loads(capsys.readouterr().err)
        assert python_error == {"ok": False, "error": message}
        completed = _rust_cli(argv)
        assert completed.returncode == 2, (message, completed.stderr)
        assert json.loads(completed.stderr) == python_error

    # Row-level absence keeps new/removed semantics byte-identically.
    empty_path = tmp_path / "row-empty.json"
    empty_path.write_text(
        json.dumps({**good, "slices": []}),
        encoding="utf-8",
    )
    argv = ["compare", "--before", str(good_path), "--after", str(empty_path)]
    assert clankerprof_main(argv) == 0
    python_stdout = capsys.readouterr().out
    completed = _rust_cli(argv)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == python_stdout


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_facts_and_scope_validation_matches_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", "/app/leaf.rb"))
    target = builder.location(builder.function("T", "/app/t.rb"))
    builder.sample((leaf, target), 7)
    facts_payload = cast(
        dict[str, Any],
        sample_facts_to_jsonable(
            decode_profile_bytes(builder.encode()).to_sample_facts()
        ),
    )
    targets_config = tmp_path / "validation-targets.json"
    targets_config.write_text(json.dumps({"T": {"App": "path:/app"}}), encoding="utf-8")

    for key in ("values", "location_ids", "stack"):
        mutated = json.loads(json.dumps(facts_payload))
        del mutated["samples"][0][key]
        facts_path = tmp_path / f"validation-missing-{key}.json"
        facts_path.write_text(json.dumps(mutated), encoding="utf-8")
        argv = [
            "targets",
            "--facts",
            str(facts_path),
            "--config",
            str(targets_config),
            "--format",
            "json",
        ]
        assert clankerprof_main(argv) == 2, key
        python_error = json.loads(capsys.readouterr().err)
        assert python_error == {
            "ok": False,
            "error": f"Sample facts payload missing required key: '{key}'.",
        }
        completed = _rust_cli(argv)
        assert completed.returncode == 2, (key, completed.stderr)
        assert json.loads(completed.stderr) == python_error

    facts_path = tmp_path / "validation-facts.json"
    facts_path.write_text(json.dumps(facts_payload), encoding="utf-8")
    scope_cases = [
        (
            "cost_kind:\n  Empty: {}\nscope:\n  - function: T\n",
            "cost_kind Empty predicate table cannot be empty.",
        ),
        (
            "scope:\n  - function: T\n    count: 1\n",
            "scope.count must be occurrence or once_per_sample.",
        ),
    ]
    for index, (config_text, message) in enumerate(scope_cases):
        config_path = tmp_path / f"validation-scopes-{index}.yml"
        config_path.write_text(config_text, encoding="utf-8")
        argv = ["scopes", "--facts", str(facts_path), "--config", str(config_path)]
        assert clankerprof_main(argv) == 2, message
        python_error = json.loads(capsys.readouterr().err)
        assert python_error == {"ok": False, "error": message}
        completed = _rust_cli(argv)
        assert completed.returncode == 2, (message, completed.stderr)
        assert json.loads(completed.stderr) == python_error

    # Owner fallback flags follow Python truthiness in both implementations.
    truthy_config = tmp_path / "validation-fallback.yml"
    truthy_config.write_text(
        "cost_kind:\n"
        '  AppWork: "name:Leaf"\n'
        "owner:\n"
        "  Everything:\n"
        '    match: "path:/nowhere"\n'
        '    fallback: "yes"\n'
        "scope:\n"
        "  - function: T\n",
        encoding="utf-8",
    )
    argv = ["scopes", "--facts", str(facts_path), "--config", str(truthy_config)]
    assert clankerprof_main(argv) == 0
    python_stdout = capsys.readouterr().out
    completed = _rust_cli(argv)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == python_stdout


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_library_regex_fallback_matches_python(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from clankerprof.cli import main as clankerprof_main

    pack_path = tmp_path / "fallback-pack.yml"
    pack_path.write_text(
        "schema_version: clankerprof.runtime_rules.v1\n"
        "name: fallback\n"
        "library_path_patterns:\n"
        '  - "regex:/gems/(foo)?bar/"\n',
        encoding="utf-8",
    )
    for stem, path in (
        ("participating", "/gems/foobar/file.rb"),
        ("fallback", "/gems/bar/file.rb"),
    ):
        builder = PprofFixtureBuilder.create()
        leaf = builder.location(builder.function("Work", path))
        root = builder.location(builder.function("Main", "/srv/app/main.rb"))
        builder.sample((leaf, root), 7)
        profile_path = tmp_path / f"library-{stem}.pb"
        profile_path.write_bytes(builder.encode())
        argv = [
            "slices",
            "--profile",
            str(profile_path),
            "--runtime-rules",
            str(pack_path),
        ]
        assert clankerprof_main(argv) == 0, stem
        python_stdout = capsys.readouterr().out
        completed = _rust_cli(argv)
        assert completed.returncode == 0, (stem, completed.stderr)
        assert completed.stdout == python_stdout, stem
        expected_library = "foo" if stem == "participating" else "gems/bar"
        default_slice = json.loads(python_stdout)["slices"][0]
        libraries = [
            entry["name"] for entry in default_slice.get("unattributed_libraries", [])
        ]
        assert libraries == [expected_library], (stem, libraries)


def _rust_cli(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
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
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_python_cli_raw(argv: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "-m", "clankerprof.cli", *argv],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _run_rust_cli_raw(argv: list[str]) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    return subprocess.run(
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


def _assert_identical_success(argv: list[str], label: str) -> str:
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 0, (label, python_run.stderr)
    assert rust_run.returncode == 0, (label, rust_run.stderr)
    assert rust_run.stdout == python_run.stdout, label
    return python_run.stdout


def _assert_identical_envelope(argv: list[str], label: str) -> str:
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 2, (label, python_run.stdout, python_run.stderr)
    assert rust_run.returncode == 2, (label, rust_run.stdout, rust_run.stderr)
    assert rust_run.stderr == python_run.stderr, label
    return python_run.stderr


def _order_scope_facts(tmp_path: Path) -> Path:
    from clankerprof.facts import dumps_sample_facts

    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    parent = builder.location(builder.function("T", "/srv/app/t.py"))
    builder.sample((leaf, parent), 7)
    facts_path = tmp_path / "order-facts.json"
    facts_path.write_text(
        dumps_sample_facts(decode_profile_bytes(builder.encode()).to_sample_facts()),
        encoding="utf-8",
    )
    return facts_path


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_regex_dialect_and_failclosed_match_python(
    tmp_path: Path,
) -> None:
    facts_path = _order_scope_facts(tmp_path)
    lookahead_pack = tmp_path / "lookahead-pack.yml"
    lookahead_pack.write_text(
        'semantic_rules:\n  - category: Lookahead\n    name_patterns: ["^(?=.*Leaf)Leaf$"]\n',
        encoding="utf-8",
    )
    config_path = tmp_path / "t-config.json"
    config_path.write_text(json.dumps({"T": {}}), encoding="utf-8")
    stdout = _assert_identical_success(
        [
            "targets",
            "--facts",
            str(facts_path),
            "--config",
            str(config_path),
            "--runtime-rules",
            str(lookahead_pack),
        ],
        "lookahead-pack",
    )
    assert '"name": "Lookahead"' in stdout

    # Explicit invalid regexes fail closed in both languages. The message
    # prefix is the byte contract; engine detail after it may differ.
    bad_config = tmp_path / "bad-regex.json"
    bad_config.write_text(json.dumps({"T": {"Bad": "regex:["}}), encoding="utf-8")
    for argv, label in (
        (
            ["targets", "--facts", str(facts_path), "--config", str(bad_config)],
            "explicit-bad-regex",
        ),
    ):
        python_run = _run_python_cli_raw(argv)
        rust_run = _run_rust_cli_raw(argv)
        assert python_run.returncode == 2, (label, python_run.stdout)
        assert rust_run.returncode == 2, (label, rust_run.stdout)
        for run in (python_run, rust_run):
            envelope = json.loads(run.stderr)
            assert envelope["ok"] is False, label
            assert str(envelope["error"]).startswith("Invalid regex pattern '[':"), (
                label,
                envelope,
            )
        assert not python_run.stdout and not rust_run.stdout, label

    bad_pack = tmp_path / "bad-pack.yml"
    bad_pack.write_text(
        'semantic_rules:\n  - category: Broken\n    name_patterns: ["("]\n',
        encoding="utf-8",
    )
    stderr = _assert_identical_envelope(
        [
            "targets",
            "--facts",
            str(facts_path),
            "--target",
            "T",
            "--runtime-rules",
            str(bad_pack),
        ],
        "bad-pack-pattern",
    )
    assert "Invalid runtime rule name pattern '('." in stderr


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_target_parent_order_matches_python(tmp_path: Path) -> None:
    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    ztarget = builder.location(builder.function("ZTarget", "/srv/app/z.py"))
    atarget = builder.location(builder.function("ATarget", "/srv/app/a.py"))
    builder.sample((leaf, ztarget, atarget), 10_000_000)
    profile_path = tmp_path / "parent-order.pb"
    profile_path.write_bytes(builder.encode())
    config_path = tmp_path / "parent-order.json"
    config_path.write_text(json.dumps({"ZTarget": {}, "ATarget": {}}), encoding="utf-8")
    base = [
        "targets",
        "--profile",
        str(profile_path),
        "--config",
        str(config_path),
    ]
    for fmt in ("csv", "simple-csv", "text", "json"):
        stdout = _assert_identical_success(
            [*base, "--format", fmt], f"parent-order-{fmt}"
        )
        if fmt == "simple-csv":
            rows = [line.split(",")[0] for line in stdout.splitlines()[1:] if line]
            # First-seen (stack) order, not alphabetical.
            assert rows == ["ZTarget", "ATarget"]


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_scope_declaration_order_matches_python(
    tmp_path: Path,
) -> None:
    facts_path = _order_scope_facts(tmp_path)
    yaml_config = tmp_path / "order.yml"
    yaml_config.write_text(
        'cost_kind:\n  ZFirst: "path:/srv/app"\n  ASecond: "path:/srv/app"\n'
        "scope:\n  - function: T\n",
        encoding="utf-8",
    )
    toml_config = tmp_path / "order.toml"
    toml_config.write_text(
        '[cost_kind]\nZFirst = "path:/srv/app"\nASecond = "path:/srv/app"\n\n'
        '[[scope]]\nfunction = "T"\n',
        encoding="utf-8",
    )
    owner_config = tmp_path / "owner-order.yml"
    owner_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\n'
        'owner:\n  ZTeam: "path:/srv/app"\n  ATeam: "path:/srv/app"\n'
        "scope:\n  - function: T\n",
        encoding="utf-8",
    )
    for config_path in (yaml_config, toml_config, owner_config):
        stdout = _assert_identical_success(
            ["scopes", "--facts", str(facts_path), "--config", str(config_path)],
            f"declaration-order-{config_path.name}",
        )
        if config_path is not owner_config:
            assert '"ZFirst"' in stdout and '"ASecond"' not in stdout

    label_config = tmp_path / "label-bool.yml"
    label_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\n'
        "scope:\n  - function: T\n    label: true\n",
        encoding="utf-8",
    )
    stderr = _assert_identical_envelope(
        ["scopes", "--facts", str(facts_path), "--config", str(label_config)],
        "label-bool",
    )
    assert "scope.label must be a string." in stderr

    dup_config = tmp_path / "dup.yml"
    dup_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\ncost_kind:\n  Other2: "name:Nope"\n'
        "scope:\n  - function: T\n",
        encoding="utf-8",
    )
    stderr = _assert_identical_envelope(
        ["scopes", "--facts", str(facts_path), "--config", str(dup_config)],
        "duplicate-yaml-key",
    )
    assert json.loads(stderr)["error"] == 'duplicate entry with key "cost_kind"'


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_runtime_flags_match_python(tmp_path: Path) -> None:
    facts_path = _order_scope_facts(tmp_path)
    scope_config = tmp_path / "flags-scopes.yml"
    scope_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    scope_base = ["scopes", "--facts", str(facts_path), "--config", str(scope_config)]
    for extra in (
        ["--fold-runtime-internals"],
        ["--no-enhanced"],
        ["--verbose-runtime-internals"],
        ["--fold-ruby-internals"],
    ):
        _assert_identical_success([*scope_base, *extra], f"scopes-{extra[0]}")
    _assert_identical_success(
        [
            "targets",
            "--facts",
            str(facts_path),
            "--target",
            "T",
            "--verbose-runtime-internals",
        ],
        "targets-verbose",
    )
    _assert_identical_success(
        ["slices", "--facts", str(facts_path), "--verbose-runtime-internals"],
        "slices-verbose",
    )
    # Verbose must actually change ruby-runtime folding output identically.
    ruby_profile = tmp_path / "flags-ruby.pb"
    ruby_profile.write_bytes(_ruby_runtime_profile_bytes())
    ruby_base = [
        "targets",
        "--profile",
        str(ruby_profile),
        "--target",
        "Example::HtmlResponder#render_template",
        "--runtime",
        "ruby",
        "--fold-runtime-internals",
    ]
    plain = _assert_identical_success(ruby_base, "ruby-fold")
    verbose = _assert_identical_success(
        [*ruby_base, "--verbose-runtime-internals"], "ruby-fold-verbose"
    )
    assert plain != verbose, "verbose flag must change ruby folding output"


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_lexical_json_matches_python(tmp_path: Path) -> None:
    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("café_leaf", "/café/x.py"))
    parent = builder.location(builder.function("T", "/srv/t.py"))
    builder.sample((leaf, parent), 1)
    # Astral-plane character in a rendered field exercises surrogate pairs.
    other_leaf = builder.location(builder.function("big", "/srv/app/😀.py"))
    builder.sample((other_leaf, parent), 99_999_999)
    profile_path = tmp_path / "lexical.pb"
    profile_path.write_bytes(builder.encode())
    config_path = tmp_path / "lexical-config.json"
    config_path.write_text(
        json.dumps({"T": {"A": "path:/café", "B": "path:/srv/app"}}),
        encoding="utf-8",
    )
    stdout = _assert_identical_success(
        [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--format",
            "json",
        ],
        "lexical-artifact",
    )
    # ensure_ascii escaping and CPython float repr are the byte contract.
    assert "\\u00e9" in stdout
    assert "\\ud83d\\ude00" in stdout
    assert "1e-06" in stdout
    # Facts artifacts stay raw UTF-8 (the documented exception).
    facts_stdout = _assert_identical_success(
        ["facts", "--profile", str(profile_path)], "lexical-facts"
    )
    assert "café_leaf" in facts_stdout
    # Envelopes escape non-ASCII identically in both languages.
    facts_path = _order_scope_facts(tmp_path)
    dup_key_config = tmp_path / "dup-unicode.yml"
    dup_key_config.write_text(
        'cost_kind:\n  "café": "name:Leaf"\n  "café": "name:Nope"\n'
        "scope:\n  - function: T\n",
        encoding="utf-8",
    )
    # Nested duplicates carry engine-specific context/location detail, so
    # only the shared message core and the \uXXXX escaping are compared.
    argv = ["scopes", "--facts", str(facts_path), "--config", str(dup_key_config)]
    for run, label in (
        (_run_python_cli_raw(argv), "py-unicode-envelope"),
        (_run_rust_cli_raw(argv), "rs-unicode-envelope"),
    ):
        assert run.returncode == 2, (label, run.stdout, run.stderr)
        assert 'duplicate entry with key \\"caf\\u00e9\\"' in run.stderr, (
            label,
            run.stderr,
        )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_value_domain_grammar_matches_python(tmp_path: Path) -> None:
    facts_path = _order_scope_facts(tmp_path)
    slices_path = tmp_path / "grammar-slices.yml"
    slices_path.write_text(
        "slices:\n  - name: app\n    paths:\n      - /srv/app\n",
        encoding="utf-8",
    )
    scopes_path = tmp_path / "grammar-scopes.yml"
    scopes_path.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    slice_base = ["slices", "--facts", str(facts_path), "--slices", str(slices_path)]
    scope_base = ["scopes", "--facts", str(facts_path), "--config", str(scopes_path)]

    # H-01: one strict int64 grammar (sign + ASCII digits + range) per flag.
    envelope = _assert_identical_envelope([*slice_base, "--top", "1_0"], "top-grammar")
    assert json.loads(envelope) == {
        "ok": False,
        "error": "--top values must be integers.",
    }
    _assert_identical_envelope(
        [*slice_base, "--top", "99999999999999999999"], "top-range"
    )
    _assert_identical_envelope(
        [*slice_base, "--unattributed-libraries", "1_0"], "unattributed-grammar"
    )
    _assert_identical_envelope([*scope_base, "--top", "1_0"], "scopes-top-grammar")
    # Signed scope limits keep Python's list[:top] tail semantics.
    _assert_identical_success([*scope_base, "--top", "-1"], "scopes-top-negative")
    _assert_identical_success(
        [*slice_base, "--top", "-9223372036854775808"], "slices-top-i64-min"
    )

    # H-02: focus flags take one comma-delimited value; repeats keep the last.
    report_path = tmp_path / "focus-report.json"
    report_path.write_text(
        json.dumps(
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [{"name": "a", "pct": 10}, {"name": "b", "pct": 5}],
            }
        ),
        encoding="utf-8",
    )
    compare_base = [
        "compare",
        "--before",
        str(report_path),
        "--after",
        str(report_path),
    ]
    focus_last_wins = _assert_identical_success(
        [*compare_base, "--focus-slices", "zzz", "--focus-slices", "a,b"],
        "focus-last-wins",
    )
    assert [row["name"] for row in json.loads(focus_last_wins)["slices"]] == ["a", "b"]

    # H-03: non-string YAML mapping keys fail closed on every YAML surface.
    key_cases: list[tuple[str, str, list[str]]] = [
        (
            "scope-config",
            'cost_kind:\n  true: "name:Leaf"\nscope:\n  - function: T\n',
            ["scopes", "--facts", str(facts_path), "--config"],
        ),
        (
            "slices-config",
            "profile: /dev/null\n1: x\n",
            ["slices", "--config"],
        ),
        (
            "slice-definitions",
            "slices:\n  - name: app\n    1: x\n",
            ["slices", "--facts", str(facts_path), "--slices"],
        ),
        (
            "rule-pack",
            "schema_version: clankerprof.runtime_rules.v1\nname: t\n1: x\n",
            [
                "slices",
                "--facts",
                str(facts_path),
                "--slices",
                str(slices_path),
                "--runtime-rules",
            ],
        ),
    ]
    for name, text, argv_prefix in key_cases:
        bad_path = tmp_path / f"badkey-{name}.yml"
        bad_path.write_text(text, encoding="utf-8")
        envelope = _assert_identical_envelope(
            [*argv_prefix, str(bad_path)], f"non-string-key-{name}"
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "YAML mapping keys must be strings.",
        }
    # Date-like keys stay plain strings in both YAML dialects (no 1.1
    # timestamp resolution) and render byte-identically.
    date_config = tmp_path / "date-key.yml"
    date_config.write_text(
        'cost_kind:\n  2026-01-01: "name:Leaf"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    date_output = _assert_identical_success(
        ["scopes", "--facts", str(facts_path), "--config", str(date_config)],
        "date-key",
    )
    assert '"2026-01-01"' in date_output

    # H-04: predicate/selector arrays require string entries.
    for name, text, message in (
        (
            "selector",
            'cost_kind:\n  AppWork: "name:Leaf"\nscope:\n  - selector: [true]\n',
            "scope selector values must be strings.",
        ),
        (
            "cost-kind-items",
            'cost_kind:\n  AppWork: ["name:Leaf", true]\nscope:\n  - function: T\n',
            "cost_kind AppWork must be a string or array of strings.",
        ),
    ):
        bad_items = tmp_path / f"string-items-{name}.yml"
        bad_items.write_text(text, encoding="utf-8")
        envelope = _assert_identical_envelope(
            ["scopes", "--facts", str(facts_path), "--config", str(bad_items)],
            f"string-items-{name}",
        )
        assert json.loads(envelope) == {"ok": False, "error": message}


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_yaml_scalars_and_attributables_match_python(
    tmp_path: Path,
) -> None:
    facts_path = _order_scope_facts(tmp_path)

    # I-01: YAML 1.1-only scalar forms no longer resolve in Python, so a
    # `top` carrying them fails closed identically in both languages...
    for label, literal in (
        ("underscore", "1_0"),
        ("sexagesimal", "1:2:3"),
        ("quoted-underscore", '"1_0"'),
    ):
        config_path = tmp_path / f"top-{label}.yml"
        config_path.write_text(
            f"profile: /dev/null\ntop: {literal}\n", encoding="utf-8"
        )
        envelope = _assert_identical_envelope(
            ["slices", "--config", str(config_path)], f"top-{label}"
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "top in slice config must be an integer.",
        }
    # ...while YAML 1.2 forms Python used to reject now resolve identically.
    exponent_config = tmp_path / "top-exponent.yml"
    exponent_config.write_text("profile: /dev/null\ntop: 1e2\n", encoding="utf-8")
    _assert_identical_success(
        ["slices", "--config", str(exponent_config)], "top-exponent"
    )
    # `017` stays a string in both YAML dialects, and both string arms then
    # parse it as decimal 17 (Rust trim+parse::<i64>, Python's ASCII mirror).
    leading_zero_config = tmp_path / "top-leading-zero.yml"
    leading_zero_config.write_text("profile: /dev/null\ntop: 017\n", encoding="utf-8")
    _assert_identical_success(
        ["slices", "--config", str(leading_zero_config)], "top-leading-zero"
    )
    yes_config = tmp_path / "label-yes.yml"
    yes_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\nscope:\n  - function: T\n    label: yes\n',
        encoding="utf-8",
    )
    yes_output = _assert_identical_success(
        ["scopes", "--facts", str(facts_path), "--config", str(yes_config)],
        "label-yes",
    )
    assert '"yes"' in yes_output

    # Out-of-64-bit integers fail the YAML parse itself in both engines; the
    # message core is shared, the location suffix is engine-specific.
    overflow_config = tmp_path / "top-overflow.yml"
    overflow_config.write_text(
        "profile: /dev/null\ntop: 18446744073709551616\n", encoding="utf-8"
    )
    argv = ["slices", "--config", str(overflow_config)]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 2, python_run.stderr
    assert rust_run.returncode == 2, rust_run.stderr
    core = (
        "invalid type: integer `18446744073709551616` as u128, expected any YAML value"
    )
    assert core in json.loads(python_run.stderr)["error"]
    assert core in json.loads(rust_run.stderr)["error"]

    # I-02: attributable metrics must be JSON numbers in both languages.
    target_config = tmp_path / "t-config.json"
    target_config.write_text(json.dumps({"T": {}}), encoding="utf-8")
    good_attributables = tmp_path / "attributables-good.json"
    good_attributables.write_text(json.dumps({"col": {"T": 2.5}}), encoding="utf-8")
    targets_base = [
        "targets",
        "--facts",
        str(facts_path),
        "--config",
        str(target_config),
        "--format",
        "csv",
    ]
    _assert_identical_success(
        [*targets_base, "--cpu-attributables", str(good_attributables)],
        "attributables-good",
    )
    for label, bad_value in (("bool", "true"), ("string", '"10"')):
        bad_attributables = tmp_path / f"attributables-{label}.json"
        bad_attributables.write_text(
            f'{{"col": {{"T": {bad_value}}}}}', encoding="utf-8"
        )
        envelope = _assert_identical_envelope(
            [*targets_base, "--cpu-attributables", str(bad_attributables)],
            f"attributables-{label}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Attributable column col values must be numbers.",
        }
    boundary_config = tmp_path / "boundary-attributables.yml"
    boundary_config.write_text(
        'cost_kind:\n  AppWork: "name:Leaf"\n'
        "scope:\n  - function: T\n    attributables:\n      p90: true\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        ["scopes", "--facts", str(facts_path), "--config", str(boundary_config)],
        "boundary-attributables-bool",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Boundary attributable p90 must be a number.",
    }

    # I-03: JSON integer literals outside [i64::MIN, u64::MAX] behave
    # identically without any guard — serde_json parses them as f64 and
    # Python's unbounded int coerces to the identical f64 in float-domain
    # fields. Pinned so neither side grows an asymmetric "fix".
    huge_attributables = tmp_path / "attributables-huge-int.json"
    huge_attributables.write_text(
        '{"col": {"T": 1000000000000000000000000000000}}', encoding="utf-8"
    )
    _assert_identical_success(
        [*targets_base, "--cpu-attributables", str(huge_attributables)],
        "attributables-huge-int",
    )


def test_clankerprof_rust_round3_fixes_match_python(tmp_path: Path) -> None:
    # R3-01 + R3-04: signed rows gate on their true relative increase, and
    # float re-emission is exact (serde_json float_roundtrip): the report
    # below carries 110.00000000000001, which default serde_json re-emitted
    # as 110.0.
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [
                    {"name": "negative", "pct": -10.0, "frames": []},
                    {"name": "positive", "pct": 110.00000000000001, "frames": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "tool": "clankerprof_slices",
                "summary": {"total_time_ns": 100},
                "slices": [
                    {"name": "negative", "pct": -5.0, "frames": []},
                    {"name": "positive", "pct": 105.0, "frames": []},
                ],
            }
        ),
        encoding="utf-8",
    )
    argv = [
        "compare",
        "--before",
        str(before),
        "--after",
        str(after),
        "--threshold-abs",
        "2",
        "--threshold-rel",
        "15",
    ]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 2, python_run.stderr
    assert rust_run.returncode == 2, rust_run.stderr
    assert rust_run.stdout == python_run.stdout
    payload = json.loads(python_run.stdout)
    negative = next(row for row in payload["slices"] if row["name"] == "negative")
    assert negative["delta_rel"] == 50.0
    assert negative["status"] == "regression"
    assert "110.00000000000001" in python_run.stdout

    # R3-02: duplicate top-level rows are rejected identically in both
    # orderings (last-wins made the gate order-dependent).
    duplicated = tmp_path / "duplicated.json"
    for pcts in ((30.0, 10.0), (10.0, 30.0)):
        duplicated.write_text(
            json.dumps(
                {
                    "tool": "clankerprof_slices",
                    "summary": {"total_time_ns": 100},
                    "slices": [
                        {"name": "hot", "pct": pcts[0], "frames": []},
                        {"name": "hot", "pct": pcts[1], "frames": []},
                    ],
                }
            ),
            encoding="utf-8",
        )
        envelope = _assert_identical_envelope(
            ["compare", "--before", str(before), "--after", str(duplicated)],
            f"duplicate-rows-{pcts}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Duplicate Slice row 'hot' in comparison input.",
        }

    # R3-03: a finite attributable that overflows during scaling fails closed
    # with the shared metric-naming message (Rust silently emitted null).
    builder = PprofFixtureBuilder.create()
    scope_fn = builder.function("ScopeFn", "/srv/app/scope.py")
    pos_leaf = builder.function("PosLeaf", "/srv/app/pos.py")
    neg_leaf = builder.function("NegLeaf", "/srv/app/neg.py")
    scope_loc = builder.location(scope_fn)
    builder.sample((builder.location(pos_leaf), scope_loc), 110)
    builder.sample((builder.location(neg_leaf), scope_loc), -10)
    overflow_profile = tmp_path / "overflow.pb"
    overflow_profile.write_bytes(builder.encode())
    overflow_config = tmp_path / "overflow.yml"
    overflow_config.write_text(
        'cost_kind:\n  Pos: "name_eq:PosLeaf"\n  Neg: "name_eq:NegLeaf"\n'
        "scope:\n  - function: ScopeFn\n"
        '    rollup:\n      Positive: ["Pos"]\n      Negative: ["Neg"]\n'
        "    attributables:\n      huge: 1.7e308\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        [
            "scopes",
            "--profile",
            str(overflow_profile),
            "--config",
            str(overflow_config),
        ],
        "attributable-overflow",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Attributable estimate for 'huge' is not finite.",
    }

    # R3-04: aggregate percentages above 2^53 divide identically (Python now
    # mirrors Rust's f64-operand casts), including --by-slice selection.
    builder = PprofFixtureBuilder.create()
    fn_a = builder.function("a_work", "/srv/a/a.py")
    fn_b = builder.function("b_work", "/srv/b/b.py")
    builder.sample((builder.location(fn_a),), 4813636180488882346)
    builder.sample((builder.location(fn_b),), 3969268729998903787)
    huge_profile = tmp_path / "huge.pb"
    huge_profile.write_bytes(builder.encode())
    huge_slices = tmp_path / "huge_slices.yml"
    huge_slices.write_text(
        "slices:\n  - name: a\n    paths: [/srv/a]\n  - name: b\n    default: true\n",
        encoding="utf-8",
    )
    output = _assert_identical_success(
        ["slices", "--profile", str(huge_profile), "--slices", str(huge_slices)],
        "huge-total-pct",
    )
    assert "54.80688029242869" in output
    selected = _assert_identical_success(
        [
            "slices",
            "--profile",
            str(huge_profile),
            "--slices",
            str(huge_slices),
            "--by-slice",
            "54.80688029242869%",
        ],
        "huge-total-by-slice",
    )
    assert [row["name"] for row in json.loads(selected)["slices"]] == ["a"]

    # R3-05: negative GC / uncollapsible pseudo-outputs are rendered.
    builder = PprofFixtureBuilder.create()
    marking = builder.function("(marking)", "")
    builder.sample((builder.location(marking),), -10)
    gc_profile = tmp_path / "gc_neg.pb"
    gc_profile.write_bytes(builder.encode())
    output = _assert_identical_success(
        ["slices", "--profile", str(gc_profile)], "gc-negative"
    )
    assert json.loads(output)["gc"] == {"pct": 100.0, "time_ns": -10}
    builder = PprofFixtureBuilder.create()
    leaf = builder.function("Leaf", "/srv/app/leaf.py")
    builder.sample((builder.location(leaf),), -10)
    collapse_profile = tmp_path / "collapse_neg.pb"
    collapse_profile.write_bytes(builder.encode())
    output = _assert_identical_success(
        [
            "slices",
            "--profile",
            str(collapse_profile),
            "--collapse",
            "path:/srv/app/leaf.py",
        ],
        "uncollapsible-negative",
    )
    assert json.loads(output)["uncollapsible"]["time_ns"] == -10

    # R3-06: non-string entries on the three remaining config surfaces fail
    # closed with shared messages instead of coercing/dropping divergently.
    builder = PprofFixtureBuilder.create()
    worker = builder.function("work", "/srv/123/file.py")
    builder.sample((builder.location(worker),), 7)
    strings_profile = tmp_path / "strings.pb"
    strings_profile.write_bytes(builder.encode())
    numeric_slices = tmp_path / "numeric_slices.yml"
    numeric_slices.write_text(
        "slices:\n  - name: numeric\n    paths: [123]\n"
        "  - name: fallback\n    default: true\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        [
            "slices",
            "--profile",
            str(strings_profile),
            "--slices",
            str(numeric_slices),
        ],
        "slice-paths-numeric",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Slice paths values must be strings.",
    }
    numeric_targets = tmp_path / "numeric_targets.json"
    numeric_targets.write_text('{"T": {"Numeric": 123}}', encoding="utf-8")
    envelope = _assert_identical_envelope(
        [
            "targets",
            "--profile",
            str(strings_profile),
            "--config",
            str(numeric_targets),
            "--format",
            "json",
        ],
        "target-pattern-numeric",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Target config pattern for Numeric must be a string.",
    }
    null_rules = tmp_path / "null_rules.yml"
    null_rules.write_text(
        "schema_version: clankerprof.runtime_rules.v1\nname: strictness\n"
        "native_rules:\n  - category: Hit\n    name_contains:\n      - null\n",
        encoding="utf-8",
    )
    plain_targets = tmp_path / "plain_targets.json"
    plain_targets.write_text('{"T": {"App": "path:/srv"}}', encoding="utf-8")
    envelope = _assert_identical_envelope(
        [
            "targets",
            "--profile",
            str(strings_profile),
            "--config",
            str(plain_targets),
            "--runtime-rules",
            str(null_rules),
        ],
        "rule-entries-null",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Runtime rule field name_contains entries must be strings.",
    }

    # R3-07: compare thresholds share one strict float grammar.
    for spelling in ("1_0", " 2 ", "nan", "1e999"):
        envelope = _assert_identical_envelope(
            [
                "compare",
                "--before",
                str(before),
                "--after",
                str(before),
                "--threshold-abs",
                spelling,
            ],
            f"threshold-{spelling!r}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Compare thresholds must be finite, non-negative numbers.",
        }
    _assert_identical_success(
        [
            "compare",
            "--before",
            str(before),
            "--after",
            str(before),
            "--threshold-abs",
            "2.5",
        ],
        "threshold-control",
    )

    # R3-04 sweep: zero-total profiles (mixed signs summing to zero) exercise
    # every pct field's zero arm — Python emits integer 0 there, and Rust must
    # spell it identically (not 0.0).
    builder = PprofFixtureBuilder.create()
    t_fn = builder.function("T", "/app/t.py")
    pos_fn = builder.function("pos", "/srv/a/a.py")
    neg_fn = builder.function("neg", "/srv/b/b.py")
    builder.sample((builder.location(pos_fn), builder.location(t_fn)), 10)
    builder.sample((builder.location(neg_fn), builder.location(t_fn)), -10)
    zero_profile = tmp_path / "zero_total.pb"
    zero_profile.write_bytes(builder.encode())
    output = _assert_identical_success(
        ["slices", "--profile", str(zero_profile)], "zero-total-slices"
    )
    assert '"pct": 0,' in output
    # The target T actually occurs in this fixture (R4-03 de-vacuated this
    # leg: it previously configured an absent target and only requested
    # JSON, so the unguarded CSV/text divisions were never exercised).
    zero_targets = tmp_path / "zero_targets.json"
    zero_targets.write_text(
        '{"T": {"Pos": "path:/srv/a", "Neg": "path:/srv/b"}}', encoding="utf-8"
    )
    for zero_format in ("json", "csv", "simple-csv", "text"):
        output = _assert_identical_success(
            [
                "targets",
                "--profile",
                str(zero_profile),
                "--config",
                str(zero_targets),
                "--format",
                zero_format,
            ],
            f"zero-total-targets-{zero_format}",
        )
        assert "inf" not in output and "NaN" not in output
        if zero_format == "text":
            assert "TOTAL" in output and "100.00%" not in output
    zero_scopes = tmp_path / "zero_scopes.yml"
    zero_scopes.write_text(
        'cost_kind:\n  Work: "name_eq:pos"\nscope:\n  - function: pos\n',
        encoding="utf-8",
    )
    _assert_identical_success(
        ["scopes", "--profile", str(zero_profile), "--config", str(zero_scopes)],
        "zero-total-scopes",
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_slice_default_boolean_matches_python(
    tmp_path: Path,
) -> None:
    facts_path = _order_scope_facts(tmp_path)

    # `default` accepts only YAML booleans: Python truthiness (`default: 1`
    # meant default) diverged from Rust's as_bool (silently non-default),
    # observably flipping slice attribution. Both now fail closed.
    for label, literal in (("int", "1"), ("string", '"yes"')):
        slices_path = tmp_path / f"default-{label}.yml"
        slices_path.write_text(
            f"slices:\n  - name: catch\n    default: {literal}\n",
            encoding="utf-8",
        )
        envelope = _assert_identical_envelope(
            ["slices", "--facts", str(facts_path), "--slices", str(slices_path)],
            f"slice-default-{label}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Slice default must be a boolean.",
        }

    # Boolean and null/absent spellings keep working byte-identically.
    good_path = tmp_path / "default-bool.yml"
    good_path.write_text(
        "slices:\n  - name: catch\n    default: true\n  - name: other\n",
        encoding="utf-8",
    )
    _assert_identical_success(
        ["slices", "--facts", str(facts_path), "--slices", str(good_path)],
        "slice-default-bool",
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_compare_fail_closed_matches_python(tmp_path: Path) -> None:
    def write(name: str, payload: str) -> Path:
        path = tmp_path / name
        path.write_text(payload, encoding="utf-8")
        return path

    good_slices = write(
        "before.json",
        '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
        '"slices": [{"name": "A", "pct": 10.0, "frames": []}]}',
    )

    # R4-01: a present null nested-row array fails closed identically (the
    # Python side previously read it as absent and turned a real regression
    # into an apparent removal).
    null_frames = write(
        "null-frames.json",
        '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
        '"slices": [{"name": "A", "pct": 10.0, "frames": null}]}',
    )
    envelope = _assert_identical_envelope(
        ["compare", "--before", str(good_slices), "--after", str(null_frames)],
        "null-frames",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Slice field 'frames' must be an array.",
    }
    good_boundaries = write(
        "boundaries.json",
        '{"tool": "clankerprof_boundaries", "summary": {"total_time_ns": 100}, '
        '"boundaries": [{"name": "B", "pct_of_profile": 40.0, "buckets": null}]}',
    )
    envelope = _assert_identical_envelope(
        ["compare", "--before", str(good_boundaries), "--after", str(good_boundaries)],
        "null-buckets",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Boundary field 'buckets' must be an array.",
    }

    # R4-02: duplicate JSON member names fail closed in BOTH orderings — the
    # message core is shared; serde's location suffix is engine-specific.
    for label, ordering in (
        ("dup-30-10", '"pct": 30.0, "pct": 10.0'),
        ("dup-10-30", '"pct": 10.0, "pct": 30.0'),
    ):
        dup_path = write(
            f"{label}.json",
            '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
            '"slices": [{"name": "A", ' + ordering + "}]}",
        )
        argv = ["compare", "--before", str(good_slices), "--after", str(dup_path)]
        python_run = _run_python_cli_raw(argv)
        rust_run = _run_rust_cli_raw(argv)
        assert python_run.returncode == 2, (label, python_run.stderr)
        assert rust_run.returncode == 2, (label, rust_run.stderr)
        core = 'duplicate entry with key "pct"'
        assert core in json.loads(python_run.stderr)["error"], label
        assert core in json.loads(rust_run.stderr)["error"], label

    # ...and the duplicate-member rule covers every JSON input surface: facts
    # artifacts and target configs, not just compare reports.
    facts_path = _order_scope_facts(tmp_path)
    target_config = write("t-config.json", '{"T": {}}')
    dup_facts = write(
        "dup-facts.json", '{"schema_version": "x", "schema_version": "y"}'
    )
    argv = [
        "targets",
        "--facts",
        str(dup_facts),
        "--config",
        str(target_config),
    ]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    core = 'duplicate entry with key "schema_version"'
    assert python_run.returncode == 2 and rust_run.returncode == 2
    assert core in json.loads(python_run.stderr)["error"]
    assert core in json.loads(rust_run.stderr)["error"]
    dup_config = write(
        "dup-config.json", '{"T": {"App": "path:/a", "App": "path:/b"}}'
    )
    argv = [
        "targets",
        "--facts",
        str(facts_path),
        "--config",
        str(dup_config),
    ]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    core = 'duplicate entry with key "App"'
    assert python_run.returncode == 2 and rust_run.returncode == 2
    assert core in json.loads(python_run.stderr)["error"]
    assert core in json.loads(rust_run.stderr)["error"]

    # R4-10: finite inputs whose derived sums/deltas overflow fail closed with
    # one shared typed message (previously a leaked CPython json message in
    # Python and a corrupt null-bearing exit-0 report in Rust).
    overflow_frames = write(
        "overflow-frames.json",
        '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
        '"slices": [{"name": "A", "pct": 10.0, "frames": ['
        '{"function": "f", "pct": 1e308}, {"function": "f", "pct": 1e308}]}]}',
    )
    envelope = _assert_identical_envelope(
        ["compare", "--before", str(good_slices), "--after", str(overflow_frames)],
        "overflow-frames",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Compare values for 'f' are not finite.",
    }
    low = write(
        "low.json",
        '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
        '"slices": [{"name": "A", "pct": -1e308}]}',
    )
    high = write(
        "high.json",
        '{"tool": "clankerprof_slices", "summary": {"total_time_ns": 100}, '
        '"slices": [{"name": "A", "pct": 1e308}]}',
    )
    envelope = _assert_identical_envelope(
        ["compare", "--before", str(low), "--after", str(high)],
        "overflow-delta",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Compare values for 'A' are not finite.",
    }

    # R4-13: negative thresholds are option-validation errors (0 > -1 would
    # gate identical reports as regressions); zero stays legal.
    envelope = _assert_identical_envelope(
        [
            "compare",
            "--before",
            str(good_slices),
            "--after",
            str(good_slices),
            "--threshold-abs=-1",
        ],
        "negative-threshold",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Compare thresholds must be finite, non-negative numbers.",
    }
    _assert_identical_success(
        [
            "compare",
            "--before",
            str(good_slices),
            "--after",
            str(good_slices),
            "--threshold-abs",
            "0",
            "--threshold-rel",
            "0",
        ],
        "zero-thresholds",
    )


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_target_renderer_semantics_match_python(
    tmp_path: Path,
) -> None:
    # R4 cluster N: signed/zero-total rendering, attributable finiteness on
    # the targets surface, deep-native caller selection, and quoted
    # core-class CSV must behave identically in both languages.
    builder = PprofFixtureBuilder.create()
    t_fn = builder.function("T", "/app/t.py")
    pos_leaf = builder.function("PosLeaf", "/srv/app/pos.py")
    neg_leaf = builder.function("NegLeaf", "/srv/app/neg.py")
    t_loc = builder.location(t_fn)
    builder.sample((builder.location(pos_leaf), t_loc), 110)
    builder.sample((builder.location(neg_leaf), t_loc), -10)
    signed_profile = tmp_path / "signed.pb"
    signed_profile.write_bytes(builder.encode())
    signed_config = tmp_path / "signed_targets.json"
    signed_config.write_text(
        json.dumps({"T": {"Positive": "pos.py", "Negative": "neg.py"}}),
        encoding="utf-8",
    )
    signed_base = [
        "targets",
        "--profile",
        str(signed_profile),
        "--config",
        str(signed_config),
    ]

    # R4-06: the simplified noise gate is magnitude-aware — the -10%
    # category renders in simple-csv, byte-identically.
    output = _assert_identical_success(
        [*signed_base, "--format", "simple-csv"], "negative-simple-csv"
    )
    assert "T,Negative,-10.0" in output

    # R4-04: a finite metric scaled by the 110% share overflows and fails
    # closed with the identical envelope in both languages.
    overflow_attrs = tmp_path / "attrs_finite.json"
    overflow_attrs.write_text(json.dumps({"huge": {"T": 1.7e308}}), encoding="utf-8")
    for fmt in ("csv", "simple-csv"):
        envelope = _assert_identical_envelope(
            [*signed_base, "--format", fmt, "--cpu-attributables", str(overflow_attrs)],
            f"attributable-overflow-{fmt}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Attributable estimate for 'huge' is not finite.",
        }
    # Non-finite at load: Python rejects the parsed inf, Rust rejects at the
    # JSON parse — exit codes match, message detail is engine-specific.
    huge_attrs = tmp_path / "attrs_1e309.json"
    huge_attrs.write_text('{"huge": {"T": 1e309}}', encoding="utf-8")
    argv = [*signed_base, "--format", "csv", "--cpu-attributables", str(huge_attrs)]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 2, python_run.stderr
    assert rust_run.returncode == 2, rust_run.stderr

    # R4-07: the first eligible application caller wins at any native-run
    # depth (the nine-frame window is gone); byte-identical outputs.
    deep_builder = PprofFixtureBuilder.create()
    deep_stack = [deep_builder.location(deep_builder.function("Leaf", "<native>"))]
    for index in range(9):
        deep_stack.append(
            deep_builder.location(
                deep_builder.function(f"Native{index + 1}", "<native>")
            )
        )
    deep_stack.append(
        deep_builder.location(deep_builder.function("AppCaller", "/app/caller.py"))
    )
    deep_stack.append(deep_builder.location(deep_builder.function("T", "/app/t.py")))
    deep_builder.sample(tuple(deep_stack), 100)
    deep_profile = tmp_path / "deep_native.pb"
    deep_profile.write_bytes(deep_builder.encode())
    output = _assert_identical_success(
        ["targets", "--profile", str(deep_profile), "--target", "T", "--format", "simple-csv"],
        "deep-native-caller",
    )
    assert "AppCaller (100.0%)" in output

    # R4-09: quoted core-class CSV fields parse with csv.reader semantics in
    # both languages (quotes unwrapped, doubled quotes unescaped).
    ruby_builder = PprofFixtureBuilder.create()
    ruby_t = ruby_builder.function("T", "/srv/app/t.py")
    ruby_leaf = ruby_builder.function("Array#map", "<cfunc>")
    ruby_builder.sample(
        (ruby_builder.location(ruby_leaf), ruby_builder.location(ruby_t)), 7
    )
    ruby_profile = tmp_path / "ruby.pb"
    ruby_profile.write_bytes(ruby_builder.encode())
    ruby_config = tmp_path / "ruby_targets.json"
    ruby_config.write_text(json.dumps({"T": {"App": "/srv/app/**"}}), encoding="utf-8")
    core_csv = tmp_path / "core.csv"
    core_csv.write_text('"Array"\n', encoding="utf-8")
    output = _assert_identical_success(
        [
            "targets",
            "--profile",
            str(ruby_profile),
            "--config",
            str(ruby_config),
            "--target",
            "T",
            "--runtime",
            "ruby",
            "--core-classes",
            str(core_csv),
        ],
        "quoted-core-classes",
    )
    assert "Ruby Core (Native)" in output


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_r4_slices_scopes_facts_validation_matches_python(
    tmp_path: Path,
) -> None:
    from clankerprof.facts import dumps_sample_facts

    # R4-05: negative scope totals scale attributable estimates by signed
    # share instead of erasing them.
    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
    parent = builder.location(builder.function("T", "/srv/app/t.py"))
    builder.sample((leaf, parent), -10)
    negative_facts = tmp_path / "negative-facts.json"
    negative_facts.write_text(
        dumps_sample_facts(decode_profile_bytes(builder.encode()).to_sample_facts()),
        encoding="utf-8",
    )
    scopes_config = tmp_path / "negative-scopes.yml"
    scopes_config.write_text(
        'cost_kind:\n  Work: "name:Leaf"\n'
        "scope:\n  - function: T\n"
        '    rollup:\n      All: ["Work"]\n'
        "    attributables:\n      p90: 100.0\n",
        encoding="utf-8",
    )
    output = _assert_identical_success(
        ["scopes", "--facts", str(negative_facts), "--config", str(scopes_config)],
        "negative-total-attributables",
    )
    boundary = json.loads(output)["boundaries"][0]
    assert boundary["buckets"][0]["attributable_estimates"] == {"p90": 100.0}
    assert boundary["buckets"][0]["categories"][0]["attributable_estimates"] == {
        "p90": 100.0
    }

    # R4-08: negated descendant filters bind to descendant existence.
    facts_path = _order_scope_facts(tmp_path)
    excluded = _assert_identical_success(
        ["slices", "--facts", str(facts_path), "--filter", "<!name:T"],
        "negated-descendant-present",
    )
    assert json.loads(excluded)["summary"]["matching_time_ns"] == 0
    kept = _assert_identical_success(
        ["slices", "--facts", str(facts_path), "--filter", "<!name:Missing"],
        "negated-descendant-absent",
    )
    assert json.loads(kept)["summary"]["matching_time_ns"] == 7

    # R4-11: an empty present --by-slice value is rejected, never a silent
    # filtering bypass.
    slices_yaml = tmp_path / "by-slice-slices.yml"
    slices_yaml.write_text(
        "slices:\n  - name: app\n    paths:\n      - /srv/app\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        [
            "slices",
            "--facts",
            str(facts_path),
            "--slices",
            str(slices_yaml),
            "--by-slice",
            "",
        ],
        "by-slice-empty",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "--by-slice values must be integers.",
    }

    # R4-12: non-finite slice metadata fails closed with one shared message
    # (Rust previously emitted a silent null).
    for label, yaml_text in (
        ("nested", "slices:\n  - name: app\n    metadata: {score: .nan}\n"),
        ("list", "slices:\n  - name: app\n    labels: [ok, .inf]\n"),
    ):
        metadata_yaml = tmp_path / f"metadata-{label}.yml"
        metadata_yaml.write_text(yaml_text, encoding="utf-8")
        envelope = _assert_identical_envelope(
            ["slices", "--facts", str(facts_path), "--slices", str(metadata_yaml)],
            f"metadata-{label}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Slice metadata values must be finite JSON-compatible numbers.",
        }

    # R4-14: a present facts summary must be an object; null keeps reading
    # as absent.
    valid = json.loads(facts_path.read_text(encoding="utf-8"))
    wrong_summaries: tuple[tuple[str, object], ...] = (
        ("array", []),
        ("string", "free-form"),
    )
    for label, summary in wrong_summaries:
        wrong = dict(valid)
        wrong["summary"] = summary
        wrong_path = tmp_path / f"summary-{label}.json"
        wrong_path.write_text(json.dumps(wrong, sort_keys=True), encoding="utf-8")
        envelope = _assert_identical_envelope(
            ["slices", "--facts", str(wrong_path)], f"summary-{label}"
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "Sample facts summary must be an object.",
        }
    null_summary = dict(valid)
    null_summary["summary"] = None
    null_path = tmp_path / "summary-null.json"
    null_path.write_text(json.dumps(null_summary, sort_keys=True), encoding="utf-8")
    _assert_identical_success(["slices", "--facts", str(null_path)], "summary-null")


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_round5_cluster_p_matches_python(tmp_path: Path) -> None:
    # R5-01: bottom slice: filters honor the descendant-attribute rescue in
    # both polarities (a rescued sample matches slice:<name> and is excluded
    # by !slice:<name>) in both languages.
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
    builder.sample((cache, wrapper, component, request), 70)
    rescue_profile = tmp_path / "rescue.pb"
    rescue_profile.write_bytes(builder.encode())
    rescue_slices = tmp_path / "rescue-slices.yml"
    rescue_slices.write_text(
        "slices:\n"
        "  - name: components\n"
        '    paths: ["app/components/**"]\n'
        "  - name: instrumentation\n",
        encoding="utf-8",
    )
    rescue_base = [
        "slices",
        "--profile",
        str(rescue_profile),
        "--slices",
        str(rescue_slices),
        "--attribute",
        "<name:TelemetryWrapper#call,to:instrumentation",
    ]
    included = _assert_identical_success(
        [*rescue_base, "--filter", "slice:instrumentation"], "rescue-include"
    )
    assert '"matching_time_ns": 70' in included
    excluded = _assert_identical_success(
        [*rescue_base, "--filter", "!slice:instrumentation"], "rescue-exclude"
    )
    assert '"matching_time_ns": 0' in excluded

    # R5-04: equal-cost owner functions keep first-seen encounter order under
    # scopes --top truncation (ZOwner is met before AOwner and must be the
    # one that survives --top 1 in both languages).
    builder = PprofFixtureBuilder.create()
    scope_fn = builder.function(
        "RequestHandler#render_response", "/app/http/request.rb"
    )
    z_owner = builder.location(builder.function("ZOwner", "app/rendering/owner.rb"))
    a_owner = builder.location(builder.function("AOwner", "app/rendering/owner.rb"))
    builder.sample((z_owner, builder.location(scope_fn)), 10)
    builder.sample((a_owner, builder.location(scope_fn)), 10)
    tie_profile = tmp_path / "tie.pb"
    tie_profile.write_bytes(builder.encode())
    tie_config = tmp_path / "tie-scopes.toml"
    tie_config.write_text(
        '[cost_kind]\n"App" = "path:app/**"\n\n'
        '[owner]\n"Rendering" = "path:app/rendering/**"\n\n'
        "[[scope]]\n"
        'label = "Request render"\n'
        'match = "name_eq:RequestHandler#render_response"\n',
        encoding="utf-8",
    )
    tie_stdout = _assert_identical_success(
        [
            "scopes",
            "--profile",
            str(tie_profile),
            "--config",
            str(tie_config),
            "--top",
            "1",
        ],
        "scope-top-tie-order",
    )
    tie_functions = json.loads(tie_stdout)["boundaries"][0]["domains"][0]["files"][0][
        "functions"
    ]
    assert list(tie_functions) == ["ZOwner"]

    # R5-05: integral config by_slice floats beyond i64 fail closed with the
    # shared strict-grammar envelope instead of Rust clamping and proceeding.
    overflow_slices = tmp_path / "overflow-slices.yml"
    overflow_slices.write_text(
        "slices:\n  - name: app\n    paths:\n      - /srv/app\n", encoding="utf-8"
    )
    for label, literal in (("positive", "1.0e20"), ("negative", "-1.0e20")):
        overflow_config = tmp_path / f"by-slice-{label}.yml"
        overflow_config.write_text(f"by_slice: {literal}\n", encoding="utf-8")
        envelope = _assert_identical_envelope(
            [
                "slices",
                "--profile",
                str(rescue_profile),
                "--slices",
                str(overflow_slices),
                "--config",
                str(overflow_config),
            ],
            f"by-slice-overflow-{label}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": "--by-slice values must be integers.",
        }

    # R5-09: quoted core-class CSV fields spanning newlines parse with
    # csv.reader semantics in both languages.
    builder = PprofFixtureBuilder.create()
    weird_leaf = builder.location(builder.function("Weird\nClass#run", "<cfunc>"))
    parent = builder.location(builder.function("Parent", "/srv/app/parent.rb"))
    builder.sample((weird_leaf, parent), 10)
    weird_profile = tmp_path / "weird.pb"
    weird_profile.write_bytes(builder.encode())
    weird_csv = tmp_path / "weird-core.csv"
    weird_csv.write_bytes(b'"Weird\nClass"\n')
    weird_stdout = _assert_identical_success(
        [
            "targets",
            "--profile",
            str(weird_profile),
            "--target",
            "Parent",
            "--runtime",
            "ruby",
            "--core-classes",
            str(weird_csv),
            "--format",
            "json",
        ],
        "multiline-core-class-csv",
    )
    assert "Ruby Core (Native)" in weird_stdout


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_round5_cluster_q_matches_python(tmp_path: Path) -> None:
    minimal: dict[str, object] = {
        "schema_version": "clankerprof.sample_facts.v2",
        "tool": "clankerprof_facts",
        "profile": {
            "value_types": [],
            "period_type": None,
            "period": 0,
            "default_sample_type": "",
            "primary_value_index": 0,
        },
        "summary": {
            "sample_count": 0,
            "empty_sample_count": 0,
            "non_empty_sample_count": 0,
            "total_primary_value": 0,
        },
        "strings": [],
        "frames": [],
        "samples": [],
    }
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(minimal), encoding="utf-8")

    # R5-03: deep JSON hits serde_json's fixed 128 limit byte-identically.
    deep_path = tmp_path / "deep.json"
    deep_path.write_text("[" * 300 + "]" * 300, encoding="utf-8")
    envelope = _assert_identical_envelope(
        ["targets", "--facts", str(deep_path), "--target", "X"], "deep-json"
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "recursion limit exceeded at line 1 column 128",
    }
    # YAML alias cycle: shared message core, location suffix engine-specific.
    cycle_path = tmp_path / "cycle.yml"
    cycle_path.write_text(
        "slices:\n  - name: app\n    metadata: &a\n      x: *a\n", encoding="utf-8"
    )
    argv = ["slices", "--facts", str(facts_path), "--slices", str(cycle_path)]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    assert python_run.returncode == 2, python_run.stderr
    assert rust_run.returncode == 2, rust_run.stderr
    for run in (python_run, rust_run):
        assert json.loads(run.stderr)["error"].startswith("recursion limit exceeded")

    # R5-07: duplicate slice names fail closed identically.
    dup_path = tmp_path / "dup.yml"
    dup_path.write_text(
        "slices:\n  - name: App\n    paths: [/app]\n  - name: App\n    paths: [/o]\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        ["slices", "--facts", str(facts_path), "--slices", str(dup_path)],
        "dup-slice-names",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Slice config declares duplicate slice name: App. "
        "Each slice name may be defined once.",
    }

    # R5-08: v2 profile metadata is validated, never defaulted or coerced.
    profile = cast(dict[str, object], minimal["profile"])
    missing_pvi = {**profile}
    del missing_pvi["primary_value_index"]
    meta_cases: list[tuple[str, dict[str, object], str]] = [
        (
            "no-pvi",
            missing_pvi,
            "Sample facts payload missing required key: 'primary_value_index'.",
        ),
        (
            "bool-type",
            {**profile, "value_types": [{"type": True, "unit": 7}]},
            "Sample facts value type type must be a string.",
        ),
        (
            "bool-dst",
            {**profile, "default_sample_type": True},
            "Sample facts profile default_sample_type must be a string.",
        ),
    ]
    for label, meta, message in meta_cases:
        case_path = tmp_path / f"meta-{label}.json"
        case_path.write_text(json.dumps({**minimal, "profile": meta}), encoding="utf-8")
        envelope = _assert_identical_envelope(
            ["targets", "--facts", str(case_path), "--target", "X"], f"meta-{label}"
        )
        assert json.loads(envelope) == {"ok": False, "error": message}

    # R5-10: tuple-identity aggregation renders two distinct rows for frames
    # whose NUL-joined spellings collide.
    nul = "\x00"
    collision = {
        **minimal,
        "strings": ["A", f"B{nul}C", f"A{nul}B", "C"],
        "frames": [[1, 1, 0, 1, 0, False], [2, 2, 2, 3, 0, False]],
        "samples": [
            {"sample_index": 0, "values": [10], "location_ids": [1], "stack": [0]},
            {"sample_index": 1, "values": [20], "location_ids": [2], "stack": [1]},
        ],
        "summary": {
            "sample_count": 2,
            "empty_sample_count": 0,
            "non_empty_sample_count": 2,
            "total_primary_value": 30,
        },
    }
    collision_path = tmp_path / "nul-collision.json"
    collision_path.write_text(json.dumps(collision), encoding="utf-8")
    output = _assert_identical_success(
        ["slices", "--facts", str(collision_path)], "nul-collision"
    )
    payload = json.loads(output)
    frames = [
        (frame["function"], frame["filename"], frame["time_ns"])
        for item in payload["slices"]
        for frame in item["frames"]
    ]
    assert sorted(frames) == [("A", f"B{nul}C", 10), (f"A{nul}B", "C", 20)]

    # R5-11: protobuf strictness — field zero and invalid UTF-8.
    field_zero = tmp_path / "field0.pb"
    field_zero.write_bytes(b"\x00\x00")
    envelope = _assert_identical_envelope(
        ["facts", "--profile", str(field_zero)], "proto-field0"
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Illegal protobuf field number 0.",
    }
    bad_utf8 = tmp_path / "badutf8.pb"
    bad_utf8.write_bytes(b"\x32\x01\xff")
    envelope = _assert_identical_envelope(
        ["facts", "--profile", str(bad_utf8)], "proto-bad-utf8"
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "Invalid UTF-8 in pprof string table.",
    }

    # R5-12 (verified non-divergence, pinned): bare CR in core-class CSV is a
    # record break through the real loaders in both languages.
    core_csv = tmp_path / "core.csv"
    core_csv.write_bytes(b'Weird\rArray\n"Quoted"\n')
    builder = PprofFixtureBuilder.create()
    leaf = builder.function("Array#map", "<cfunc>")
    parent = builder.function("Parent", "/app/parent.rb")
    builder.sample((builder.location(leaf), builder.location(parent)), 10)
    profile_path = tmp_path / "core-cr.pb"
    profile_path.write_bytes(builder.encode())
    config_path = tmp_path / "core-cr-config.json"
    config_path.write_text(json.dumps({"Parent": {}}), encoding="utf-8")
    output = _assert_identical_success(
        [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
            "--runtime",
            "ruby",
            "--core-classes",
            str(core_csv),
        ],
        "core-cr-record-break",
    )
    assert "Ruby Core (Native)" in output


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_regex_engine_edge_semantics_are_pinned(tmp_path: Path) -> None:
    # DOCUMENTED engine-native divergences, not bug pins (spec: regex dialect
    # passage): these patterns compile in BOTH engines but match per each
    # engine's own anchor/case-fold/Unicode-class semantics, outside the
    # parity guarantee. The pins exist to catch engine drift on either side.
    cases = [
        ("anchor", "/app/x\n", "regex:x$", "Matched", "Other"),
        ("casefold", "/İ", "regex:(?i)i", "Matched", "Other"),
        ("wordclass", "/́", "regex:\\w", "Other", "Matched"),
        ("spaceclass", "/\x1f", "regex:\\s", "Matched", "Other"),
    ]
    for label, filename, pattern, python_expected, rust_expected in cases:
        builder = PprofFixtureBuilder.create()
        leaf = builder.location(builder.function("Leaf", filename))
        parent = builder.location(builder.function("T", "/srv/app/t.py"))
        builder.sample((leaf, parent), 7)
        profile_path = tmp_path / f"regex-{label}.pb"
        profile_path.write_bytes(builder.encode())
        config_path = tmp_path / f"regex-{label}.json"
        config_path.write_text(
            json.dumps({"T": {"Matched": pattern}}), encoding="utf-8"
        )
        argv = [
            "targets",
            "--profile",
            str(profile_path),
            "--config",
            str(config_path),
        ]
        python_run = _run_python_cli_raw(argv)
        rust_run = _run_rust_cli_raw(argv)
        assert python_run.returncode == 0, (label, python_run.stderr)
        assert rust_run.returncode == 0, (label, rust_run.stderr)

        def category_names(stdout: str) -> list[str]:
            payload = cast(dict[str, Any], json.loads(stdout))
            parents = cast(dict[str, Any], payload["parents"])
            parent_payload = cast(dict[str, Any], parents["T"])
            categories = cast(list[dict[str, Any]], parent_payload["categories"])
            return [cast(str, item["name"]) for item in categories]

        assert category_names(python_run.stdout) == [python_expected], label
        assert category_names(rust_run.stdout) == [rust_expected], label


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_yaml_tags_match_python(tmp_path: Path) -> None:
    facts_path = _order_scope_facts(tmp_path)

    # Global tags are ignored per serde_yaml semantics in BOTH languages:
    # !!binary stays the base64 string, !!set stays the mapping — a
    # deterministic structure, never a hash-randomized Python set repr.
    global_slices = tmp_path / "tagged-global.yml"
    global_slices.write_text(
        "slices:\n"
        '  - name: app\n'
        '    paths: ["/srv/**"]\n'
        "    metadata:\n"
        "      blob: !!binary SGVsbG8=\n"
        "      members: !!set {alpha: null, beta: null, gamma: null, delta: null}\n",
        encoding="utf-8",
    )
    output = _assert_identical_success(
        ["slices", "--facts", str(facts_path), "--slices", str(global_slices)],
        "yaml-tags-global",
    )
    assert '"SGVsbG8="' in output
    assert '"alpha"' in output and '"delta"' in output

    # Local tags are rejected on every YAML surface with the shared message.
    local_tag_error = {
        "ok": False,
        "error": "YAML local tags are not supported in clankerprof inputs.",
    }
    local_slices = tmp_path / "tagged-local-slices.yml"
    local_slices.write_text(
        "slices:\n  - name: app\n    metadata:\n      x: !foo bar\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        ["slices", "--facts", str(facts_path), "--slices", str(local_slices)],
        "yaml-tags-local-slices",
    )
    assert json.loads(envelope) == local_tag_error

    local_scope = tmp_path / "tagged-local-scope.yml"
    local_scope.write_text(
        'cost_kind:\n  AppWork: !foo "name:Leaf"\nscope:\n  - function: T\n',
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        ["scopes", "--facts", str(facts_path), "--config", str(local_scope)],
        "yaml-tags-local-scope",
    )
    assert json.loads(envelope) == local_tag_error

    local_pack = tmp_path / "tagged-local-pack.yml"
    local_pack.write_text(
        "semantic_rules:\n  - category: !x Hit\n    name_contains: [Leaf]\n",
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        [
            "targets",
            "--facts",
            str(facts_path),
            "--target",
            "T",
            "--runtime-rules",
            str(local_pack),
        ],
        "yaml-tags-local-pack",
    )
    assert json.loads(envelope) == local_tag_error

    # Typed core tags apply the strict scalar grammars with serde's message
    # core (location suffix engine-specific).
    typed_config = tmp_path / "tagged-typed.yml"
    typed_config.write_text("profile: /dev/null\ntop: !!int 1_0\n", encoding="utf-8")
    argv = ["slices", "--config", str(typed_config)]
    python_run = _run_python_cli_raw(argv)
    rust_run = _run_rust_cli_raw(argv)
    core = 'invalid value: string "1_0", expected an integer'
    assert python_run.returncode == 2, python_run.stderr
    assert rust_run.returncode == 2, rust_run.stderr
    assert core in cast(str, json.loads(python_run.stderr)["error"])
    assert core in cast(str, json.loads(rust_run.stderr)["error"])


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_round6_cluster_s_matches_python(tmp_path: Path) -> None:
    # R6-01: the pseudo-slice names are reserved at validation in both
    # languages instead of being attributed and silently stripped at render.
    facts_builder = PprofFixtureBuilder.create()
    facts_builder.sample(
        (facts_builder.location(facts_builder.function("work", "/app/x.rb")),), 7
    )
    pseudo_profile = tmp_path / "pseudo.pb"
    pseudo_profile.write_bytes(facts_builder.encode())
    for reserved in ("(gc)", "(uncollapsible)"):
        pseudo_slices = tmp_path / "pseudo-slices.yml"
        pseudo_slices.write_text(
            f'slices:\n  - name: "{reserved}"\n    paths: [/app]\n',
            encoding="utf-8",
        )
        envelope = _assert_identical_envelope(
            [
                "slices",
                "--profile",
                str(pseudo_profile),
                "--slices",
                str(pseudo_slices),
            ],
            f"reserved-{reserved}",
        )
        assert json.loads(envelope) == {
            "ok": False,
            "error": (
                f"Slice config declares reserved pseudo-slice name: {reserved}. "
                "The names (gc) and (uncollapsible) are reserved for analyzer "
                "pseudo-outputs."
            ),
        }

    # R6-02: a filtered matching total that cancels to zero renders every
    # dependent percentage through the zero arm, never the whole-profile
    # total; byte-identical across languages.
    cancel_builder = PprofFixtureBuilder.create()
    cancel_builder.sample(
        (cancel_builder.location(cancel_builder.function("match_a", "/a/one.rb")),), 10
    )
    cancel_builder.sample(
        (cancel_builder.location(cancel_builder.function("match_b", "/b/two.rb")),), -10
    )
    cancel_builder.sample(
        (cancel_builder.location(cancel_builder.function("other_c", "/c/three.rb")),), 5
    )
    cancel_profile = tmp_path / "cancel.pb"
    cancel_profile.write_bytes(cancel_builder.encode())
    cancel_slices = tmp_path / "cancel-slices.yml"
    cancel_slices.write_text(
        "slices:\n"
        "  - name: A\n    paths: [/a/**]\n"
        "  - name: B\n    paths: [/b/**]\n"
        "  - name: default\n    default: true\n",
        encoding="utf-8",
    )
    cancelled = _assert_identical_success(
        [
            "slices",
            "--profile",
            str(cancel_profile),
            "--slices",
            str(cancel_slices),
            "--filter",
            "name:match",
        ],
        "zero-sum-filter",
    )
    payload = json.loads(cancelled)
    assert payload["summary"]["matching_time_ns"] == 0
    assert {item["name"]: (item["time_ns"], item["pct"]) for item in payload["slices"]} == {
        "A": (10, 0),
        "B": (-10, 0),
    }

    # R6-03: a bucket whose signed categories cancel to zero keeps its
    # nonzero category rows in both languages.
    bucket_builder = PprofFixtureBuilder.create()
    scope_loc = bucket_builder.location(bucket_builder.function("S", "/srv/app/s.py"))
    bucket_builder.sample(
        (bucket_builder.location(bucket_builder.function("Pos", "/srv/app/pos.py")), scope_loc),
        10,
    )
    bucket_builder.sample(
        (bucket_builder.location(bucket_builder.function("Neg", "/srv/app/neg.py")), scope_loc),
        -10,
    )
    bucket_builder.sample(
        (bucket_builder.location(bucket_builder.function("Unrelated", "/x/u.py")),), 5
    )
    bucket_profile = tmp_path / "cancel-bucket.pb"
    bucket_profile.write_bytes(bucket_builder.encode())
    bucket_config = tmp_path / "cancel-bucket.yml"
    bucket_config.write_text(
        'cost_kind:\n  Positive: "name:Pos"\n  Negative: "name:Neg"\n'
        "scope:\n  - function: S\n"
        '    rollup:\n      Work: ["Positive", "Negative"]\n',
        encoding="utf-8",
    )
    bucket_output = _assert_identical_success(
        ["scopes", "--profile", str(bucket_profile), "--config", str(bucket_config)],
        "zero-sum-bucket",
    )
    boundary = json.loads(bucket_output)["boundaries"][0]
    assert [bucket["name"] for bucket in boundary["buckets"]] == ["Work"]
    assert {
        category["name"]: category["time_ns"]
        for category in boundary["buckets"][0]["categories"]
    } == {"Positive": 10, "Negative": -10}

    # R6-04: the rollup name `Other` is reserved (the renderer appends the
    # implicit leftover bucket under that name), identical envelopes.
    other_config = tmp_path / "other.yml"
    other_config.write_text(
        'cost_kind:\n  Positive: "name:Pos"\n'
        "scope:\n  - function: S\n"
        '    rollup:\n      Other: ["Positive"]\n',
        encoding="utf-8",
    )
    envelope = _assert_identical_envelope(
        ["scopes", "--profile", str(bucket_profile), "--config", str(other_config)],
        "reserved-other",
    )
    assert json.loads(envelope) == {
        "ok": False,
        "error": "scope.rollup name 'Other' is reserved for unbucketed cost kinds.",
    }
