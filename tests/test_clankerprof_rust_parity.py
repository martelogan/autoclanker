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
from clankerprof.compare import CompareOptions, compare_slice_json
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


@pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not installed")
def test_clankerprof_rust_targets_match_python_generic_projection(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile_bytes = _profile_bytes(gzipped=False)
    profile_path = tmp_path / "profile.pb"
    config_path = tmp_path / "targets.json"
    profile_path.write_bytes(profile_bytes)
    config = {
        "RequestHandler#render_response": {
            "Components": "path:srv/app/components",
            "Cache Client": "library:cache-client",
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
    assert rust_payload == python_payload


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
