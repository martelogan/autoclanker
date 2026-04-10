from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path
from typing import cast

import pytest

from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError("Expected a JSON object.")
    return cast(dict[str, object], value)


def _require_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise AssertionError(f"Expected string field {key!r}.")
    return value


@covers("M5-LIVE-001", "M5-LIVE-002")
@pytest.mark.upstream_live
@pytest.mark.parametrize(
    ("exercise", "expected_gene", "expected_state"),
    (
        ("autoresearch_simple", "train.depth", "depth_10"),
        ("cevolve_synergy", "sort.partition", "partition_hoare"),
    ),
)
def test_live_ideas_demo_short_replays_run(
    exercise: str,
    expected_gene: str,
    expected_state: str,
) -> None:
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "live" / "replay_ideas_demo.py"),
            "--exercise",
            exercise,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _require_mapping(json.loads(process.stdout))
    observed = _require_mapping(payload["observed"])
    top_genotype = _require_mapping(observed["top_genotype"])

    assert payload["exercise"] == exercise
    assert _require_string(observed, "top_candidate").startswith("cand_auto_")
    assert _require_string(top_genotype, expected_gene) == expected_state
