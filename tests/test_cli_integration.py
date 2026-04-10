from __future__ import annotations

import json
import subprocess
import sys

from pathlib import Path

import pytest

from tests.compliance import covers

ROOT = Path(__file__).resolve().parents[1]


@covers("M0-002")
@pytest.mark.integration
def test_cli_module_entrypoint_validates_beliefs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "autoclanker.cli",
            "beliefs",
            "validate",
            "--input",
            str(ROOT / "examples/human_beliefs/basic_session.yaml"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["belief_count"] == 4
    assert completed.stderr == ""
