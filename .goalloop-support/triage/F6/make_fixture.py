"""Build a one-sample pprof profile fixture for F6 triage (scratch only)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = ROOT / ".goalloop-support" / "triage" / "F6"

builder = PprofFixtureBuilder.create()
child = builder.location(builder.function("Child#work", "/srv/app/child.py"))
target = builder.location(builder.function("T", "/srv/app/t.py"))
builder.sample((child, target), 10_000_000)
(OUT / "profile.pb").write_bytes(builder.encode(gzipped=False))
print("wrote", OUT / "profile.pb")
