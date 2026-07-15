"""Build the J3 reproduction: two samples (110, -10) under one Scope frame."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage3" / "J3"

builder = PprofFixtureBuilder.create()
scope_fn = builder.location(builder.function("ScopeFn", "/srv/app/scope.py"))
pos_leaf = builder.location(builder.function("PosLeaf", "/srv/app/pos.py"))
neg_leaf = builder.location(builder.function("NegLeaf", "/srv/app/neg.py"))
builder.sample((pos_leaf, scope_fn), 110)
builder.sample((neg_leaf, scope_fn), -10)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
