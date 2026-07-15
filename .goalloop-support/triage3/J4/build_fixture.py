"""Build the J4 reproduction inputs: two large valid int64 samples in slices a/b."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage3" / "J4"

A_VALUE = 4813636180488882346
B_VALUE = 3969268729998903787
TOTAL = A_VALUE + B_VALUE

builder = PprofFixtureBuilder.create()
loc_a = builder.location(builder.function("A", "/srv/a/a.py"))
loc_b = builder.location(builder.function("B", "/srv/b/b.py"))
builder.sample((loc_a,), A_VALUE)
builder.sample((loc_b,), B_VALUE)
(OUT / "profile.pb").write_bytes(builder.encode())
print("wrote", OUT / "profile.pb")
print("total:", TOTAL, "u64::MAX:", 2**64 - 1, "in range:", TOTAL <= 2**64 - 1)
print("i64::MAX:", 2**63 - 1, "values fit i64:", A_VALUE <= 2**63 - 1 and B_VALUE <= 2**63 - 1)

# Arithmetic pre-check: exact int/int true division (Python) vs operand-rounded (Rust).
py_pct = A_VALUE / TOTAL * 100
rs_pct = float(A_VALUE) / float(TOTAL) * 100.0
print("python-style pct:", repr(py_pct))
print("rust-style pct:  ", repr(rs_pct))
print("bit-identical:", py_pct == rs_pct)
