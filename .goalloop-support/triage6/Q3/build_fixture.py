"""Build the Q3 reproduction profile: scope S with Positive +10 / Negative -10
(zero-sum bucket 'Work') plus an unrelated +5 sample."""

import sys
from pathlib import Path

sys.path.insert(
    0, "/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit"
)
from tests.fixtures.pprof_builder import PprofFixtureBuilder

builder = PprofFixtureBuilder.create()
scope = builder.location(builder.function("S", "/srv/app/s.py"))
pos = builder.location(builder.function("Pos", "/srv/app/pos.py"))
neg = builder.location(builder.function("Neg", "/srv/app/neg.py"))
other = builder.location(builder.function("Unrelated", "/srv/app/u.py"))
builder.sample((pos, scope), 10)
builder.sample((neg, scope), -10)
builder.sample((other,), 5)
out = Path(__file__).with_name("profile.pb")
out.write_bytes(builder.encode())
print(f"wrote {out}")
