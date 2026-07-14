"""Real-profile variant where the overwrite fully masks the regression."""

from __future__ import annotations

import pathlib
import sys

WORKTREE = pathlib.Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(WORKTREE))

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = WORKTREE / ".goalloop-support" / "triage" / "F3"


def build(f_one_ns: int, f_two_ns: int, other_ns: int) -> bytes:
    builder = PprofFixtureBuilder.create()
    one = builder.location(builder.function("f", "/one.rb"))
    two = builder.location(builder.function("f", "/two.rb"))
    other = builder.location(builder.function("other", "/other.rb"))
    builder.sample((one,), f_one_ns)
    builder.sample((two,), f_two_ns)
    builder.sample((other,), other_ns)
    return builder.encode()


# f aggregate regresses 25% -> 30%; the growth is in the larger duplicate
# (f@/one), which the descending-time sort places first, so the smaller
# stable duplicate (f@/two) overwrites it in the compare map both times.
(OUT / "mask_before.pb").write_bytes(build(15_000_000, 10_000_000, 75_000_000))
(OUT / "mask_after.pb").write_bytes(build(20_000_000, 10_000_000, 70_000_000))
print("wrote mask_before.pb and mask_after.pb")
