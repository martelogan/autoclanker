"""Build real pprof profiles matching the auditor's F3 scenario."""

from __future__ import annotations

import pathlib
import sys

WORKTREE = pathlib.Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(WORKTREE))

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = WORKTREE / ".goalloop-support" / "triage" / "F3"


def build(f_one_ns: int, f_two_ns: int, g_ns: int, other_ns: int) -> bytes:
    builder = PprofFixtureBuilder.create()
    one = builder.location(builder.function("f", "/one.rb"))
    two = builder.location(builder.function("f", "/two.rb"))
    g = builder.location(builder.function("g", "/g.rb"))
    other = builder.location(builder.function("other", "/other.rb"))
    builder.sample((one,), f_one_ns)
    builder.sample((two,), f_two_ns)
    if g_ns:
        builder.sample((g,), g_ns)
    builder.sample((other,), other_ns)
    return builder.encode()


# total 100ms each; "slice" is the default (all) slice at 30% ... actually
# f/one + f/two + g = 30ms of 100ms. Everything lands in the default slice,
# so pct values match the auditor's per-frame numbers exactly.
(OUT / "before.pb").write_bytes(build(10_000_000, 15_000_000, 5_000_000, 70_000_000))
(OUT / "after.pb").write_bytes(build(15_000_000, 15_000_000, 0, 70_000_000))
print("wrote before.pb and after.pb")
