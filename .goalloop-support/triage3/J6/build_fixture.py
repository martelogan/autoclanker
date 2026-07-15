"""Build J6 reproduction inputs: three profiles for slices/targets/rules divergence."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage3" / "J6"

# Case 1: slices — leaf on /srv/123/file.py so a coerced "123" pattern matches.
b1 = PprofFixtureBuilder.create()
work = b1.location(b1.function("work", "/srv/123/file.py"))
main = b1.location(b1.function("main", "/srv/app/main.py"))
b1.sample((work, main), 7)
(OUT / "profile_slices.pb").write_bytes(b1.encode())

# Case 2: targets — leaf on /srv/123/leaf.py under parent T so coerced "123" matches.
b2 = PprofFixtureBuilder.create()
leaf = b2.location(b2.function("Leaf", "/srv/123/leaf.py"))
target = b2.location(b2.function("T", "/srv/app/t.py"))
b2.sample((leaf, target), 7)
(OUT / "profile_targets.pb").write_bytes(b2.encode())

# Case 3: rules — native leaf named NoneWork on <native> pseudo-path under parent T.
b3 = PprofFixtureBuilder.create()
nleaf = b3.location(b3.function("NoneWork", "<cfunc>"))
t3 = b3.location(b3.function("T", "/srv/app/t.py"))
b3.sample((nleaf, t3), 7)
(OUT / "profile_rules.pb").write_bytes(b3.encode())

print("wrote profiles under", OUT)
