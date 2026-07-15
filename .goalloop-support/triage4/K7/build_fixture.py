"""Build K7 reproduction profiles: Leaf, Native1..NativeN, AppCaller, T (leaf-to-root)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/Users/logan_martel/Projects/autoclanker_bootstrap_v5/clankerprof-v2-audit")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.pprof_builder import PprofFixtureBuilder  # noqa: E402

OUT = ROOT / ".goalloop-support" / "triage4" / "K7"


def build(n_native: int) -> None:
    builder = PprofFixtureBuilder.create()
    leaf = builder.location(builder.function("Leaf", "<native>"))
    natives = [
        builder.location(builder.function(f"Native{i}", "<native>"))
        for i in range(1, n_native + 1)
    ]
    app_caller = builder.location(builder.function("AppCaller", "/app/caller.py"))
    target = builder.location(builder.function("T", "/app/t.py"))
    builder.sample((leaf, *natives, app_caller, target), 100)
    path = OUT / f"profile_n{n_native}.pb"
    path.write_bytes(builder.encode())
    print("wrote", path)


build(8)
build(9)
