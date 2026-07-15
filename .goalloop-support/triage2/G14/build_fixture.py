"""Build the G14 reproduction profile: one sample with stack Leaf -> ZTarget -> ATarget."""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.pprof_builder import PprofFixtureBuilder

builder = PprofFixtureBuilder.create()
leaf = builder.location(builder.function("Leaf", "/srv/app/leaf.py"))
ztarget = builder.location(builder.function("ZTarget", "/srv/app/ztarget.py"))
atarget = builder.location(builder.function("ATarget", "/srv/app/atarget.py"))
# Leaf-to-root: Leaf, ZTarget, ATarget
builder.sample((leaf, ztarget, atarget), 10_000_000)

out = Path(__file__).parent / "profile.pb"
out.write_bytes(builder.encode())
print(f"wrote {out}")
