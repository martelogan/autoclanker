"""Build K3 zero-total target fixtures (signed samples cancelling under parent T)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.fixtures.pprof_builder import PprofFixtureBuilder

OUT = Path(__file__).resolve().parent

# Fixture A: parent T with two leaves in distinct categories: +10 and -10.
builder = PprofFixtureBuilder.create()
t_fn = builder.function("T", "/app/t.py")
leaf_fn = builder.function("Leaf", "/srv/a/a.py")
neg_fn = builder.function("Neg", "/srv/b/b.py")
builder.sample((builder.location(leaf_fn), builder.location(t_fn)), 10)
builder.sample((builder.location(neg_fn), builder.location(t_fn)), -10)
(OUT / "zero_total.pb").write_bytes(builder.encode())

# Fixture B: parent T with a single zero-valued sample.
builder2 = PprofFixtureBuilder.create()
t2_fn = builder2.function("T", "/app/t.py")
leaf2_fn = builder2.function("Leaf", "/srv/a/a.py")
builder2.sample((builder2.location(leaf2_fn), builder2.location(t2_fn)), 0)
(OUT / "zero_sample.pb").write_bytes(builder2.encode())

# Fixture C: the parity test's own fixture (no parent T anywhere).
builder3 = PprofFixtureBuilder.create()
pos_fn = builder3.function("pos", "/srv/a/a.py")
neg3_fn = builder3.function("neg", "/srv/b/b.py")
builder3.sample((builder3.location(pos_fn),), 10)
builder3.sample((builder3.location(neg3_fn),), -10)
(OUT / "parity_zero_total.pb").write_bytes(builder3.encode())

(OUT / "targets_two_cats.json").write_text(
    '{"T": {"Pos": "path:/srv/a", "Neg": "path:/srv/b"}}', encoding="utf-8"
)
(OUT / "targets_parity.json").write_text('{"T": {"App": "path:/srv"}}', encoding="utf-8")
print("built", sorted(p.name for p in OUT.iterdir()))
