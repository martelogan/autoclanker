import json, sys
sys.path.insert(0, "tests/fixtures")
from pprof_builder import PprofFixtureBuilder
b = PprofFixtureBuilder.create()
t = b.function("T", "/app/t.py")
pos = b.function("PosLeaf", "/srv/app/pos.py")
neg = b.function("NegLeaf", "/srv/app/neg.py")
tl = b.location(t)
b.sample((b.location(pos), tl), 100000)
b.sample((b.location(neg), tl), -10)
open(".goalloop-support/triage7/S5/profile2.pb","wb").write(b.encode())
