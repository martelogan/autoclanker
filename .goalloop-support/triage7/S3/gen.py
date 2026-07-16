import json
import sys

sys.path.insert(0, ".")
from tests.fixtures.pprof_builder import PprofFixtureBuilder

d = ".goalloop-support/triage7/S3"

b = PprofFixtureBuilder.create()
t = b.location(b.function("T", "/srv/t.py"))
leaf = b.location(b.function("Leaf", "/app/app"))
b.sample((leaf, t), 7)
open(f"{d}/profile1.pb", "wb").write(b.encode())
json.dump({"T": {"Hit": "regex:(?P<x>/app)\\1"}}, open(f"{d}/cfg1.json", "w"))

b2 = PprofFixtureBuilder.create()
t2 = b2.location(b2.function("T", "/srv/t.py"))
leaf2 = b2.location(b2.function("Leaf", "/app/a"))
b2.sample((leaf2, t2), 7)
open(f"{d}/profile2.pb", "wb").write(b2.encode())
json.dump({"T": {"Hit": "regex:(?P<x>/app)/a(?(x)|x)"}}, open(f"{d}/cfg2.json", "w"))
