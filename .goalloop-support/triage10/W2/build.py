import sys
sys.path.insert(0, "tests")
from fixtures.pprof_builder import PprofFixtureBuilder
b = PprofFixtureBuilder.create()
f = b.function("leaf", "leaf.rb")
l = b.location(f)
b.sample((l,), 7)
open(".goalloop-support/triage10/W2/one.pb", "wb").write(b.encode())
