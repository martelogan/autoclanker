import sys
sys.path.insert(0, "tests/fixtures")
from pprof_builder import PprofFixtureBuilder
b = PprofFixtureBuilder.create()
leaf = b.function("Leaf", "/else/leaf.py")
b.sample((b.location(leaf),), 10)
open(".goalloop-support/triage7/S7/profile.pb","wb").write(b.encode())
