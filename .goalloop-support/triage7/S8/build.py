import gzip, sys
sys.path.insert(0, "tests/fixtures")
from pprof_builder import PprofFixtureBuilder
b = PprofFixtureBuilder.create()
leaf = b.function("Leaf", "/app/leaf.py")
b.sample((b.location(leaf),), 7)
raw = b.encode()
open(".goalloop-support/triage7/S8/padded.pb.gz","wb").write(gzip.compress(raw) + b"\x00"*16)
