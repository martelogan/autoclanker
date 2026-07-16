import sys
sys.path.insert(0, "tests/fixtures")
from pprof_builder import PprofFixtureBuilder
b = PprofFixtureBuilder.create()
s = b.function("S", "/app/s.py")
blk = b.function("Block", "/app/block.py")
b.sample((b.location(blk), b.location(s)), 10)
open(".goalloop-support/triage7/S6/profile.pb","wb").write(b.encode())
