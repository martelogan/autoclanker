import sys
sys.path.insert(0, "tests")
from fixtures.pprof_builder import PprofFixtureBuilder

def make(ab_value: int, path: str) -> None:
    b = PprofFixtureBuilder.create()
    fa = b.function("fa", "a.rb"); fb = b.function("fb", "b.rb"); fab = b.function("fab", "ab.rb"); fo = b.function("fo", "other.rb")
    la = b.location(fa); lb = b.location(fb); lab = b.location(fab); lo = b.location(fo)
    b.sample((la,), 1)
    b.sample((lb,), 1)
    b.sample((lab,), ab_value)
    b.sample((lo,), 100 - 2 - ab_value)
    open(path, "wb").write(b.encode())

make(10, ".goalloop-support/triage10/W1/before.pb")
make(20, ".goalloop-support/triage10/W1/after.pb")
