import sys

sys.path.insert(0, ".")
from tests.fixtures.pprof_builder import PprofFixtureBuilder

d = ".goalloop-support/triage7/S4"
b = PprofFixtureBuilder.create()
fa = b.location(b.function("FuncA", "/a/x.py"))
fb = b.location(b.function("FuncB", "/b/x.py"))
b.sample((fa,), 10)
b.sample((fb,), -10)
open(f"{d}/profile.pb", "wb").write(b.encode())
open(f"{d}/sl.yml", "w").write(
    "slices:\n"
    "  - name: A\n    paths:\n      - \"/a/*\"\n"
    "  - name: B\n    paths:\n      - \"/b/*\"\n"
)
