import sys

sys.path.insert(0, ".")
from tests.fixtures.pprof_builder import PprofFixtureBuilder

b = PprofFixtureBuilder.create()
leaf = b.location(b.function("Leaf", "/app/leaf.py"))
b.sample((leaf,), 10)
open(".goalloop-support/triage7/S1/profile.pb", "wb").write(b.encode())
open(".goalloop-support/triage7/S1/slices.yml", "w").write(
    "slices:\n  - name: App\n    paths: ['/app/*']\n"
)
