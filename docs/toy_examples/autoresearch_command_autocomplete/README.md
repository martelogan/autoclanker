# Autoresearch Token Kernel

This toy codebase is a readable mirror of the backing `autoresearch` live exercise.

Read it in this order:

1. `app.py`
2. `variants/`
3. `benchmark.py`

What the app is doing:

- `app.py` is a tiny shell-history autocomplete helper
- it looks at recent commands and suggests completions for prefixes like `git p`
- lower `val_bpb` is better
- `peak_vram_gb` is the resource constraint

Why this example fits autoresearch-style search:

- each good change mostly helps for its own reason
- there is no deep multi-way interaction you must discover before seeing progress
- a greedy or hill-climbing search is therefore a natural fit

What each variant means:

- `variants/baseline.py`: the default profile
- `variants/optimized.py`: combine several individually helpful constant edits
- `variants/failure_variant.py`: push the profile into a tempting but OOM-prone regime

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/benchmark.py --variant optimized
```

How this maps to `autoclanker`:

- the toy app is only the human-readable mirror
- the real backing demo lives in `examples/live_exercises/autoresearch_simple/`
- the probeable adapter config is `examples/live_exercises/autoresearch_simple/adapter.local.yaml`
- the shipped replay stays deterministic and local for testability
- in a real deployment, the upstream autoresearch loop can still be LLM-backed

Replay the real method-backed demo:

```bash
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/autoresearch_simple/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase autoresearch_command_autocomplete
```
