# cEvolve Partition Synergy

This toy codebase is a readable mirror of the backing `cevolve` live exercise.

Read it in this order:

1. `app.py`
2. `variants/`
3. `benchmark.py`

What the app is doing:

- `app.py` is a tiny integer-sorting program
- it actually sorts a few small lists with configurable quicksort-style choices
- lower `time_ms` is better
- the point is that the best result depends on a compatible combination of choices

Why this example fits cEvolve-style search:

- single local edits help a little, but they are not the whole story
- the strongest payoff only appears when several good choices arrive together
- that makes combination-heavy evolutionary search a natural fit

What each variant means:

- `variants/baseline.py`: the default sorting strategy
- `variants/single_threshold.py`: only improve the insertion threshold
- `variants/single_partition.py`: only improve the partition scheme
- `variants/optimized.py`: combine the compatible threshold, partition, and iterative changes

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/benchmark.py --variant optimized
```

How this maps to `autoclanker`:

- the toy app is only the human-readable mirror
- the real backing demo lives in `examples/live_exercises/cevolve_synergy/`
- the probeable adapter config is `examples/live_exercises/cevolve_synergy/adapter.local.yaml`
- the toy app does not call an LLM; it exists to make the interaction-heavy search landscape easy to see

Replay the real method-backed demo:

```bash
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/cevolve_synergy/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase cevolve_sort_partition
```
