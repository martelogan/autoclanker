# cEvolve Synergy

This exercise is meant to reward combinations, not isolated single moves.

If you want the smallest starter input first, begin with:

- `examples/idea_inputs/cevolve_synergy.json`
- `./bin/dev exec -- autoclanker beliefs preview --input examples/idea_inputs/cevolve_synergy.json --adapter-config examples/live_exercises/cevolve_synergy/adapter.local.yaml`

Use `examples/idea_inputs/cevolve_synergy.yaml` if you want the commented teaching version.

The target exposes four knobs:

- `INSERTION_THRESHOLD`
- `PARTITION_SCHEME`
- `PIVOT_STRATEGY`
- `USE_ITERATIVE`

The deterministic cost model is shaped so that:

- `INSERTION_THRESHOLD=32` helps,
- `PARTITION_SCHEME="hoare"` helps,
- `USE_ITERATIVE=True` barely helps on its own,
- the real gain appears when those ideas are combined.

That makes this a good exercise for:

- validating the real `cevolve` session runner,
- showing why evolutionary search can outperform greedy one-change-at-a-time updates,
- demonstrating how Bayesian pair priors can reinforce the right interaction.

Minimum required files:

- `examples/idea_inputs/cevolve_synergy.json`: recommended front-door idea file
- `beliefs.yaml`: richer typed belief file used by the deeper manual workflow
- `adapter.local.yaml`: checkout-backed exercise config for the real upstream smoke lane
- provisioned upstream repo from `./bin/dev test-upstream-live`

Optional files:

- `train.py`: explanatory target only
- `expected_outcome.json`: replay assertion data only

Lowest-cruft replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy
```

For normal non-exercise usage, the first-party adapter does not require a checkout
path. You can use `mode: auto` with `python_module` or `command` instead.

Allowed idea inputs for this exercise:

| `gene_id` | Allowed `state_id` values | Plain-language meaning |
| --- | --- | --- |
| `sort.threshold` | `threshold_16`, `threshold_32`, `threshold_64` | Small, medium, or large insertion-sort threshold |
| `sort.partition` | `partition_lomuto`, `partition_hoare` | Lomuto or Hoare partitioning |
| `sort.pivot` | `pivot_median_of_three`, `pivot_middle`, `pivot_random` | Default, middle, or random pivot strategy |
| `sort.iterative` | `iterative_off`, `iterative_on` | Recursive or iterative quicksort |

What the starter ideas are saying in plain English:

- `ce1`: use threshold 32
- `ce2`: Hoare partitioning usually helps
- `ce3`: threshold 32 and Hoare together are the real win
- `ce4`: avoid random pivots on this workload

Which belief fields are free-form vs bounded:

- free-form: `id`, `author`, `rationale`
- bounded enums / ranges: `kind`, `confidence_level`, `evidence_sources`, `effect_strength`, `relation`, `constraint_type`, `severity`
- strict registry identifiers: `gene_id`, `state_id`

For the exact bounds, see [`docs/BELIEF_INPUT_REFERENCE.md`](../../../docs/BELIEF_INPUT_REFERENCE.md).

How to read the lowest-cruft replay:

- it previews the ideas against the live cevolve registry,
- creates a session,
- applies the preview digest automatically,
- and runs `session suggest` so you can see the synergistic combination rise to the top.

Longer manual path:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- autoclanker beliefs preview \
  --input examples/live_exercises/cevolve_synergy/beliefs.yaml \
  --adapter-config examples/live_exercises/cevolve_synergy/adapter.local.yaml
./bin/dev exec -- autoclanker session init \
  --beliefs-input examples/live_exercises/cevolve_synergy/beliefs.yaml \
  --adapter-config examples/live_exercises/cevolve_synergy/adapter.local.yaml \
  --session-root .autoclanker-exercises
```

Then apply the preview digest and run `session suggest`. If you do not want to wire the digest manually, use `replay_ideas_demo.py`.

Expected outcome:

- the improved candidate in `expected_outcome.json` beats baseline,
- it also beats the named single-change candidates by a visible margin.
