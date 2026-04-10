# Companion Toy Examples

These are companion examples to the live exercises.

If you are trying to learn the actual `autoclanker` CLI or the recommended
first demos, do not start here. Start with:

- `examples/live_exercises/bayes_quickstart/`
- `examples/live_exercises/autoresearch_simple/`
- `examples/live_exercises/cevolve_synergy/`
- `examples/README.md`

Important distinction:

- the files under `docs/toy_examples/` are readable toy apps;
- the files under `examples/live_exercises/` are the actual `autoclanker`-backed demos;
- the primary toy-example layout is `app.py`, `benchmark.py`, and `variants/`;
- running `app.py --variant ...` shows the toy app behavior on a sample input;
- running `benchmark.py --variant ...` or a file in `variants/` only measures one toy snapshot;
- those toy files do not invoke `autoclanker` or the live adapters;
- to replay the actual method-backed demo, use `scripts/showcase/replay_backing_exercise.py`.

Two useful entrypoints:

```bash
./bin/dev exec -- python scripts/showcase/run_toy_examples.py
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```

The first command summarizes the readable toy-app outcomes.
The second command replays the real backing demo for one toy example.

If you run a toy file directly, its JSON tells you:

- what the toy app itself does
- what metric is being optimized
- what changed in that variant
- which live exercise it mirrors
- which replay command demonstrates the real method-backed run

## Autoresearch Command Autocomplete

Primary files:

- `docs/toy_examples/autoresearch_command_autocomplete/app.py`
- `docs/toy_examples/autoresearch_command_autocomplete/benchmark.py`
- `docs/toy_examples/autoresearch_command_autocomplete/variants/`

Toy app:

- a tiny shell-history autocomplete helper
- it reads recent commands and suggests completions for a few prefixes
- lower `val_bpb` is better
- `peak_vram_gb` is the resource constraint

Why this method fits:

- `app.py` shows a mostly additive search space
- each good constant change helps for its own reason
- hill-climbing and simple priors are therefore a natural fit

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/autoresearch_command_autocomplete/benchmark.py --variant optimized
```

Real demo replay:

```bash
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/autoresearch_simple/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase autoresearch_command_autocomplete
```

Important note:

- the self-contained replay does not make remote model calls
- in a real deployment, the upstream autoresearch loop can still be LLM-backed

## cEvolve Sort Partition

Primary files:

- `docs/toy_examples/cevolve_sort_partition/app.py`
- `docs/toy_examples/cevolve_sort_partition/benchmark.py`
- `docs/toy_examples/cevolve_sort_partition/variants/`

Toy app:

- a tiny integer-sorting program
- it actually sorts a few small lists using configurable quicksort-style choices
- lower `time_ms` is better
- the best result depends on combining multiple compatible choices

Why this method fits:

- `app.py` shows the interaction-heavy cost model directly
- one-off edits help, but not enough
- the strongest payoff only appears when several good changes arrive together

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/cevolve_sort_partition/benchmark.py --variant optimized
```

Real demo replay:

```bash
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/cevolve_synergy/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase cevolve_sort_partition
```

## Bayes Pair-Feature Trainer

Primary files:

- `docs/toy_examples/bayes_pair_feature_trainer/app.py`
- `docs/toy_examples/bayes_pair_feature_trainer/benchmark.py`
- `docs/toy_examples/bayes_pair_feature_trainer/variants/`

Toy app:

- a tiny numeric trainer over a few labeled samples
- it can optionally include an interaction feature for the model
- higher `utility` is better
- `oom` is the explicit failure mode to avoid
- the best unseen pair is better than the best already-observed single move

Why this method fits:

- `app.py` shows the good unseen pair and the risky branch directly
- a structured prior can justify trying the unseen pair
- feasibility beliefs matter because one branch is clearly dangerous

Important distinction:

- this toy example is not the beginner Bayes parser demo;
- it exists only to make the "better unseen pair" Bayes story easy to read in code;
- the primary Bayes onboarding example is still `examples/live_exercises/bayes_quickstart/`.

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/benchmark.py --variant belief_guided
```

Real demo replay:

```bash
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```

If you want the raw CLI sequence instead of the helper replay script:

```bash
./bin/dev exec -- autoclanker session init \
  --session-id bayes-showcase \
  --beliefs-input examples/live_exercises/bayes_complex/beliefs.yaml \
  --session-root .autoclanker-exercises

./bin/dev exec -- autoclanker session apply-beliefs \
  --session-id bayes-showcase \
  --preview-digest <preview_digest_from_init> \
  --session-root .autoclanker-exercises

./bin/dev exec -- autoclanker session suggest \
  --session-id bayes-showcase \
  --candidates-input examples/live_exercises/bayes_complex/candidates.json \
  --session-root .autoclanker-exercises
```

Important note:

- the toy app itself does not call an LLM
- the real `autoclanker` session flow is where human or LLM-authored beliefs enter

If you want the simplest live Bayes demo for ordinary human-authored ideas, use
`examples/live_exercises/bayes_quickstart/` before the more advanced
`examples/live_exercises/bayes_complex/`.
