# Companion Toy Examples

These examples complement the live exercises with small, readable toy codebases.

Each toy example has the same primary structure:

- `app.py`: the thing being optimized
- `benchmark.py`: the deterministic scoring harness
- `variants/`: named baseline and optimized choices

Important distinction:

- running `app.py --variant ...` shows the toy program's behavior on a sample input;
- running `benchmark.py --variant ...` or a file in `variants/` only measures a readable toy snapshot;
- those files do not run `autoclanker`;
- the actual method-backed demo is replayed with `scripts/showcase/replay_backing_exercise.py`.

Two ways to use this directory:

- inspect the standalone toy app and its variants directly;
- replay the actual backing `autoclanker` demo that produced the story.

Suggested flow for each toy example:

1. Run `app.py --variant baseline` to understand the toy app itself.
2. Run `benchmark.py --variant ...` to see the optimization metric change.
3. Run `scripts/showcase/replay_backing_exercise.py --showcase ...` to replay the real adapter/session-backed demo.

Run the summary script to see the code-level outcomes side by side:

```bash
./bin/dev exec -- python scripts/showcase/run_toy_examples.py
```

Replay a backing demo:

```bash
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```

If you run one of the toy files directly, its JSON output tells you:

- what the toy app itself does,
- what metric is being optimized,
- what changed in that variant,
- which live exercise it mirrors,
- which replay command to run for the real demo.

The three toy-example families are:

1. `autoresearch_command_autocomplete`
   Tiny command autocomplete helper. Mostly additive tuning. This is the hill-climbing / simple-priors case.

2. `cevolve_sort_partition`
   Tiny integer sorter. Interaction-heavy algorithm tuning. This is the evolutionary-combination case.

3. `bayes_pair_feature_trainer`
   Tiny numeric trainer. Beliefs justify an unseen but better pairwise change.
   This is a secondary companion story, not the main Bayes quickstart.

See [docs/TOY_EXAMPLES.md](../../docs/TOY_EXAMPLES.md) for the walkthrough.
