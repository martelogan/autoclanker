# Examples

Start here if you are browsing the repository for runnable demos.

## Start Here

These are the primary usage examples. They are the best places to learn what
running `autoclanker` actually looks like.

| Demo | Best for | Start with | First command |
| --- | --- | --- | --- |
| `bayes_quickstart` | the clearest first run; parser-based Bayes guidance | `examples/idea_inputs/minimal.json` or `examples/idea_inputs/bayes_quickstart.json` | `autoclanker beliefs preview --input examples/idea_inputs/minimal.json --era-id era_log_parser_v1` |
| `autoresearch_simple` | real-upstream `autoresearch` contrast; mostly additive search | `examples/idea_inputs/autoresearch_simple.json` | `./bin/dev test-upstream-live` |
| `cevolve_synergy` | real-upstream `cevolve` contrast; interaction-heavy search | `examples/idea_inputs/cevolve_synergy.json` | `./bin/dev test-upstream-live` |
| `bayes_complex` | advanced Bayes beliefs, decay, and feasibility | `examples/live_exercises/bayes_complex/README.md` | follow the manual workflow in `docs/LIVE_EXERCISES.md` |

The `bayes_quickstart` row is the main onboarding path. The `autoresearch` and
`cevolve` rows are contrast exercises on different workloads, not the same demo
app with different engines swapped in.

The quickest repo-native replay commands are:

```bash
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy
```

For the exact field bounds and the difference between free-form text and strict
registry identifiers, see [../docs/BELIEF_INPUT_REFERENCE.md](../docs/BELIEF_INPUT_REFERENCE.md).

If you want the explicit multi-path frontier shape rather than the smaller idea
inputs, start from [`frontiers/parser_frontier.json`](frontiers/parser_frontier.json)
and compare it with `autoclanker session suggest --candidates-input ...` or
`autoclanker session run-frontier --frontier-input ...`.

## Secondary Toy Examples

`docs/toy_examples/` is intentionally secondary.

Those files are readable toy examples of the same optimization stories, but they
do not run `autoclanker` themselves. Use them after the live exercises if you
want a code-level companion view:

- `docs/toy_examples/autoresearch_command_autocomplete/`
- `docs/toy_examples/cevolve_sort_partition/`
- `docs/toy_examples/bayes_pair_feature_trainer/`

See [../docs/TOY_EXAMPLES.md](../docs/TOY_EXAMPLES.md) for that layer.
