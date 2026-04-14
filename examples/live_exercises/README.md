# Live Exercise Cases

These exercises are the primary runnable demos for `autoclanker`.
If you only click one thing first, click `bayes_quickstart`.

The two upstream exercises use checkout-backed `adapter.local.yaml` files because
they are real-upstream smoke demos. Outside these exercises, the first-party
adapters can also use `mode: auto` with an installed adapter module or runnable
adapter command instead of a checkout path.

## Fastest starting points

| Exercise | Minimum files | Lowest-cruft command |
| --- | --- | --- |
| `bayes_quickstart` | `examples/idea_inputs/minimal.json` or `examples/idea_inputs/bayes_quickstart.json` | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart` |
| `autoresearch_simple` | `examples/idea_inputs/autoresearch_simple.json`, `adapter.local.yaml`, plus `./bin/dev test-upstream-live` once | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple` |
| `cevolve_synergy` | `examples/idea_inputs/cevolve_synergy.json`, `adapter.local.yaml`, plus `./bin/dev test-upstream-live` once | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy` |
| `bayes_complex` | `beliefs.yaml`, `candidates.json` | see [docs/LIVE_EXERCISES.md](../../docs/LIVE_EXERCISES.md) |

For exact field bounds and which fields are human-language versus strict enums, see
[docs/BELIEF_INPUT_REFERENCE.md](../../docs/BELIEF_INPUT_REFERENCE.md).

1. `bayes_quickstart`
   Uses the native `autoclanker` fixture registry plus only the beginner belief kinds:
   `idea`, `relation`, and `proposal`.
   This is the easiest place to see how rough optimization ideas can already move the
   cold-start ranking.

2. `autoresearch_simple`
   Uses the real `karpathy/autoresearch` checkout and a lightweight upstream-anchored evaluator over real `train.py` knobs.
   This is a mostly main-effect landscape, so greedy search and simple Bayesian priors both do well.

3. `cevolve_synergy`
   Uses the first-party `cevolve` adapter against a real `jnormore/cevolve` checkout and deterministic interaction-heavy target, preferring the repo benchmark subprocess and only falling back to the thinner private-session shim when needed.
   This highlights where evolutionary combination search is stronger than one-change-at-a-time reasoning.

4. `bayes_complex`
   Uses the native `autoclanker` fixture registry plus richer priors, feasibility beliefs, and graph directives.
   This is the best place to exercise era-local decay, feasibility tradeoffs, and structured human guidance.

Important distinction:

- `bayes_quickstart` is the simplest single-app onboarding path.
- `autoresearch_simple` and `cevolve_synergy` are contrast exercises against different upstream-shaped workloads.
- if you want side-by-side intuition rather than runnable product demos, use the secondary toy examples under `docs/toy_examples/`.

See [docs/LIVE_EXERCISES.md](../../docs/LIVE_EXERCISES.md) for end-to-end commands and expected outcomes.
For the quickest idea-first reruns, use `scripts/live/replay_ideas_demo.py`.
For code-level before/after toy codebases that mirror these stories, see
[docs/TOY_EXAMPLES.md](../../docs/TOY_EXAMPLES.md) and `docs/toy_examples/`.
