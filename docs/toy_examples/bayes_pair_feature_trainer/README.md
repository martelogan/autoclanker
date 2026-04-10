# Bayes Guided Trainer

This toy codebase mirrors the core Bayes showcase, but in direct code form instead of candidate JSON.

Read it in this order:

1. `app.py`
2. `variants/`
3. `benchmark.py`

What the app is doing:

- `app.py` is a tiny numeric trainer over a few labeled samples
- it can optionally include an interaction feature, which is the key hidden win
- higher `utility` is better
- `oom` is the failure mode to avoid
- the key story is that one unseen pair is better than the best already-observed single move

Why this example fits the Bayesian layer:

- the best local move is not the best overall move
- a structured prior can justify trying a promising unseen pair
- feasibility beliefs also matter because one branch is obviously risky

What each variant means:

- `variants/baseline.py`: the default trainer config
- `variants/local_observed_best.py`: the best single move already seen
- `variants/belief_guided.py`: the unseen pair-feature change the Bayes demo is meant to promote
- `variants/risky_oom.py`: the risky branch the feasibility model should penalize

Manual toy-app run:

```bash
./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/app.py --variant baseline
```

Manual benchmark run:

```bash
./bin/dev exec -- python docs/toy_examples/bayes_pair_feature_trainer/benchmark.py --variant belief_guided
```

How this maps to `autoclanker`:

- the toy app is only the human-readable mirror
- the real backing demo lives in `examples/live_exercises/bayes_complex/`
- the toy app itself does not call an LLM; the real `autoclanker` session flow is where human or LLM-authored beliefs enter

Replay the real method-backed demo:

```bash
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```
