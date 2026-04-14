# Live Exercises

These exercises are the practical demos for the adapter and Bayesian layers.
If you only run one thing first, run `bayes_quickstart`.

## 0. Fastest path and field reference

If you want the shortest usable command path instead of the full manual workflow:

| Exercise | Minimum files | Lowest-cruft command |
| --- | --- | --- |
| `bayes_quickstart` | `examples/idea_inputs/minimal.json` or `examples/idea_inputs/bayes_quickstart.json` | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart` |
| `autoresearch_simple` | `examples/idea_inputs/autoresearch_simple.json`, `adapter.local.yaml`, plus `./bin/dev test-upstream-live` once | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple` |
| `cevolve_synergy` | `examples/idea_inputs/cevolve_synergy.json`, `adapter.local.yaml`, plus `./bin/dev test-upstream-live` once | `./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy` |
| `bayes_complex` | `beliefs.yaml`, `candidates.json` | use the manual workflow in section 3.4 |

If you want an explicit field-by-field reference, use
[`BELIEF_INPUT_REFERENCE.md`](./BELIEF_INPUT_REFERENCE.md).

## 1. Upstream-live setup

Fetch or refresh the real upstreams:

```bash
./bin/dev test-upstream-live
```

The live lane provisions:

- `karpathy/autoresearch`
- `jnormore/cevolve`

into `.local/real-upstreams/`, then targets them through the built-in
`autoclanker.bayes_layer.live_upstreams` contract module.

This lane is a real-upstream contract smoke test. It proves that the first-party
adapters bind to real upstream checkouts without fixture fallback and surface the
execution backend and metric source they actually used, while keeping the exercise
scoring harness repo-native and deterministic enough for repeatable coverage.

Outside this smoke lane, first-party adapters do not require a checkout path. Normal
usage can use `mode: auto` plus `python_module` or `command` when the upstream
integration is already installed and runnable on the system.

## 2. Billed LLM live setup

If you want actual model-provider canonicalization instead of the deterministic-only
or stub-backed lanes, use:

```bash
AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live
```

That lane keeps the same CLI flow but swaps the canonicalization provider to a real
LLM. The bundled provider alias is `anthropic`.

Smallest direct command:

```bash
autoclanker beliefs canonicalize-ideas \
  --era-id era_log_parser_v1 \
  --canonicalization-model anthropic \
  --ideas-json '["When the same incident family keeps appearing, bias the search toward reusing the parse path and stitching each incident to the clue line beside it."]'
```

## 3. Exercise menu

### 3.1 Bayes Quickstart

Files:

- `examples/idea_inputs/minimal.json`
- `examples/idea_inputs/bayes_quickstart.json`
- `examples/live_exercises/bayes_quickstart/app.py`
- `examples/live_exercises/bayes_quickstart/beliefs.yaml`
- `examples/live_exercises/bayes_quickstart/candidates.json`
- `examples/live_exercises/bayes_quickstart/expected_outcome.json`
- `scripts/live/replay_bayes_quickstart.py`

Minimum required files:

- `examples/idea_inputs/minimal.json` or `examples/idea_inputs/bayes_quickstart.json`
- `candidates.json` only if you want the exact documented ranking
- `app.py` is explanatory only

What it shows:

- a single-file log parser you can understand quickly,
- rough ideas that already look like ordinary optimization guidance,
- a cold-start ranking change before any evals exist,
- the difference between canonicalized ideas and `metadata_only` proposals.

Lowest-cruft replay:

```bash
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart
```

Fastest manual path:

```bash
./bin/dev exec -- autoclanker beliefs preview \
  --input examples/idea_inputs/minimal.json \
  --era-id era_log_parser_v1

./bin/dev exec -- autoclanker beliefs canonicalize-ideas \
  --input examples/idea_inputs/bayes_quickstart.json
```

Allowed idea inputs:

| `gene_id` | Allowed `state_id` values |
| --- | --- |
| `parser.matcher` | `matcher_basic`, `matcher_compiled`, `matcher_jit` |
| `parser.plan` | `plan_default`, `plan_context_pair`, `plan_full_scan` |
| `capture.window` | `window_default`, `window_wide` |
| `io.chunk` | `chunk_default`, `chunk_large` |
| `emit.summary` | `summary_default`, `summary_streaming` |

Why this is the recommended first demo:

- it is parser-based rather than ML-config-shaped;
- the JSON input looks close to what a new user would actually author;
- it shows Bayes value before you need advanced priors or graph directives.

Expected outcome:

- with beliefs, `cand_c_compiled_context_pair` rises to the top;
- without beliefs, the control session stays flat and falls back to candidate ordering;
- wide-window candidates start out disfavored because the beginner ideas already contain an explicit risk.

### 3.2 Autoresearch Simple

Files:

- `examples/idea_inputs/autoresearch_simple.json`
- `examples/live_exercises/autoresearch_simple/adapter.local.yaml`
- `examples/live_exercises/autoresearch_simple/beliefs.yaml`
- `examples/live_exercises/autoresearch_simple/expected_outcome.json`

Minimum required files:

- `examples/idea_inputs/autoresearch_simple.json`
- `adapter.local.yaml`

What it shows:

- a mostly additive landscape over real `autoresearch/train.py` knobs,
- simple priors help,
- hill-climbing is adequate,
- extreme batch/depth settings can trip feasibility.

Expected live outcome:

- the `improved` candidate from `expected_outcome.json` beats `baseline`,
- the `failure_candidate` returns `oom`.

Lowest-cruft replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple
```

Allowed idea inputs:

| `gene_id` | Allowed `state_id` values |
| --- | --- |
| `train.depth` | `depth_6`, `depth_8`, `depth_10` |
| `train.window_pattern` | `window_L`, `window_SSSL` |
| `batch.total` | `batch_2_18`, `batch_2_19`, `batch_2_20` |
| `optim.matrix_lr` | `lr_0_03`, `lr_0_04`, `lr_0_05` |
| `schedule.warmup_ratio` | `warmup_0_0`, `warmup_0_1` |

Manual replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/autoresearch_simple/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase autoresearch_command_autocomplete
```

Interpretation:

- this is an upstream-backed contract-smoke adapter demo, not a native `autoclanker session` walkthrough;
- the replay script evaluates the named baseline, improved, and failure candidates against the real `autoresearch` checkout;
- the scoring function is still the repo-native exercise harness rather than the full upstream training semantics;
- the companion toy code lives in `docs/toy_examples/autoresearch_command_autocomplete/`.

### 3.3 cEvolve Synergy

Files:

- `examples/idea_inputs/cevolve_synergy.json`
- `examples/live_exercises/cevolve_synergy/adapter.local.yaml`
- `examples/live_exercises/cevolve_synergy/train.py`
- `examples/live_exercises/cevolve_synergy/beliefs.yaml`
- `examples/live_exercises/cevolve_synergy/expected_outcome.json`

Minimum required files:

- `examples/idea_inputs/cevolve_synergy.json`
- `adapter.local.yaml`

What it shows:

- a deterministic interaction-heavy landscape,
- single moves help a little,
- the best result comes from combining them,
- this is the cleanest case for the first-party `cevolve` adapter over a real checkout-backed benchmark target.

Expected live outcome:

- the `improved` candidate beats `baseline`,
- it also beats both `single_changes` by at least the margin in `expected_outcome.json`.

Lowest-cruft replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy
```

Allowed idea inputs:

| `gene_id` | Allowed `state_id` values |
| --- | --- |
| `sort.threshold` | `threshold_16`, `threshold_32`, `threshold_64` |
| `sort.partition` | `partition_lomuto`, `partition_hoare` |
| `sort.pivot` | `pivot_median_of_three`, `pivot_middle`, `pivot_random` |
| `sort.iterative` | `iterative_off`, `iterative_on` |

Manual replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- autoclanker adapter probe --input examples/live_exercises/cevolve_synergy/adapter.local.yaml
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase cevolve_sort_partition
```

Interpretation:

- this is also an upstream-backed contract-smoke adapter demo rather than a native `autoclanker session` walkthrough;
- the replay script evaluates baseline, single-change, and combined candidates against the real `cevolve` checkout;
- the preferred successful path is the repo benchmark subprocess against the checked-out `cevolve` repo and exercise target, with a thinner private-session fallback only when that subprocess path is unavailable;
- the companion toy code lives in `docs/toy_examples/cevolve_sort_partition/`.

### 3.4 Bayes Complex

Files:

- `examples/live_exercises/bayes_complex/beliefs.yaml`
- `examples/live_exercises/bayes_complex/candidates.json`
- `examples/live_exercises/bayes_complex/expected_outcome.json`
- `scripts/live/generate_bayes_complex_evals.py`

What it shows:

- preview/apply session gating,
- cold-start priors that move suggestion order before any evals exist,
- decayed expert priors that still influence unseen pair structure after observations,
- feasibility priors and graph directives on risky branches,
- a commit recommendation that flips to `true` only in the Bayes-guided session.

Suggested run:

```bash
./bin/dev exec -- autoclanker session init \
  --beliefs-input examples/live_exercises/bayes_complex/beliefs.yaml \
  --session-root .autoclanker-exercises
```

Then:

1. apply the preview digest for that session;
2. create a control session with the same era but no beliefs;
3. compare cold-start `session suggest` output against `examples/live_exercises/bayes_complex/candidates.json`;
4. generate the deterministic observation files:

```bash
./bin/dev exec -- python scripts/live/generate_bayes_complex_evals.py \
  --output-dir .autoclanker-exercises/evals
```

5. ingest those evals into both sessions, run `session fit`, and compare `session suggest` plus `session recommend-commit`.

Fast replay:

```bash
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```

Expected outcome:

- cold start: the Bayes-guided session ranks `cand_c_compiled_context_pair` first, while the control session stays flat and falls back to candidate ordering;
- the risky `cand_d_wide_window_large_chunk` branch stays at the bottom with a lower feasibility score than the default candidate;
- after ingesting the three deterministic evals, the Bayes-guided session gives `cand_c_compiled_context_pair` a material uplift versus control;
- the more exact posterior keeps the Bayes-guided session ahead of control, but
  may still leave both sessions below the final commit threshold until more
  evidence is ingested.

## 4. Why the exercises are split this way

- `autoresearch` is a real upstream repo but not a composable optimization library, so the upstream-live exercise is upstream-anchored and lightweight, with explicit repo-subprocess-vs-heuristic metric-source labeling.
- `cevolve` is a composable real upstream, so the upstream-live exercise uses a repo benchmark subprocess when available and only falls back to the thinner private-session shim when needed.
- `bayes_quickstart` is the recommended entry point for ordinary human-authored Bayes beliefs.
- the complex Bayes exercise focuses on `autoclanker` itself, where the Bayesian layer is the primary subject.

## 5. Companion Toy Examples

If you want the same stories in human-readable toy codebases instead of candidate payloads, use:

- `docs/toy_examples/autoresearch_command_autocomplete/`
- `docs/toy_examples/cevolve_sort_partition/`
- `docs/toy_examples/bayes_pair_feature_trainer/`

If you want the simplest Bayes-first live demo instead, start with:

- `examples/live_exercises/bayes_quickstart/`

These are intentionally secondary documentation aids, not the primary product demos.

Those examples are documented in [`TOY_EXAMPLES.md`](./TOY_EXAMPLES.md) and summarized by:

```bash
./bin/dev exec -- python scripts/showcase/run_toy_examples.py
```

For each toy example, the intended reading order is:

1. `app.py --variant baseline` to understand the tiny app itself,
2. `benchmark.py --variant ...` to see the metric change,
3. `scripts/showcase/replay_backing_exercise.py --showcase ...` to replay the real `autoclanker`-backed demo.
