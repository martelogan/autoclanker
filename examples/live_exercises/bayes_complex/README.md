# Bayes Complex

This exercise stays inside the native `autoclanker` stack and is meant to show the value of:

- era-local priors,
- decayed human beliefs,
- feasibility priors,
- explicit graph directives,
- session preview/apply gating.

It uses the fixture registry, because the point here is not upstream orchestration. The point is the Bayesian layer itself:

- `parser.matcher=matcher_compiled` and `parser.plan=plan_context_pair` should be favored together,
- `capture.window=window_wide` raises OOM risk,
- `io.chunk=chunk_large` is acceptable alone but dangerous with a wide capture window.

Files:

- `beliefs.yaml`
- `candidates.json`
- `expected_outcome.json`

Minimum required files:

- `beliefs.yaml`
- `candidates.json`

Allowed idea inputs for this exercise:

| `gene_id` | Allowed `state_id` values | Plain-language meaning |
| --- | --- | --- |
| `parser.matcher` | `matcher_basic`, `matcher_compiled`, `matcher_jit` | Simple token splitting, compiled regex matching, or an aggressive regex plan |
| `parser.plan` | `plan_default`, `plan_context_pair`, `plan_full_scan` | Parse each error alone, pair it with nearby context, or scan a wider context neighborhood |
| `capture.window` | `window_default`, `window_wide` | Keep a small or large surrounding log window in memory |
| `io.chunk` | `chunk_default`, `chunk_large` | Parse a small or large chunk of lines at a time |
| `emit.summary` | `summary_default`, `summary_streaming` | Emit the normal summary or stream summary lines as incidents are found |

Which belief fields are free-form vs bounded:

- free-form: `id`, `author`, `rationale`, free-text proposal content
- bounded enums / ranges: `kind`, `confidence_level`, `evidence_sources`, `effect_strength`, `relation`, `constraint_type`, `directive`
- strict registry identifiers: `gene_id`, `state_id`

For the exact bounds, see [`docs/BELIEF_INPUT_REFERENCE.md`](../../../docs/BELIEF_INPUT_REFERENCE.md).

Fast replay:

```bash
./bin/dev exec -- python scripts/showcase/replay_backing_exercise.py --showcase bayes_pair_feature_trainer
```

Raw `autoclanker` CLI replay:

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

Then generate evals, ingest them, fit, and compare `session recommend-commit` against a no-beliefs control session.

Expected outcome:

- after `apply-beliefs`, the cold-start suggestion should rank `cand_c_compiled_context_pair` first,
- the risky OOM candidate should stay at the bottom with a visibly worse feasibility score,
- after ingesting the three named observations from `expected_outcome.json`, the Bayes-guided session should give `cand_c_compiled_context_pair` a material uplift versus control,
- the Bayes-guided session should stay materially stronger than control, but the
  more exact posterior may still keep both sessions below the final commit
  threshold until more evidence is gathered.

Use this when you want the cleanest end-to-end autoclanker exercise without live upstream dependencies.
