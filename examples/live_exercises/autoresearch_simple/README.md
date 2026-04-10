# Autoresearch Simple

This exercise uses the real `karpathy/autoresearch` repository surface, specifically its current `train.py` knobs:

If you want the smallest starter input first, begin with:

- `examples/idea_inputs/autoresearch_simple.json`
- `./bin/dev exec -- autoclanker beliefs preview --input examples/idea_inputs/autoresearch_simple.json --adapter-config examples/live_exercises/autoresearch_simple/adapter.local.yaml`

Use `examples/idea_inputs/autoresearch_simple.yaml` if you want the commented teaching version.

- `DEPTH`
- `WINDOW_PATTERN`
- `TOTAL_BATCH_SIZE`
- `MATRIX_LR`
- `WARMUP_RATIO`

The objective is a lightweight `val_bpb` landscape anchored to those real constants. It is intentionally mostly additive:

- deeper models help,
- smaller total batches help on the toy hardware profile,
- slightly lower `MATRIX_LR` helps,
- a small warmup helps,
- `WINDOW_PATTERN="L"` hurts versus the repo default.

This makes it a good exercise for:

- baseline autoresearch-style hill-climbing,
- simple Bayesian main-effect priors,
- sanity-checking that real upstream integration works without requiring an H100.

Minimum required files:

- `examples/idea_inputs/autoresearch_simple.json`: recommended front-door idea file
- `beliefs.yaml`: richer typed belief file used by the deeper manual workflow
- `adapter.local.yaml`: checkout-backed exercise config for the real upstream smoke lane
- provisioned upstream repo from `./bin/dev test-upstream-live`

Optional files:

- `expected_outcome.json`: replay assertion data only
- `docs/toy_examples/autoresearch_command_autocomplete/`: human-readable mirror only

Lowest-cruft replay:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple
```

For normal non-exercise usage, the first-party adapter does not require a checkout
path. You can use `mode: auto` with `python_module` or `command` instead.

Allowed idea inputs for this exercise:

| `gene_id` | Allowed `state_id` values | Plain-language meaning |
| --- | --- | --- |
| `train.depth` | `depth_6`, `depth_8`, `depth_10` | Shallower, baseline, or deeper model |
| `train.window_pattern` | `window_L`, `window_SSSL` | Alternate or default attention-window preset |
| `batch.total` | `batch_2_18`, `batch_2_19`, `batch_2_20` | Smaller, baseline, or larger total batch |
| `optim.matrix_lr` | `lr_0_03`, `lr_0_04`, `lr_0_05` | Lower, baseline, or higher matrix learning rate |
| `schedule.warmup_ratio` | `warmup_0_0`, `warmup_0_1` | No warmup or 10% warmup |

What the starter ideas are saying in plain English:

- `ar1`: use the deeper model
- `ar2`: lower the matrix learning rate slightly
- `ar3`: depth and lower LR work especially well together
- `ar4`: avoid the largest batch because it tends to create memory pressure

Which belief fields are free-form vs bounded:

- free-form: `id`, `author`, `rationale`
- bounded enums / ranges: `kind`, `confidence_level`, `evidence_sources`, `effect_strength`, `relation`, `constraint_type`, `severity`
- strict registry identifiers: `gene_id`, `state_id`

For the exact bounds, see [`docs/BELIEF_INPUT_REFERENCE.md`](../../../docs/BELIEF_INPUT_REFERENCE.md).

How to read the lowest-cruft replay:

- it previews the ideas against the live autoresearch registry,
- creates a session,
- applies the preview digest automatically,
- and runs `session suggest` so you can see which genotype rises to the top.

Longer manual path:

```bash
./bin/dev test-upstream-live
./bin/dev exec -- autoclanker beliefs preview \
  --input examples/live_exercises/autoresearch_simple/beliefs.yaml \
  --adapter-config examples/live_exercises/autoresearch_simple/adapter.local.yaml
./bin/dev exec -- autoclanker session init \
  --beliefs-input examples/live_exercises/autoresearch_simple/beliefs.yaml \
  --adapter-config examples/live_exercises/autoresearch_simple/adapter.local.yaml \
  --session-root .autoclanker-exercises
```

Then apply the preview digest and run `session suggest`. If you do not want to wire the digest manually, use `replay_ideas_demo.py`.

Expected outcome:

- the improved candidate in `expected_outcome.json` beats baseline on `val_bpb`,
- the failure candidate hits a simulated OOM regime.
