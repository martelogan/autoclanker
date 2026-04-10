# Bayes Quickstart

This is the beginner-friendly Bayes demo.

If you want the smallest starter input instead of the full belief file first, begin with:

- a plain inline idea list:
  - `autoclanker beliefs canonicalize-ideas --era-id era_log_parser_v1 --ideas-json '["Compiled regex matching probably helps this parser on repeated log formats."]'`
- a real model-backed inline idea list:
  - `autoclanker beliefs canonicalize-ideas --era-id era_log_parser_v1 --canonicalization-model anthropic --ideas-json '["When the same incident family keeps appearing, bias the search toward reusing the parse path and stitching each incident to the clue line beside it."]'`
- or the reusable minimal file:
  - `examples/idea_inputs/minimal.json`
- or the richer JSON beginner file:
  - `examples/idea_inputs/bayes_quickstart.json`

Use `bayes_quickstart.yaml` when you want the same richer starter example with inline
comments, not just the runnable JSON input.

Use this when you want to see the intended "most users, most of the time" path:

- a real single-file app you can read quickly,
- a short belief file with rough optimization ideas,
- no `expert_prior` or graph-directive authoring,
- and a cold-start ranking change that is visibly caused by those beliefs.

Read it in this order:

1. `app.py`
2. `beliefs.yaml`
3. `candidates.json`

What the app is doing:

- `app.py` is a tiny single-file log parser
- it parses a few realistic app log lines and groups warnings/errors into incidents
- the main question is which parser config to try next under a short budget

What the belief file is demonstrating:

- `idea`: "this single change probably helps"
- `relation`: "these two changes likely work well together"
- `proposal`: "here is a rough patch idea, but I have not canonicalized it yet"
- and the newer beginner shorthand still works:
  - a plain string idea defaults to confidence `2`
  - an object idea may omit `confidence` and still default to `2`
  - `autoclanker beliefs canonicalize-ideas` tries to bind those rough ideas to registry options before you refine them further

Minimum required files:

- `examples/idea_inputs/minimal.json` or `examples/idea_inputs/bayes_quickstart.json`: recommended front-door inputs
- `beliefs.yaml`: richer typed beginner file used by the deeper manual workflow
- `candidates.json`: required only if you want the exact documented ranking outcome
- `app.py`: explanatory only
- `expected_outcome.json`: test assertion data only

Lowest-cruft replay:

```bash
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart
```

Allowed idea inputs for this exercise:

| `gene_id` | Allowed `state_id` values | Plain-language meaning |
| --- | --- | --- |
| `parser.matcher` | `matcher_basic`, `matcher_compiled`, `matcher_jit` | Simple token splitting, compiled regex matching, or an aggressive regex plan |
| `parser.plan` | `plan_default`, `plan_context_pair`, `plan_full_scan` | Parse each error alone, pair it with nearby context, or scan a wider context neighborhood |
| `capture.window` | `window_default`, `window_wide` | Keep a small or large surrounding log window in memory |
| `io.chunk` | `chunk_default`, `chunk_large` | Parse a small or large chunk of lines at a time |
| `emit.summary` | `summary_default`, `summary_streaming` | Emit the normal summary or stream summary lines as incidents are found |

What the starter ideas are saying in plain English:

- `qs1`: compiled regex matching probably helps
- `qs2`: compiled matching and the context-pair plan work best together
- `qs3`: a wide capture window is risky and likely to hit OOM
- `qs4`: a free-form patch suggestion is still allowed, but it stays `metadata_only` until canonicalized

Which belief fields are free-form vs bounded:

- free-form: `id`, `author`, `rationale`, `proposal_text`
- bounded enums / ranges: `kind`, `confidence_level`, `evidence_sources`, `effect_strength`, `relation`, `suggested_scope`, `risk_hints.*`
- strict registry identifiers: `gene_id`, `state_id`

For the exact bounds, see [`docs/BELIEF_INPUT_REFERENCE.md`](../../../docs/BELIEF_INPUT_REFERENCE.md).

Shortest replay:

```bash
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart
```

Manual walkthrough:

```bash
./bin/dev exec -- python examples/live_exercises/bayes_quickstart/app.py
./bin/dev exec -- autoclanker beliefs canonicalize-ideas \
  --input examples/idea_inputs/bayes_quickstart.json \
  --era-id era_log_parser_v1
./bin/dev exec -- autoclanker beliefs preview --input examples/live_exercises/bayes_quickstart/beliefs.yaml
./bin/dev exec -- autoclanker session init \
  --beliefs-input examples/live_exercises/bayes_quickstart/beliefs.yaml \
  --session-root .autoclanker-exercises
```

Then apply the preview digest from `session init` and suggest over the candidate set:

```bash
./bin/dev exec -- autoclanker session apply-beliefs \
  --session-id quickstart_log_parser \
  --preview-digest <preview_digest_from_init> \
  --session-root .autoclanker-exercises

./bin/dev exec -- autoclanker session suggest \
  --session-id quickstart_log_parser \
  --candidates-input examples/live_exercises/bayes_quickstart/candidates.json \
  --session-root .autoclanker-exercises
```

What you should see:

- `cand_c_compiled_context_pair` rises to the top
- the control session stays flat and falls back to candidate ordering
- the wide-window candidates start out disfavored
- the free-form proposal stays `metadata_only` in preview until it is turned into a concrete gene or patch
