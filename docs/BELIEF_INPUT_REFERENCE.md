# Belief Input Reference

Start here if you want the smallest runnable `autoclanker` belief file and a clear
answer to three questions:

- which files are actually required,
- which fields are free-form human language,
- which fields have strict allowed values.

The short answer for new users:

- no, you do not need a YAML file everywhere you go
- a tiny JSON file is now the easiest reusable starting point
- the smallest beginner shape is just `ideas:` plus plain-language idea strings
- `era_id` is still required somewhere, either in the file or on the CLI
- `option` means an app option identifier from `autoclanker adapter registry`, not an `autoclanker` setting
- when `option` is omitted, `autoclanker` tries to auto-canonicalize the idea against registry descriptions and aliases
- `beliefs canonicalize-ideas` and `beliefs expand-ideas` currently produce the same normalized typed belief payload; `canonicalize-ideas` is the more intention-revealing name when you care about provenance metadata

## Plain-language mental model

The same vocabulary should stay legible whether you start in `autoclanker`
directly or through a thin wrapper:

- `optimization lever (gene)`: one explicit upstream knob
- `setting (state)`: one concrete value of that knob
- `candidate lane` or `pathway`: one concrete combination being compared
- `frontier`: the explicit set of lanes under comparison
- `belief`: a typed claim about one lever, relation, risk, or preference
- `comparison query`: the next concrete lane-vs-lane or family-vs-family
  question worth asking

The engine learns over explicit candidate features and typed relations, not
hidden prompt state.

## No-file path

All beginner belief commands accept stdin with `--input -`, and `session init`
accepts `--beliefs-input -`. The shortest installed-user path is now `--ideas-json`.

Smallest real example:

```yaml
ideas:
  - Compiled regex matching probably helps this parser on repeated log formats.
```

Preview it without creating a file:

```bash
autoclanker beliefs preview \
  --era-id era_my_app_v1 \
  --ideas-json '["Compiled regex matching probably helps this parser on repeated log formats."]'
```

Or pipe YAML on stdin:

```bash
cat <<'YAML' | autoclanker beliefs preview --input - --era-id era_my_app_v1
ideas:
  - Compiled regex matching probably helps this parser on repeated log formats.
YAML
```

If you want to write the JSON payload to disk, `--output` works either before the
command family or after the leaf subcommand:

```bash
autoclanker beliefs preview --input ideas.yaml --output preview.json
autoclanker --output preview.json beliefs preview --input ideas.yaml
```

## Smallest runnable files

| Goal | Required files | Optional but helpful |
| --- | --- | --- |
| Preview beginner Bayes ideas against the fixture registry | no file at all if you use `--ideas-json`, otherwise `examples/idea_inputs/minimal.json` | `autoclanker adapter registry` |
| Reproduce the exact Bayes quickstart ranking | `examples/idea_inputs/bayes_quickstart.json`, `examples/live_exercises/bayes_quickstart/candidates.json` | `examples/live_exercises/bayes_quickstart/app.py` |
| Reproduce the frontier-aware parser comparison | `examples/frontiers/parser_frontier.json` plus a live session | `autoclanker session run-frontier` |
| Preview live `autoresearch` ideas | `examples/idea_inputs/autoresearch_simple.json`, `examples/live_exercises/autoresearch_simple/adapter.local.yaml` | `examples/live_exercises/autoresearch_simple/README.md` |
| Preview live `cevolve` ideas | `examples/idea_inputs/cevolve_synergy.json`, `examples/live_exercises/cevolve_synergy/adapter.local.yaml` | `examples/live_exercises/cevolve_synergy/README.md` |

Notes:

- `app.py` and `train.py` are explanatory. They are not required by the CLI.
- `expected_outcome.json` files are for tests and replay assertions, not for normal usage.
- `examples/frontiers/parser_frontier.json` is the smallest shipped frontier
  example showing lineage, families, and merge-ready candidates.
- The upstream-backed exercise configs in `examples/live_exercises/*/adapter.local.yaml`
  are checkout-backed examples. For a normal installed integration, you can instead
  use `mode: auto` with `python_module` or `command` and omit `repo_path`.
- The shortest checkout-backed setup is:

```bash
./bin/dev test-upstream-live
```

If you want the lowest-cruft demo commands, use:

```bash
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise bayes_quickstart
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise autoresearch_simple
./bin/dev exec -- python scripts/live/replay_ideas_demo.py --exercise cevolve_synergy
```

The non-fixture exercises need `./bin/dev test-upstream-live` first.

If you want actual billed model-backed canonicalization instead of only the
deterministic path, the shortest opt-in setup is:

```bash
AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live
```

The direct CLI provider aliases are `anthropic` and `openai-compatible`. The
OpenAI-compatible provider reads `OPENAI_API_KEY` or
`AUTOCLANKER_OPENAI_API_KEY`; use `AUTOCLANKER_OPENAI_API_URL` for proxy
endpoints and `AUTOCLANKER_OPENAI_MODEL` for provider-specific model handles.

## Smallest valid beginner belief batch

This is the smallest useful shape for most users:

```yaml
ideas:
  - Compiled regex matching usually improves parser throughput.
```

Notes:

- top-level `ideas` is required.
- each `ideas[]` item may be either a plain string or an object.
- for string items, `autoclanker` treats the string as `idea` and defaults `confidence` to `2`.
- for object items, `idea` is required and `confidence` also defaults to `2` if omitted.
- `id` is optional and autogenerated if omitted.
- `option` / `target` or `options` / `members` are optional when the registry has enough semantic metadata for auto-canonicalization.
- if auto-canonicalization is not confident enough, the idea remains a `proposal` with suggested registry options for review.
- `session_context.era_id` is still required somewhere, but for beginner idea files you
  can provide it on the CLI with `--era-id`.
- `session_context.session_id` is optional in schema, and for beginner idea files you
  can provide it on the CLI with `--session-id`.
- `option` and `options` are the beginner names for strict registry identifiers. `target`
  and `members` remain accepted aliases.

The smallest reusable file in the repo is [minimal.json](../examples/idea_inputs/minimal.json).
The richer front-door quickstart files are:

- [bayes_quickstart.json](../examples/idea_inputs/bayes_quickstart.json)
- [autoresearch_simple.json](../examples/idea_inputs/autoresearch_simple.json)
- [cevolve_synergy.json](../examples/idea_inputs/cevolve_synergy.json)

The same beginner shapes also exist as commented YAML teaching files, including
[minimal.yaml](../examples/idea_inputs/minimal.yaml).

## Free-Form vs Bounded Fields

### Shared fields

| Field | Required | Allowed values | Free-form? |
| --- | --- | --- | --- |
| `session_context.era_id` | yes | non-empty string | yes |
| `session_context.session_id` | no, but recommended for `session init` | non-empty string | yes |
| `session_context.author` | no | non-empty string | yes |
| `session_context.user_profile` | no | `basic`, `expert` | no |
| `beliefs[].id` | yes | unique non-empty string within the file | yes |
| `beliefs[].kind` | yes | `proposal`, `idea`, `relation`, `preference`, `constraint`, `expert_prior`, `graph_directive` | no |
| `beliefs[].confidence_level` | yes | `1`, `2`, `3`, `4` | no |
| `beliefs[].evidence_sources[]` | no | `intuition`, `prior_run`, `paper`, `code_inspection`, `benchmark`, `other` | no |
| `beliefs[].rationale` | no | string | yes |

### Beginner kinds

These are the recommended kinds for most users.

| Kind | Required fields | Strict values | Free-form fields |
| --- | --- | --- | --- |
| `proposal` | `proposal_text` | `suggested_scope`: `prompt`, `gene`, `patch`, `constraint`, `other` | `proposal_text`, `rationale` |
| `idea` | `gene.gene_id`, `gene.state_id`, `effect_strength` | `effect_strength`: `-3` to `3`; `risk.*`: `0` to `3` | `rationale` |
| `relation` | `members`, `relation`, `strength` | `relation`: `synergy`, `conflict`, `dependency`, `exclusion`; `strength`: `1` to `3`; `joint_effect_strength`: `-3` to `3` if present | `rationale` |

Risk fields for `proposal.risk_hints` and `idea.risk`:

- `compile_fail`
- `runtime_fail`
- `oom`
- `timeout`
- `metric_instability`

Each risk value is bounded to `0`, `1`, `2`, or `3`.

### Advanced kinds

| Kind | Required fields | Strict values |
| --- | --- | --- |
| `preference` | `left_pattern`, `right_pattern`, `preference`, `strength` | `preference`: `left`, `right`, `tie`; `strength`: `1` to `5` |
| `constraint` | `constraint_type`, `severity`, `scope` | `constraint_type`: `hard_exclude`, `soft_avoid`, `require`, `budget_cap`; `severity`: `1` to `3` |
| `expert_prior` | `target`, `prior_family`, `mean`, `scale` | `target.target_kind`: `main_effect`, `pair_effect`, `feasibility_logit`, `vram_effect`; `prior_family`: `normal`, `logit_normal`; `scale > 0`; `observation_weight > 0` if present |
| `graph_directive` | `members`, `directive`, `strength` | `directive`: `screen_include`, `screen_exclude`, `linkage_positive`, `linkage_negative`; `strength`: `1` to `3` |

Decay override bounds for `expert_prior.decay_override`:

- `per_eval_multiplier`: greater than `0`, at most `1`
- `cross_era_transfer`: between `0` and `1`

## What Is Still Free-Form

These are the fields where ordinary human language is expected:

- `session_context.author`
- `beliefs[].id`
- `beliefs[].rationale`
- `proposal.proposal_text`
- `context.tags[]`

These are not free-form and must match the schema or the exercise registry:

- `kind`
- `confidence_level`
- `evidence_sources`
- `suggested_scope`
- `relation`
- `constraint_type`
- `preference`
- `directive`
- `gene.gene_id`
- `gene.state_id`
- `members[].gene_id`
- `members[].state_id`

## Where `gene_id` And `state_id` Come From

- In the fixture Bayes demos, they come from the built-in registry in
  [`autoclanker/bayes_layer/registry.py`](../autoclanker/bayes_layer/registry.py).
- In the live adapter demos, they come from the exercise registries in
  [`autoclanker/bayes_layer/live_upstreams.py`](../autoclanker/bayes_layer/live_upstreams.py).
- Each live exercise README lists the allowed `gene_id` and `state_id` values for that demo in plain language.
- The auto-canonicalizer matches high-level idea text against registry descriptions,
  aliases, and state descriptions. High-quality registry metadata makes the
  “ideas first” workflow much more useful.

Concrete examples from the shipped demos:

- Bayes quickstart: `parser.matcher=matcher_compiled`, `parser.plan=plan_context_pair`
- Live autoresearch: `train.depth=depth_10`, `optim.matrix_lr=lr_0_03`
- Live cevolve: `sort.threshold=threshold_32`, `sort.partition=partition_hoare`

For the model-backed path, you can skip files entirely:

```bash
autoclanker beliefs canonicalize-ideas \
  --era-id era_log_parser_v1 \
  --canonicalization-model anthropic \
  --ideas-json '["When the same incident family keeps appearing, bias the search toward reusing the parse path and stitching each incident to the clue line beside it."]'
```

## Recommended Starting Point

If you want the easiest Bayes path, start with:

- [`examples/idea_inputs/minimal.json`](../examples/idea_inputs/minimal.json)
- `autoclanker beliefs canonicalize-ideas --input examples/idea_inputs/minimal.json --era-id era_my_app_v1`
- [`examples/live_exercises/bayes_quickstart/README.md`](../examples/live_exercises/bayes_quickstart/README.md)
- [`scripts/live/replay_ideas_demo.py`](../scripts/live/replay_ideas_demo.py)

If you want the common advanced workflow after that, use the assistant skill in
[`skills/advanced-belief-author/SKILL.md`](../skills/advanced-belief-author/SKILL.md).
The intended path is:

```text
rough ideas
→ canonicalize-ideas
→ preview
→ assistant-guided refinement for only the unresolved or high-value cases
→ machine-authored JSON belief batch by default
→ apply
```

YAML still makes sense when you expect to hand-edit the advanced file next.

## Frontier inputs

`session suggest` accepts either the older flat candidate-pool shape:

```json
{"candidates": [{"candidate_id": "cand_a", "genotype": [...]}]}
```

or the richer frontier shape:

```json
{
  "frontier_id": "parser_frontier_demo",
  "default_family_id": "family_default",
  "candidates": [
    {
      "candidate_id": "cand_a_default",
      "family_id": "baseline",
      "origin_kind": "seed",
      "genotype": [...]
    }
  ]
}
```

The richer frontier document is what enables persisted lineage metadata,
normalized family budget allocations, heuristic merge suggestions, and
`session frontier-status`.

Useful commands:

```bash
autoclanker session suggest \
  --session-id parser-demo \
  --candidates-input examples/frontiers/parser_frontier.json

autoclanker session run-frontier \
  --session-id parser-demo \
  --frontier-input examples/frontiers/parser_frontier.json

autoclanker session frontier-status \
  --session-id parser-demo
```
