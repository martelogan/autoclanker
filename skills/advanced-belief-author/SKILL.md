---
name: advanced-belief-author
description: Use when turning rough optimization ideas into advanced autoclanker expert_prior or graph_directive belief files, especially when the user already has a registry or a beginner ideas file and wants a precise Bayesian declaration.
---

# Advanced Belief Author

Use this skill when the user wants a stronger Bayesian config than the beginner
`examples/idea_inputs/*.yaml` files.

## Workflow

1. Inspect the allowed registry surface.

- Fixture/default registry:
  - `autoclanker adapter registry`
- Adapter-backed registry:
  - `autoclanker adapter registry --input <adapter_config.yaml>`

2. If the user already has a simple ideas file, expand it first.

- `autoclanker beliefs canonicalize-ideas --input <ideas.yaml> --era-id <era_id>`
- `autoclanker beliefs expand-ideas --input <ideas.yaml> --era-id <era_id>`
- model-backed pass:
  - `AUTOCLANKER_CANONICALIZATION_MODEL=anthropic autoclanker beliefs canonicalize-ideas --input <ideas.yaml> --era-id <era_id>`

Use `canonicalize-ideas` first when the user begins with rough high-level ideas and
needs help binding them to concrete registry options. `expand-ideas` is currently an
alias that emits the same normalized typed payload, so use it only when that name reads
better for the handoff you are preparing.

When a real model provider is available, prefer the provider-backed pass first and
then inspect the emitted `canonicalization_summary` and any `surface_overlay`
artifacts before escalating to advanced beliefs.

3. Upgrade only the beliefs that truly need more precision.

- Use `expert_prior` when the user wants an explicit prior family, mean, scale, or decay.
- Use `graph_directive` when the user wants explicit pair inclusion/exclusion or linkage control.
- Leave rough human-language suggestions as `proposal` unless the user has enough information to bind them to concrete genes or pairs.

4. Prefer this escalation ladder.

- `proposal`
- `idea` / `relation`
- `expert_prior`
- `graph_directive`

Do not jump to `expert_prior` or `graph_directive` unless the user actually needs that precision.

5. For complex targets, use the provider-assisted escalation loop.

- inspect `autoclanker adapter surface` to see both concrete knobs and higher-level strategy/risk families
- run `beliefs canonicalize-ideas` with `AUTOCLANKER_CANONICALIZATION_MODEL=anthropic`
- accept typed `idea` or `relation` beliefs when they already capture the intent
- promote only the remaining unresolved or underspecified cases into `expert_prior` or `graph_directive`
- keep the model output inspectable: free text never bypasses preview/apply

## References

- Beginner field/reference surface:
  - [`docs/BELIEF_INPUT_REFERENCE.md`](../../docs/BELIEF_INPUT_REFERENCE.md)
- Beginner examples:
  - [`examples/idea_inputs/README.md`](../../examples/idea_inputs/README.md)
- Full typed examples:
  - [`examples/human_beliefs/basic_session.yaml`](../../examples/human_beliefs/basic_session.yaml)
  - [`examples/human_beliefs/expert_session.yaml`](../../examples/human_beliefs/expert_session.yaml)

## Output Preference

When authoring advanced beliefs:

- prefer a full JSON belief batch for machine-authored outputs,
- use YAML only when the user is likely to hand-edit the file next,
- keep `session_context` intact,
- add brief rationales,
- use only registry-valid `gene_id` / `state_id` values,
- and favor the smallest advanced file that captures the user’s real intent.
