---
name: clankerprof-operator
description: Use when analyzing pprof CPU profiles with clankerprof, authoring runtime rule packs or slice configs, checking sample-facts parity, or preparing a cross-language port of the clankerprof sample-facts engine.
---

# Clankerprof Operator

Use this skill when a task involves `clankerprof` profiles, target attribution,
slice attribution, runtime rule packs, sample-facts artifacts, or parity checks.

## Workflow

1. Read the contract before changing behavior.

- Treat [`docs/CLANKERPROF_SPEC.md`](../../docs/CLANKERPROF_SPEC.md) as the
  normative compatibility target.
- Treat [`docs/CLANKERPROF_PARITY.md`](../../docs/CLANKERPROF_PARITY.md) as the
  current tested capability matrix.
- Use [`clankerprof/README.md`](../../clankerprof/README.md) for compact user
  examples and [`docs/CLANKERPROF.md`](../../docs/CLANKERPROF.md) for the full
  guide.

2. Start with the smallest useful projection.

```bash
clankerprof targets --profile profile.pb.gz --target Boundary#call
clankerprof slices --profile profile.pb.gz
clankerprof facts --profile profile.pb.gz --output profile-facts.json
```

Use `--facts profile-facts.json` when replaying multiple projections from one
decoded profile. This is also the preferred seam for golden tests and
cross-language ports.

3. Keep behavior declarative.

- Put language/runtime details in `--runtime-rules runtime-rules.yml`.
- Put core/native class lists in `--core-classes core_classes.csv`.
- Put ownership, contacts, docs, or other domain labels in slice metadata.
- Prefer `library:`, `dependency:`, `package:`, `vendor:`, or selectors declared
  by `library_selector_path_patterns`.
- Keep compatibility aliases such as `gem:` and older flag names only for
  existing configs or parity checks.

4. Make folding explicit and reviewable.

- `--fold-runtime-internals` folds configured runtime-internal categories into
  the first meaningful caller.
- `native_name_category_rules` classify folded native callers whose pseudo path
  cannot carry the semantic category.
- `caller_fallback_name_prefixes` controls native/delegated caller fallback when
  semantic runtime categorization is disabled.
- Rule order matters. Put specific rules before broad fallbacks.

5. Verify both profile and fact paths.

```bash
clankerprof facts --profile profile.pb.gz --output profile-facts.json
clankerprof targets --profile profile.pb.gz --config target_config.json --output targets-from-profile.json
clankerprof targets --facts profile-facts.json --config target_config.json --output targets-from-facts.json
clankerprof slices --profile profile.pb.gz --config clankerprof-slices.yml --output slices-from-profile.json
clankerprof slices --facts profile-facts.json --config clankerprof-slices.yml --output slices-from-facts.json
```

When real-profile reference artifacts are available locally, use the parity
helper. Do not commit profiles or private reference outputs.

```bash
CLANKERPROF_REAL_PROFILE_PARITY=1 \
python scripts/clankerprof/check_real_profile_parity.py \
  --profile profile.pb.gz \
  --target-config target_config.json \
  --expected-target-json reference-targets.json
```

6. Validate changes.

```bash
./bin/dev format
./bin/dev exec -- pytest tests/test_clankerprof.py tests/test_package.py tests/test_cli_commands.py -q
./bin/dev check
```

## Cross-Language Port Checklist

- Preserve `clankerprof.sample_facts.v1` byte-level semantics, not Python object
  internals.
- Preserve sparse pprof IDs, inline frames, folded-location markers, sample
  order, all sample values, and leaf-to-root stack order.
- Implement target and slice projections over sample facts, not over a separate
  decoder-specific call graph.
- Match JSON/CSV outputs through golden tests before optimizing internals.
- Keep runtime rules and slice metadata data-driven; do not bake one language,
  framework, owner model, or dependency layout into the core.
