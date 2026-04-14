# DESIGN.md

## 1. System boundary

The Bayesian layer sits between an outer search loop and an expensive evaluation harness.

### Inputs
- candidate genotypes from the outer loop,
- realized eval results,
- structured human / LLM beliefs,
- rough high-level ideas that must be canonicalized into typed beliefs or remain metadata-only,
- current-era baseline and metadata,
- adapter metadata from the active integration target,
- optional benchmark history from previous eras.

### Outputs
- ranked candidate scores,
- posterior summaries,
- suggested human queries,
- commit / no-commit recommendations,
- rethink summaries,
- previewable compiled priors,
- stable session artifacts for CLI / wrapper tooling.

## 2. High-level architecture

```text
Human / LLM input (JSON or YAML)
    ↓
schema validation
    ↓
deterministic canonicalization
    ↓
optional provider-agnostic model canonicalization
    ↓
typed beliefs + optional session-local surface overlay
    ↓
preview / apply boundary + locked eval contract + influence summary
    ↓
preview compiler  ───────────────┐
    ↓                           │
belief compiler                  │
    ↓                           │
prior store ────────────────────┤
                                ↓
frontier-aware outer-loop adapter → feature encoder → exact joint linear objective posterior
                             ↓                ↓
                             └────────────→ feasibility surrogate
                                              ↓
                                 posterior interaction graph
                                              ↓
              sampled-or-optimistic acquisition + family-aware query policy
                                              ↓
                 session store + CLI JSON outputs + eval run records
```

The key layering is:
- **library core** for all real logic,
- **thin CLI** for humans and scripts,
- **thin adapters** for autoresearch / cevolve / future wrappers.

The exact objective path is intentionally narrow:
- explicit main effects,
- screened pair effects,
- compact metadata features already supported by the feature encoder.

There are no embedding features, latent prompt-state features, or attempts to
model provider hidden state. If the exact solve or posterior sampling path
becomes ill-conditioned, the engine falls back to the heuristic objective path
and records why in machine-readable artifacts.

Important constraint:

- free text never directly updates the posterior;
- it must become an inspectable typed belief or a metadata-only proposal first.

## 3. Repository layout

Canonical runtime layout:

```text
autoclanker/
  bayes_layer/
```

The CLI should be rooted at `autoclanker`, not at a separate helper binary.

## 4. Core module layout

Suggested module tree:

```text
autoclanker/
  __init__.py
  cli.py
  py.typed
  bayes_layer/
    __init__.py
    config.py
    types.py
    registry.py
    belief_io.py
    belief_compiler.py
    feature_encoder.py
    surrogate_objective.py
    surrogate_feasibility.py
    posterior_graph.py
    acquisition.py
    query_policy.py
    commit_policy.py
    session_store.py
    adapters/
      __init__.py
      protocols.py
      fixture.py
      autoresearch.py
      cevolve.py
```

## 5. Adapter architecture

### 5.1 Why adapters exist

`autoclanker` should be a reusable Bayesian layer, not a hard-coded rewrite of one upstream loop.

The adapter seam is what makes all of these possible at once:
- first-party support for `autoresearch`,
- first-party support for `cevolve`,
- future third-party loops,
- fixture-backed self-contained tests,
- future wrappers or companion tools.

### 5.2 Adapter modes

Support at least these modes:

- `fixture`
- `auto` for first-party adapters
- `local_repo_path`
- `installed_module`
- `subprocess_cli` (optional but useful)

Not every adapter has to support every mode, but the config surface should allow them.
For `autoresearch` and `cevolve`, `auto` is the recommended front door because it
can use a checkout path, importable adapter module, or runnable adapter command
without forcing users into one integration shape.

### 5.3 Adapter protocol responsibilities

The generic adapter should be able to:
- identify itself and report status,
- discover or build a gene / idea registry,
- expose a richer optimization surface with semantic metadata, not just low-level knobs,
- capture a reproducible eval contract for the active benchmark and environment,
- materialize a candidate genotype into an eval-ready form,
- run an evaluation under an explicit execution context and emit a schema-valid result,
- optionally recommend or apply a commit step,
- expose enough metadata for rethink summaries.

### 5.4 Fixture adapter

The fixture adapter is required so the repo can be implemented and tested with no external dependencies beyond the repo itself.

It should simulate:
- a registry,
- candidate evaluation,
- intended vs realized genotype,
- patch-hash aggregation,
- feasibility failure modes.

### 5.5 First-party upstream adapters

`autoresearch` and `cevolve` adapters should be part of the repo from the outset.

They should:
- load config from a validated adapter config file,
- probe for local paths or importability,
- fail clearly when a requested path is missing,
- allow optional integration tests that skip when upstream references are unavailable.

Current implementation note:

- the fixture adapter is the fully self-contained execution target;
- first-party upstream adapters probe and report local path / module / subprocess availability;
- when `allow_missing` is enabled and the upstream target is absent, those adapters may fall back to the fixture-backed contract shim so the repo stays runnable and testable without external checkouts.

## 6. Session-store design

Default session layout:

```text
.autoclanker/<session_id>/
  session_manifest.yaml
  eval_contract.json
  beliefs.yaml
  compiled_preview.json
  compiled_priors.json
  observations.jsonl
  frontier_status.json
  posterior_summary.json
  query.json
  commit_decision.json
  influence_summary.json
  eval_runs/
  RESULTS.md
  convergence.png
  candidate_rankings.png
  belief_graph_prior.png
  belief_graph_posterior.png
```

The same store must be relocatable under another parent path, e.g.:

```text
.cevolve/<session_id>/autoclanker/
```

## 7. CLI design

The CLI is the primary user surface in v1.

Recommended command mapping:

```text
beliefs canonicalize-ideas
beliefs expand-ideas
beliefs validate
beliefs preview
beliefs compile
eval validate
adapter registry
adapter surface
adapter validate-config
adapter list
adapter probe
session init
session apply-beliefs
session ingest-eval
session run-eval
session run-frontier
session fit
session suggest
session frontier-status
session recommend-commit
session status
```

The CLI should be:
- non-interactive,
- JSON-by-default on stdout for machine-readable commands,
- file-friendly,
- script-friendly,
- easy for future wrappers to call.

`run-eval` and `run-frontier` are the hardened execution path when the engine
itself is responsible for running the benchmark. They must default to isolated
temp git worktrees when a repo snapshot is available, and only fall back to
copy-mode or fixture mode when the adapter cannot support git-backed isolation.
When the active eval policy is measurement-sensitive, the measured phase should
also take a contract-scoped advisory lease and record soft-stabilization
metadata so local performance runs do not silently overlap.

`suggest` remains finite-pool only. When the objective posterior is sampleable,
the configured constrained Thompson path samples explicit utility over the
current candidate pool and combines that with the feasibility model. When
sampling is unavailable or unsafe, the CLI records that it used the optimistic
fallback path instead.

## 8. User-lane design

### 8.1 Simple lane

A normal user should be able to operate through inline rough ideas or small JSON / YAML files with bounded fields.

### 8.2 Expert lane

A power user should be able to supply direct priors and graph directives, but those must remain optional.

### 8.3 Round-trip preview

Every belief batch should be previewable in optimizer-relevant units before it is actually applied.

### 8.4 Hybrid canonicalization lane

When a canonicalization model is configured, the beginner path becomes:

```text
rough ideas
→ deterministic canonicalization
→ optional model-assisted typed suggestions / overlays
→ preview
→ apply
→ run-eval or ingest
→ fit / suggest / frontier-status / recommend
```

This lane must stay inspectable and reproducible through session artifacts rather than hidden model state.

### 8.6 Frontier-aware exploration

Candidate pools now support a persisted frontier document rather than only a
flat list input. The current frontier state preserves:

- family representatives and normalized per-family budget allocations based on
  the supplied candidate weights,
- parent candidate and parent belief lineage metadata,
- heuristic pending merge suggestions when two frontier families remain strong,
- unresolved cross-family queries.

When uncertainty is localized enough to ask a clean question, the follow-up
query path should prefer concrete candidate or family comparisons instead of
abstract coefficient-sign prompts.

### 8.5 Common assistant-authoring workflow

The common advanced workflow should be:

```text
rough ideas
→ adapter registry / adapter surface
→ beliefs canonicalize-ideas
→ beliefs expand-ideas (optional alias)
→ preview
→ assistant-guided escalation for unresolved or high-impact beliefs
→ expert_prior / graph_directive in JSON by default
→ apply
```

This is the right division of labor:
- deterministic canonicalization handles easy cases without a model;
- provider-backed canonicalization resolves rough strategy text into typed candidate beliefs;
- `expand-ideas` is currently an alias when the caller wants the normalized typed payload under that name;
- the assistant skill only escalates the cases that truly need direct prior or graph control.

### 8.6 Validation lanes

The implementation maintains three validation classes:

- `check`: required, self-contained, non-billed;
- `test-upstream-live`: optional real-upstream, non-billed;
- `test-live`: optional billed model-provider validation.

This split is deliberate. Real-model canonicalization is part of the product, but it should not make the core required gate nondeterministic or billing-dependent.

## 9. Wrapper mapping

A future wrapper or companion tool should be able to wrap:

```text
create session      → session init
preview beliefs     → beliefs preview
append eval         → session ingest-eval
refresh posterior   → session fit
get next step       → session suggest
read status         → session status
commit gate         → session recommend-commit
adapter probe       → adapter probe
```

This is why the library / CLI / file contracts matter more than a rich interactive UI in v1.
