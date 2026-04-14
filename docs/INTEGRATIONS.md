# Integrations

## 1. Integration philosophy

`autoclanker` is a Bayesian layer over an existing optimization loop. The core repo
owns belief ingestion, prior compilation, posterior summaries, and session
artifacts. Upstream engines own the search space, candidate materialization, eval
execution, and optional commit behavior.

That boundary is expressed through adapters.

For hardened sessions, the adapter boundary also owns two trust-critical jobs:

- capture the eval contract that defines the locked benchmark surface
- execute candidates under an explicit isolated execution context when
  `autoclanker` itself is running the eval

## 2. Built-in adapter kinds

`autoclanker` ships these adapter kinds:

- `fixture`
- `autoresearch`
- `cevolve`
- `python_module`
- `subprocess`

`fixture` keeps the repo self-contained.
`autoresearch` and `cevolve` are first-party integrations.
`python_module` and `subprocess` are the generic escape hatches for external loops.

## 3. First-party adapter modes

The first-party `autoresearch` and `cevolve` adapters support four modes:

- `auto`
- `local_repo_path`
- `installed_module`
- `subprocess_cli`

`auto` is the recommended default. It tries whichever integration hints you provide,
in this order:

1. `repo_path`
2. `python_module`
3. `command`

That means you do not need a local checkout just to use a first-party adapter. If
the upstream integration is already available as an importable adapter module or a
runnable adapter command, omit `repo_path`.

## 4. Recommended configs

If you want the shipped upstream smoke exercises without choosing paths manually, use:

```bash
./bin/dev deps upstreams
```

or provision one upstream at a time:

```bash
./bin/dev deps autoresearch
./bin/dev deps cevolve
```

### 4.1 Installed module or runnable command

Use this shape when the upstream is already installed on your machine and you do not
want to point `autoclanker` at a checkout path:

```yaml
adapter:
  kind: autoresearch
  mode: auto
  python_module: my_project.autoclanker_autoresearch_adapter
  command:
    - my-autoresearch-autoclanker
  session_root: .autoclanker
  allow_missing: true
```

Only one usable hint has to resolve. `auto` will pick the first available one.

### 4.2 Local checkout

Use this shape when you want to bind directly to a local repo checkout:

```yaml
adapter:
  kind: cevolve
  mode: auto
  repo_path: ../../.local/real-upstreams/cevolve
  session_root: .autoclanker
  allow_missing: true
```

`repo_path` is resolved relative to the adapter config file, not the current shell
working directory.

The shipped checkout-backed examples are:

- `examples/adapters/autoresearch.local.yaml`
- `examples/adapters/cevolve.local.yaml`

They resolve to:

- `repo_path: ../../.local/real-upstreams/autoresearch`
- `repo_path: ../../.local/real-upstreams/cevolve`

Both point at `.local/real-upstreams/` rather than a repo-specific vendor directory.

If you keep manual checkout copies elsewhere, the path-resolution helpers also
honor `references/autoresearch` and `references/cevolve` as optional local
fallbacks. Those paths are intentionally not tracked by the repo.

## 5. CLI expectations

The adapter CLI should support at least:

```bash
autoclanker adapter list
autoclanker adapter validate-config --input examples/adapters/autoresearch.local.yaml
autoclanker adapter probe --input examples/adapters/autoresearch.local.yaml
autoclanker adapter registry --input examples/adapters/autoresearch.local.yaml
autoclanker adapter surface --input examples/adapters/autoresearch.local.yaml
```

## 6. Eval-contract metadata

Adapter metadata may include:

- `benchmark_root`
- `eval_harness_path`
- `environment_paths`
- `workspace_root`
- `workspace_snapshot_mode`
- `eval_policy.mode`
- `eval_policy.stabilization`
- `eval_policy.performance_sensitive`
- `eval_policy.lease_scope`

Those fields let `autoclanker` capture digests for the benchmark tree, eval
harness, adapter config, and environment inputs. They are generic on purpose:
the same contract model works for fixture-backed tests, checkout-backed
upstreams, importable module adapters, and future orchestration layers.

When the adapter can bind to a real repo snapshot, hardened execution defaults to
temp git worktrees. Non-git or fixture adapters may fall back to copy-mode or
fixture-mode isolation. Measurement-sensitive runs may also take a
contract-scoped advisory lease and record soft-stabilization metadata so local
performance probes do not silently overlap.

## 7. Real-upstream smoke lane

`./bin/dev test-upstream-live` provisions the public upstream repos into
`.local/real-upstreams/` by default and then runs checkout-backed contract-smoke
tests against them.

That lane exists to prove that the first-party adapters can bind to real upstream
source trees without fixture fallback and surface the execution backend and
metric source they actually used. It is intentionally separate from the more
general installed-module and runnable-command integrations described above.

It is a real-upstream contract smoke test, not a claim that every scoring path
is fully upstream-native end to end.

## 8. Stronger native-ish first-party paths

The first-party adapters still stay thin, but the preferred live path is now
more legitimate than a pure fixture shim:

- `autoresearch`: when a real binding is available, the adapter prefers a repo
  subprocess under the locked eval contract and isolated workspace. Metrics come
  from actual subprocess output when available and are otherwise marked as a
  repo-subprocess heuristic fallback.
- `cevolve`: the shipped smoke lane prefers a repo benchmark subprocess against
  the checked-out target and only falls back to the thinner private-session shim
  when that subprocess path is unavailable. A configured public command may also
  be used through adapter metadata, but the smoke lane does not overclaim that.

Both integrations reuse the shared trust boundary:
- isolated workspace execution,
- contract digests,
- measurement policy,
- lease and stabilization metadata,
- non-destructive default behavior.

## 9. Environment passthrough

When an upstream adapter resolves to an importable module or subprocess command,
`autoclanker` passes through the normal process environment. That means upstream
provider auth and model selection keep working the way the underlying engine expects,
instead of being remapped by `autoclanker`.

## 10. Future-proofing

This boundary is meant to stay extension-friendly:

- the Bayes core remains stable and typed;
- first-party adapters stay thin integration layers;
- future users can plug in their own loop via `python_module` or `subprocess`;
- future wrappers like `pi-autoclanker` can orchestrate the same machine-readable
  adapter, eval-contract, and frontier contracts.
