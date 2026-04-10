# Developer Environment

This project keeps a simple default workflow and two optional strict lanes.

## Goals

1. Keep onboarding simple through `bin/dev`.
2. Keep tool/runtime installs project-local by default.
3. Avoid forcing a global `mise` install.
4. Preserve one coherent baseline across local, Nix, and devcontainer lanes.

## Canonical Entry Point

Use `bin/dev` for day-to-day work.

```bash
bin/dev setup
bin/dev format
bin/dev lint
bin/dev typecheck
bin/dev pylint
bin/dev test
bin/dev test-full
bin/dev build
bin/dev check
bin/dev doctor
bin/dev strict-env status
bin/dev strict-env validate
```

`make` targets are compatibility aliases and route through `bin/dev`.

## Workflow Modes

| Mode | Best for | Requires | Activation owner |
| --- | --- | --- | --- |
| Default (`bin/dev`) | Most contributors | Python + uv | none |
| Default + `mise activate` | Auto-activation on repo entry | `mise` | `mise` |
| Strict `devenv + direnv` | Nix-first reproducibility | `devenv` + `direnv` | `direnv` |
| `.devcontainer` | Containerized editor workflows | container runtime + devcontainer support | container runtime |

Single-activation-owner rule: do not let both `mise` and `direnv` mutate the
same repository environment at the same time.

## Behavior Model

`bin/dev` resolves execution in this order:

1. `AUTOCLANKER_DEV_MISE_BIN`
2. Project-local mise at `.local/dev/mise/bin/mise`
3. System `mise` on `PATH`
4. Best-effort bootstrap via `scripts/dev/bootstrap-mise.sh`
5. Direct fallback commands (`uv` + repo scripts)

This keeps core workflows usable even when `mise` is unavailable.

## Core Checks

The project exposes four first-class code quality lanes:

1. `bin/dev format` for automatic Ruff fixes and formatting.
2. `bin/dev lint` for Ruff validation.
3. `bin/dev typecheck` for Pyright.
4. `bin/dev pylint` for lightweight error-only static checks.

## Local Install Root

- Install root: `.local/dev`
- Local bin directory: `.local/dev/bin`
- UV cache: `.local/dev/uv-cache`

`bin/dev exec -- <command...>` prepends `.local/dev/bin` and `.venv/bin` to
`PATH`.

## Local Environment Variables

Use `.env.local` for machine-local secrets or overrides.

- starter file: `.env.example`
- gitignored files: `.env`, `.env.local`, `.env.*.local`, `.envrc`
- load behavior:
  - `bin/dev` helper scripts load `.env` and then `.env.local`
  - strict `devenv + direnv` mode also loads `.env` and `.env.local`
  - `scripts/test-live.sh` optionally loads `AUTOCLANKER_LLM_ENV_FILE` after the repo dotenv files

Common values:

- `AUTOCLANKER_ANTHROPIC_API_KEY` or `ANTHROPIC_API_KEY` for billed live canonicalization
- `AUTOCLANKER_ENABLE_LLM_LIVE=1` to opt into billed live tests
- `AUTOCLANKER_LIVE_AUTORESEARCH_PATH` and `AUTOCLANKER_LIVE_CEVOLVE_PATH` for the checkout-backed upstream smoke lane only

Normal first-party adapter usage does not require those path variables. If an
upstream adapter module or runnable adapter command is already installed, use
`mode: auto` plus `python_module` or `command` in the adapter config instead.

If you want the repo to provision the optional checkout-backed upstream smoke
dependencies for you, run:

```bash
bin/dev deps upstreams
```

## Strict Environment Lanes

### `devenv + direnv`

Quickstart:

```bash
bin/dev strict-env devenv
direnv allow
bin/dev doctor
bin/dev test
```

Repo assets:

1. `devenv.nix`
2. `dev/env/envrc.devenv.example`

### `.devcontainer`

Quickstart:

1. Open the repo in a devcontainer-capable editor/runtime.
2. Reopen in the container.
3. Run `bin/dev setup` and `bin/dev test`.

Repo assets:

1. `.devcontainer/devcontainer.json`
2. `.devcontainer/README.md`

## Drift Protection

Keep toolchain versions and core environment keys coherent by running:

```bash
bin/dev strict-env validate
```

## Git Hooks

This project does not install Git hooks automatically.
