# Contributing to autoclanker

Thanks for your interest in improving `autoclanker`.

## Ways to contribute

- report bugs or confusing behavior
- improve examples and documentation
- add tests for new behavior
- improve adapter integrations
- refine the Bayesian or canonicalization layers

## Development setup

Supported Python: 3.11+.

From a checkout:

```bash
./bin/dev setup
```

If you prefer a manual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Validation

Use `./bin/dev` as the canonical command surface.

Typical local loop:

```bash
./bin/dev format
./bin/dev lint
./bin/dev typecheck
./bin/dev pylint
./bin/dev test
```

Before opening a substantial change, run:

```bash
./bin/dev test-full
./bin/dev check
```

Optional live lanes:

```bash
./bin/dev test-upstream-live
AUTOCLANKER_ENABLE_LLM_LIVE=1 ./bin/dev test-live
```

`test-live` requires a real model-provider key. `test-upstream-live` does not.

## Pull requests

Prefer small, focused changes.

Before submitting:

- add or update tests when behavior changes
- update docs and examples when user-facing behavior changes
- avoid unrelated cleanup in the same change
- keep public APIs typed

## Project conventions

- `./bin/dev` is the source of truth; `make` is only a compatibility layer
- runtime code lives under `autoclanker/`
- tests live in `tests/`
- docs live in `docs/`
- schemas live in `schemas/`
- examples live in `examples/`

See [`STYLE.md`](STYLE.md) for day-to-day coding conventions.
