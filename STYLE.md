# autoclanker Style Guide

Keep the codebase small, typed, and predictable.

## Formatting and linting

- use `./bin/dev format` before committing
- use `./bin/dev lint` for validation
- line length is 88 characters
- let Ruff handle import ordering

## Types

- fully annotate public functions and methods
- keep Pyright strict-clean
- run `./bin/dev typecheck` so `pyright` stays clean
- prefer small typed dataclasses and protocols over loose dictionaries
- preserve `py.typed` support

## Testing

- `./bin/dev test` is the default non-live lane
- `./bin/dev test-full` is the broader local sweep
- `./bin/dev check` is the required self-contained gate
- use `upstream_live` and `live` markers only for optional lanes

## Design habits

- keep free text out of the posterior unless it has been canonicalized into typed beliefs
- prefer explicit session artifacts over hidden state
- keep adapters thin and treat upstream loops as integrations, not vendored code
- preserve the preview-then-apply boundary for beliefs

## Practical workflow

```bash
./bin/dev format
./bin/dev lint
./bin/dev typecheck
./bin/dev pylint
./bin/dev test
```
