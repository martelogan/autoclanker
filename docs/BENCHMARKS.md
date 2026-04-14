# Benchmarks

`autoclanker` keeps the required value proof deterministic and self-contained.
The main comparison script is:

```bash
./bin/dev exec -- python scripts/benchmarks/compare_optimizer_lanes.py
```

It emits machine-readable JSON covering two shipped parser targets:

## 1. Zero-eval cold start

Lanes:

- plain outer-loop control
- proposal-only rough text
- deterministic canonical Bayes
- hybrid stub-backed canonical Bayes

This target proves the minimum claim: unresolved free text stays flat, while
typed canonical beliefs shift the ranking toward the better parser pathway.

## 2. Frontier-heavy family summaries

Lanes:

- control over the same frontier document
- deterministic canonical Bayes over the same frontier
- hybrid stub-backed canonical Bayes over the same frontier

The frontier input is [`examples/frontiers/parser_frontier.json`](../examples/frontiers/parser_frontier.json).
This target proves that `autoclanker` preserves explicit pathway families,
lineage metadata, heuristic merge suggestions, and normalized budget
allocations instead of treating every candidate as a flat list entry.

The required test coverage for these reports lives in:

- [`tests/test_optimizer_benchmarks.py`](../tests/test_optimizer_benchmarks.py)
- [`tests/test_eval_contract_frontier.py`](../tests/test_eval_contract_frontier.py)

The hardened execution path also records whether the measured phase used a
contract-scoped lease, whether soft stabilization ran, and whether the local
system looked noisy during the measurement window. That metadata is part of the
trust story for local performance work, even when the required benchmark proof
stays deterministic.

Optional billed-provider and real-upstream demos remain separate live lanes. The
required benchmark proof stays deterministic on purpose.

## 3. Backend comparison metadata

The deterministic report now also includes an additive backend-comparison
section. It compares:

- heuristic objective + optimistic acquisition
- exact joint linear objective + optimistic acquisition
- exact joint linear objective + constrained Thompson acquisition

For each lane, the report records:

- top-ranked candidate
- objective backend used
- acquisition backend used
- whether the objective posterior was sampleable
- fit runtime and suggest runtime overhead
- condition number when the exact backend is active
- fallback reason when the exact or sampled path was abandoned

That section is the honest proof for this phase. It shows whether the new math
was actually used, how much overhead it added, and whether the engine had to
fall back to the older heuristic or optimistic paths.

## 4. What the deterministic benchmark proves

The required deterministic lane proves:

- typed canonical beliefs shift ranking where the shipped parser targets were
  designed to make that useful;
- explicit frontier family state survives round-trips through `suggest` and
  `frontier-status`;
- the exact objective posterior and sampled acquisition paths can be observed in
  outputs when the problem is numerically safe;
- the fallback path is machine-readable when the exact or sampled path is not
  safe, including a forced exact-objective / optimistic-acquisition comparison
  lane that makes sampled-acquisition fallback explicit.

It does **not** prove:

- that every upstream binding is running a fully native upstream search loop;
- that local performance numbers are universally scheduler-clean under arbitrary
  machine load;
- that billed model-provider behavior is deterministic.

Those broader claims remain in the optional live lanes.

## 5. Optional live lanes

When the environment supports them:

- `./bin/dev test-upstream-live` exercises the real-upstream adapter bindings
  under the same hardened eval-contract and isolated-execution model
- `./bin/dev test-live` exercises the billed provider-backed canonicalization
  path

Those lanes are intentionally separate from the required deterministic proof so
the core gate remains self-contained and non-billed.
