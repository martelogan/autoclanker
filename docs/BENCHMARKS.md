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
