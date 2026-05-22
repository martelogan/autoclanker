---
name: bigbets-portfolio-curator
description: Use when ingesting normalized idea-family issues, grouping lanes into big bets, ranking a bigbets registry, maintaining unlock states, or regenerating portfolio artifacts.
---

# Bigbets Portfolio Curator

Use this skill when issue-family lanes need to roll up into a ranked big-bet
portfolio.

## Workflow

1. Validate the current registry.

```bash
bigbets validate --input <bigbets.yaml>
```

2. Import or merge normalized issue exports.

```bash
bigbets issues import --input <issues.json>
bigbets issues merge --registry <bigbets.yaml> --input <issues.json> --output <next.json>
```

3. Curate big bets.

- Every idea family maps to exactly one big bet.
- Multiple idea families may map to the same big bet.
- Use `P0`, `P1`, `P2`, ... as the visible priority layers.
- Keep `unlock_state` honest: `locked`, `emerging`, or `unlocked`.
- Mark a bet `unlocked` only with short durable `unlock_evidence`.
- Prefer fewer, clearer big bets over one bet per issue.
- Before reranking, check whether issue bodies have current pathway-status
  tables and whether stale run comments should be folded back into the issue
  contract.

4. Regenerate artifacts.

```bash
bigbets render --input <bigbets.yaml> --output-dir <out>
bigbets site scaffold --input <bigbets.yaml> --output-dir <site>
```

## References

- For ranking and dependency guidance, read
  [`references/portfolio-mapping.md`](references/portfolio-mapping.md).
