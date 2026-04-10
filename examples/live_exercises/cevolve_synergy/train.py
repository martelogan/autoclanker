"""
Deterministic interaction-heavy target for the live cEvolve adapter exercise.

The metric is synthetic but intentionally structured:

- single changes provide modest improvements,
- the best result comes from combining threshold, partition, and iterative choices,
- a few conflicts make naive greedy search easier to trap.
"""

INSERTION_THRESHOLD = 16
PARTITION_SCHEME = "lomuto"
PIVOT_STRATEGY = "median_of_three"
USE_ITERATIVE = False


def score_configuration() -> tuple[float, float, float]:
    time_ms = 120.0
    synergy_bonus = 0.0
    robustness_score = 0.94

    if INSERTION_THRESHOLD == 32:
        time_ms -= 9.0
    elif INSERTION_THRESHOLD == 64:
        time_ms += 6.0

    if PARTITION_SCHEME == "hoare":
        time_ms -= 5.0
        robustness_score += 0.01

    if PIVOT_STRATEGY == "middle":
        time_ms += 3.0
        robustness_score -= 0.01
    elif PIVOT_STRATEGY == "random":
        time_ms += 8.0
        robustness_score -= 0.03

    if USE_ITERATIVE:
        time_ms -= 1.0

    if INSERTION_THRESHOLD == 32 and PARTITION_SCHEME == "hoare":
        time_ms -= 12.0
        synergy_bonus += 12.0
        robustness_score += 0.02

    if INSERTION_THRESHOLD == 32 and PARTITION_SCHEME == "hoare" and USE_ITERATIVE:
        time_ms -= 6.0
        synergy_bonus += 6.0
        robustness_score += 0.01

    if PIVOT_STRATEGY == "middle" and PARTITION_SCHEME == "hoare":
        time_ms += 4.0
    if PIVOT_STRATEGY == "random" and USE_ITERATIVE:
        time_ms += 5.0
    if INSERTION_THRESHOLD == 64 and USE_ITERATIVE:
        time_ms += 7.0

    return time_ms, synergy_bonus, robustness_score


def main() -> int:
    time_ms, synergy_bonus, robustness_score = score_configuration()
    print("Scenario: cevolve_synergy")
    print(f"INSERTION_THRESHOLD: {INSERTION_THRESHOLD}")
    print(f"PARTITION_SCHEME: {PARTITION_SCHEME}")
    print(f"PIVOT_STRATEGY: {PIVOT_STRATEGY}")
    print(f"USE_ITERATIVE: {USE_ITERATIVE}")
    print("---")
    print(f"time_ms: {time_ms:.3f}")
    print(f"synergy_bonus: {synergy_bonus:.3f}")
    print(f"robustness_score: {robustness_score:.3f}")
    print("errors: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
