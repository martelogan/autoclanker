# Audit Output

Return these artifacts when auditing work into bigbets.

1. Candidate lanes table.

Columns: `slug`, `title`, `source`, `baseline`, `evidence`,
`candidate_big_bet`, `priority`, `role`, `next_action`, `status`.

2. Big-bet draft.

For each big bet include `id`, `title`, `near_term_win`, `long_term_unlock`,
`unlock_state`, dependencies, and edge labels when useful.

3. Issue bodies.

Create one normalized issue body per accepted lane. Include an embedded
`bigbets:idea-family` metadata block, evidence summary, related-work section,
evaluation contract, kickoff prompt/command, pathway-status table, and
maintenance contract.

4. Rejected or parked material.

List stale, duplicate, or low-evidence sources with a one-line reason.

5. Open questions.

List only the questions that materially block issue publication, such as a
missing baseline artifact, unclear harness, or conflicting ownership.
