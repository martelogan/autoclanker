from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import math

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from autoclanker.bayes_layer.review_bundle import build_review_bundle
from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.types import (
    BeliefDeltaSummary,
    CommitDecision,
    CompiledPriorBundle,
    JsonValue,
    PosteriorGraph,
    PosteriorSummary,
    ProposalLedger,
    QuerySuggestion,
    RankedCandidate,
    SessionManifest,
    ValidAdapterConfig,
    ValidEvalResult,
)


@dataclass(frozen=True, slots=True)
class ReportGraphEdge:
    source: str
    target: str
    weight: float
    relation: str


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _graph_note(message: str, *, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_convergence_plot(
    path: Path,
    *,
    observations: Sequence[ValidEvalResult],
    baseline_utility: float,
) -> None:
    if not observations:
        _graph_note(
            "No eval observations yet. Ingest eval JSON to populate the convergence view.",
            title="Session Convergence",
            path=path,
        )
        return

    utilities = [item.utility for item in observations]
    eval_indices = list(range(1, len(observations) + 1))
    cumulative_best: list[float] = []
    best_so_far = utilities[0]
    for utility in utilities:
        best_so_far = max(best_so_far, utility)
        cumulative_best.append(best_so_far)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        eval_indices,
        utilities,
        color="#9A6B2E",
        linewidth=1.8,
        marker="o",
        markersize=4,
        label="observed utility",
    )
    axis.plot(
        eval_indices,
        cumulative_best,
        color="#143D59",
        linewidth=2.4,
        label="best so far",
    )
    axis.axhline(
        baseline_utility,
        color="#7A869A",
        linestyle="--",
        linewidth=1.2,
        label="posterior baseline",
    )
    axis.set_title("Session Convergence")
    axis.set_xlabel("Observation")
    axis.set_ylabel("Utility")
    axis.grid(alpha=0.25, linestyle=":")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _candidate_label(candidate_id: str) -> str:
    return candidate_id.replace("cand_", "", 1)


def _write_candidate_rankings_plot(
    path: Path,
    *,
    ranked_candidates: Sequence[RankedCandidate],
) -> None:
    if not ranked_candidates:
        _graph_note(
            "No ranked candidates are available for this session yet.",
            title="Candidate Rankings",
            path=path,
        )
        return

    top_candidates = list(ranked_candidates[:8])
    labels = [_candidate_label(item.candidate_id) for item in reversed(top_candidates)]
    scores = [item.acquisition_score for item in reversed(top_candidates)]
    figure_height = max(4.0, 0.7 * len(top_candidates) + 1.8)
    figure, axis = plt.subplots(figsize=(9, figure_height))
    bars = axis.barh(labels, scores, color="#1F6B8A")
    axis.set_title("Candidate Rankings")
    axis.set_xlabel("Acquisition score")
    axis.grid(axis="x", alpha=0.25, linestyle=":")

    for bar, candidate in zip(bars, reversed(top_candidates), strict=True):
        axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            (
                f"  utility={_format_float(candidate.predicted_utility)}"
                f"  valid={_format_float(candidate.valid_probability)}"
                f"  uncertainty={_format_float(candidate.uncertainty)}"
            ),
            va="center",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _pair_ref_to_nodes(pair_ref: str) -> tuple[str, str]:
    pair_body = pair_ref.removeprefix("pair:")
    left, right = pair_body.split("+", maxsplit=1)
    return left, right


def _main_ref_to_node(target_ref: str) -> str:
    return target_ref.removeprefix("main:")


def _build_prior_graph(
    compiled: CompiledPriorBundle,
) -> tuple[tuple[str, ...], tuple[ReportGraphEdge, ...]]:
    nodes: set[str] = set()
    edges: list[ReportGraphEdge] = []

    for spec in compiled.main_effect_priors:
        nodes.add(_main_ref_to_node(spec.item.target_ref))
    for spec in compiled.pair_priors:
        left, right = _pair_ref_to_nodes(spec.item.target_ref)
        nodes.update((left, right))
        edges.append(
            ReportGraphEdge(
                source=left,
                target=right,
                weight=spec.item.mean,
                relation="prior_pair_effect",
            )
        )
    for spec in compiled.linkage_hints:
        left, right = _pair_ref_to_nodes(spec.item.target_ref)
        nodes.update((left, right))
        edges.append(
            ReportGraphEdge(
                source=left,
                target=right,
                weight=spec.item.mean,
                relation="graph_directive",
            )
        )

    return (
        tuple(sorted(nodes)),
        tuple(
            sorted(edges, key=lambda item: (item.source, item.target, item.relation))
        ),
    )


def _edge_color(edge: ReportGraphEdge) -> str:
    if "negative" in edge.relation or edge.weight < 0:
        return "#B24020"
    if edge.relation == "graph_directive":
        return "#6B5B95"
    return "#1F6B8A"


def _compact_node_label(node_ref: str) -> str:
    if "=" not in node_ref:
        return node_ref
    gene_id, state_id = node_ref.split("=", maxsplit=1)
    return (
        f"{gene_id.rsplit('.', maxsplit=1)[-1]}\n{state_id.rsplit('.', maxsplit=1)[-1]}"
    )


def _draw_graph(
    path: Path,
    *,
    title: str,
    nodes: Sequence[str],
    edges: Sequence[ReportGraphEdge],
) -> None:
    if not nodes:
        _graph_note("No graph structure is available yet.", title=title, path=path)
        return

    ordered_nodes = list(nodes)
    count = len(ordered_nodes)
    positions: dict[str, tuple[float, float]] = {}
    radius = 0.38 if count > 1 else 0.0
    for index, node in enumerate(ordered_nodes):
        angle = (2.0 * math.pi * index) / max(count, 1)
        positions[node] = (
            0.5 + radius * math.cos(angle),
            0.5 + radius * math.sin(angle),
        )

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.set_title(title)
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    annotate_edges = len(edges) <= 6
    for edge in edges:
        source_position = positions.get(edge.source)
        target_position = positions.get(edge.target)
        if source_position is None or target_position is None:
            continue
        x_values = [source_position[0], target_position[0]]
        y_values = [source_position[1], target_position[1]]
        line_width = 1.4 + min(abs(edge.weight), 2.5) * 1.4
        axis.plot(
            x_values,
            y_values,
            color=_edge_color(edge),
            linewidth=line_width,
            alpha=0.65,
            zorder=1,
        )
        if annotate_edges:
            axis.text(
                (x_values[0] + x_values[1]) / 2,
                (y_values[0] + y_values[1]) / 2,
                _format_float(edge.weight),
                ha="center",
                va="center",
                fontsize=7,
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.7,
                },
                zorder=3,
            )

    for node, position in positions.items():
        axis.scatter(
            [position[0]],
            [position[1]],
            s=1800,
            color="#F4EAD5",
            edgecolors="#143D59",
            linewidths=1.5,
            zorder=2,
        )
        axis.text(
            position[0],
            position[1],
            _compact_node_label(node),
            ha="center",
            va="center",
            fontsize=8,
            zorder=3,
        )

    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _posterior_graph_edges(graph: PosteriorGraph) -> tuple[ReportGraphEdge, ...]:
    return tuple(
        ReportGraphEdge(
            source=edge.source,
            target=edge.target,
            weight=edge.weight,
            relation=edge.relation,
        )
        for edge in graph.edges
    )


def _bundle_mapping(value: object | None) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    return cast(Mapping[str, object], value)


def _bundle_sequence(value: object | None) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(cast(Sequence[object], value))


def _bundle_string(value: object | None) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _bundle_brief_lines(
    title: str,
    payload: Mapping[str, object] | None,
) -> list[str]:
    lines = ["", f"## {title}"]
    if payload is None:
        lines.append("- not available")
        return lines
    summary = _bundle_string(payload.get("summary"))
    if summary is not None:
        lines.append(summary)
    bullets = [
        item
        for item in _bundle_sequence(payload.get("bullets"))
        if _bundle_string(item) is not None
    ]
    if bullets:
        lines.extend(f"- {_bundle_string(item)}" for item in bullets)
    elif summary is None:
        lines.append("- not available")
    return lines


def _lane_decision_lines(review_bundle: Mapping[str, object]) -> list[str]:
    lines = ["", "## Lane Decisions"]
    lanes = [
        _bundle_mapping(item) for item in _bundle_sequence(review_bundle.get("lanes"))
    ]
    lane_rows = [item for item in lanes if item is not None]
    if not lane_rows:
        lines.append("- no explicit lane decisions are recorded yet")
        return lines
    lines.extend(
        [
            "| Lane | Family | Decision | Proposal | Next Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for lane in lane_rows[:8]:
        lines.append(
            "| "
            f"`{_bundle_string(lane.get('lane_id')) or 'unknown'}` | "
            f"`{_bundle_string(lane.get('family_id')) or 'family_default'}` | "
            f"`{_bundle_string(lane.get('decision_status')) or 'hold'}` | "
            f"`{_bundle_string(lane.get('proposal_status')) or 'not_ready'}` | "
            f"{_bundle_string(lane.get('next_step')) or 'review evidence'} |"
        )
        evidence_summary = _bundle_string(lane.get("evidence_summary"))
        if evidence_summary is not None:
            lines.append(f"  evidence: {evidence_summary}")
    return lines


def _proposal_recommendation_lines(review_bundle: Mapping[str, object]) -> list[str]:
    lines = ["", "## Proposal Recommendations"]
    proposals = [
        _bundle_mapping(item)
        for item in _bundle_sequence(review_bundle.get("proposals"))
    ]
    proposal_rows = [item for item in proposals if item is not None]
    if not proposal_rows:
        lines.append("- no durable proposal recommendations are recorded yet")
        return lines
    lines.extend(
        [
            "| Proposal | Readiness | Source Lane | Resume |",
            "| --- | --- | --- | --- |",
        ]
    )
    for proposal in proposal_rows[:6]:
        lines.append(
            "| "
            f"`{_bundle_string(proposal.get('proposal_id')) or 'proposal'}` | "
            f"`{_bundle_string(proposal.get('readiness')) or 'not_ready'}` | "
            f"`{_bundle_string(proposal.get('source_lane_id')) or 'lane'}` | "
            f"`{_bundle_string(proposal.get('resume_hint')) or 'none'}` |"
        )
        evidence_basis = _bundle_string(proposal.get("evidence_basis"))
        if evidence_basis is not None:
            lines.append(f"  evidence: {evidence_basis}")
    return lines


def _evidence_explainer_lines(review_bundle: Mapping[str, object]) -> list[str]:
    lines = ["", "## How to read the evidence views"]
    evidence = _bundle_mapping(review_bundle.get("evidence"))
    if evidence is None:
        lines.append("- no evidence views are recorded yet")
        return lines
    notes = [
        item
        for item in _bundle_sequence(evidence.get("notes"))
        if _bundle_string(item) is not None
    ]
    if notes:
        lines.extend(f"- {_bundle_string(item)}" for item in notes)
    views = [_bundle_mapping(item) for item in _bundle_sequence(evidence.get("views"))]
    present_views = [item for item in views if item is not None]
    if present_views:
        lines.append("- Key evidence views:")
        for view in present_views[:5]:
            label = _bundle_string(view.get("label")) or "view"
            description = _bundle_string(view.get("description")) or "no description"
            lines.append(f"  - `{label}`: {description}")
    return lines


def _write_results_markdown(
    path: Path,
    *,
    manifest: SessionManifest,
    summary: PosteriorSummary,
    ranked_candidates: Sequence[RankedCandidate],
    queries: Sequence[QuerySuggestion],
    decision: CommitDecision | None,
    belief_delta_summary: BeliefDeltaSummary | None,
    proposal_ledger: ProposalLedger | None,
    review_bundle: Mapping[str, JsonValue] | None,
    artifact_filenames: Sequence[str],
) -> None:
    lines = [
        "# Session Results",
        "",
        "## At a glance",
        f"- Session: `{manifest.session_id}`",
        f"- Era: `{manifest.era_id}`",
        f"- Observation count: `{summary.observation_count}`",
        f"- Aggregated patch count: `{summary.aggregate_count}`",
        f"- Objective baseline: `{_format_float(summary.objective_baseline)}`",
        f"- Valid baseline: `{_format_float(summary.valid_baseline)}`",
        "",
        "## Top candidates",
    ]

    if ranked_candidates:
        lines.extend(
            [
                "| Rank | Candidate | Acquisition | Utility | Valid |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for index, candidate in enumerate(ranked_candidates[:5], start=1):
            lines.append(
                "| "
                f"{index} | `{candidate.candidate_id}` | "
                f"{_format_float(candidate.acquisition_score)} | "
                f"{_format_float(candidate.predicted_utility)} | "
                f"{_format_float(candidate.valid_probability)} |"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Top posterior features"])
    if summary.top_features:
        lines.extend(
            [
                "| Target | Mean | Prior | Variance | Support |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for feature in summary.top_features[:6]:
            lines.append(
                "| "
                f"`{feature.feature_name}` | "
                f"{_format_float(feature.posterior_mean)} | "
                f"{_format_float(feature.prior_mean)} | "
                f"{_format_float(feature.posterior_variance)} | "
                f"{feature.support} |"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Follow-up queries"])
    if queries:
        lines.extend(f"- {query.prompt}" for query in queries[:5])
    else:
        lines.append("- none")

    lines.extend(["", "## Belief changes"])
    if belief_delta_summary is None:
        lines.append("- no belief delta summary is available yet")
    else:
        if belief_delta_summary.strengthened:
            lines.append("- Strengthened:")
            lines.extend(
                f"  - {entry.summary}"
                for entry in belief_delta_summary.strengthened[:4]
            )
        if belief_delta_summary.weakened:
            lines.append("- Weakened:")
            lines.extend(
                f"  - {entry.summary}" for entry in belief_delta_summary.weakened[:4]
            )
        if belief_delta_summary.uncertain:
            lines.append("- Remaining uncertainty:")
            lines.extend(
                f"  - {entry.summary}" for entry in belief_delta_summary.uncertain[:3]
            )
        if belief_delta_summary.promoted_candidate_ids:
            lines.append(
                "- Promoted lanes: "
                + ", ".join(
                    f"`{candidate_id}`"
                    for candidate_id in belief_delta_summary.promoted_candidate_ids
                )
            )
        if belief_delta_summary.dropped_family_ids:
            lines.append(
                "- Dropped families: "
                + ", ".join(
                    f"`{family_id}`"
                    for family_id in belief_delta_summary.dropped_family_ids
                )
            )
        if belief_delta_summary.notes:
            lines.extend(f"- {note}" for note in belief_delta_summary.notes)

    lines.extend(["", "## Commit recommendation"])
    if decision is None:
        lines.append("- not generated yet")
    else:
        lines.extend(
            [
                f"- Recommended: `{str(decision.recommended).lower()}`",
                f"- Candidate: `{decision.candidate_id or 'none'}`",
                f"- Gain probability: `{_format_float(decision.gain_probability)}`",
                f"- Valid probability: `{_format_float(decision.valid_probability)}`",
                f"- Reason: {decision.reason}",
            ]
        )

    lines.extend(["", "## Proposal summary"])
    if proposal_ledger is None or not proposal_ledger.entries:
        lines.append("- no durable proposal state is available yet")
    else:
        current_entry = (
            None
            if proposal_ledger.current_proposal_id is None
            else next(
                (
                    entry
                    for entry in proposal_ledger.entries
                    if entry.proposal_id == proposal_ledger.current_proposal_id
                ),
                None,
            )
        )
        if current_entry is not None:
            lines.extend(
                [
                    f"- Current proposal: `{current_entry.proposal_id}`",
                    f"- Readiness: `{current_entry.readiness_state}`",
                    f"- Lane: `{current_entry.candidate_id}`",
                    f"- Evidence: {current_entry.evidence_summary}",
                ]
            )
            if current_entry.unresolved_risks:
                lines.extend(
                    f"- Unresolved risk: {risk}"
                    for risk in current_entry.unresolved_risks[:3]
                )
        alternates = [
            entry
            for entry in proposal_ledger.entries
            if current_entry is None or entry.proposal_id != current_entry.proposal_id
        ]
        if alternates:
            lines.append("- Alternates worth keeping:")
            lines.extend(
                f"  - `{entry.proposal_id}` ({entry.readiness_state}): {entry.evidence_summary}"
                for entry in alternates[:3]
            )

    if review_bundle is not None:
        lines.extend(
            _bundle_brief_lines(
                "Prior Brief",
                _bundle_mapping(review_bundle.get("prior_brief")),
            )
        )
        lines.extend(
            _bundle_brief_lines(
                "Run Brief",
                _bundle_mapping(review_bundle.get("run_brief")),
            )
        )
        lines.extend(
            _bundle_brief_lines(
                "Posterior Brief",
                _bundle_mapping(review_bundle.get("posterior_brief")),
            )
        )
        lines.extend(
            _bundle_brief_lines(
                "Proposal Brief",
                _bundle_mapping(review_bundle.get("proposal_brief")),
            )
        )
        lines.extend(_lane_decision_lines(review_bundle))
        lines.extend(_proposal_recommendation_lines(review_bundle))
        lines.extend(_evidence_explainer_lines(review_bundle))

    lines.extend(["", "## Artifacts"])
    lines.extend(f"- `{filename}`" for filename in artifact_filenames)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_session_report_bundle(
    *,
    store: FilesystemSessionStore,
    session_id: str,
    manifest: SessionManifest,
    adapter_config: ValidAdapterConfig | None = None,
    compiled: CompiledPriorBundle,
    observations: Sequence[ValidEvalResult],
    summary: PosteriorSummary,
    ranked_candidates: Sequence[RankedCandidate] | None = None,
    queries: Sequence[QuerySuggestion] = (),
    decision: CommitDecision | None = None,
) -> dict[str, str]:
    artifacts = store.artifact_filenames
    ranked = tuple(ranked_candidates or summary.top_candidates)
    belief_delta_summary = store.load_belief_delta_summary(session_id)
    proposal_ledger = store.load_proposal_ledger(session_id)
    review_bundle = build_review_bundle(
        store=store,
        session_id=session_id,
        manifest=manifest,
        adapter_config=adapter_config,
    )

    results_path = store.artifact_path(session_id, artifacts.results_markdown)
    convergence_path = store.artifact_path(session_id, artifacts.convergence_plot)
    candidate_rankings_path = store.artifact_path(
        session_id,
        artifacts.candidate_rankings_plot,
    )
    prior_graph_path = store.artifact_path(session_id, artifacts.prior_graph_plot)
    posterior_graph_path = store.artifact_path(
        session_id,
        artifacts.posterior_graph_plot,
    )

    _write_results_markdown(
        results_path,
        manifest=manifest,
        summary=summary,
        ranked_candidates=ranked,
        queries=queries,
        decision=decision,
        belief_delta_summary=belief_delta_summary,
        proposal_ledger=proposal_ledger,
        review_bundle=review_bundle,
        artifact_filenames=(
            artifacts.results_markdown,
            artifacts.convergence_plot,
            artifacts.candidate_rankings_plot,
            artifacts.prior_graph_plot,
            artifacts.posterior_graph_plot,
            artifacts.belief_delta_summary,
            artifacts.proposal_ledger,
        ),
    )
    _write_convergence_plot(
        convergence_path,
        observations=observations,
        baseline_utility=summary.objective_baseline,
    )
    _write_candidate_rankings_plot(
        candidate_rankings_path,
        ranked_candidates=ranked,
    )
    prior_nodes, prior_edges = _build_prior_graph(compiled)
    _draw_graph(
        prior_graph_path,
        title="Belief Graph: Prior Structure",
        nodes=prior_nodes,
        edges=prior_edges,
    )
    _draw_graph(
        posterior_graph_path,
        title="Belief Graph: Posterior Interactions",
        nodes=summary.graph.nodes,
        edges=_posterior_graph_edges(summary.graph),
    )
    return {
        "results_markdown": str(results_path),
        "convergence_plot": str(convergence_path),
        "candidate_rankings_plot": str(candidate_rankings_path),
        "prior_graph_plot": str(prior_graph_path),
        "posterior_graph_plot": str(posterior_graph_path),
    }
