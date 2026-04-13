from __future__ import annotations

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import math

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from autoclanker.bayes_layer.session_store import FilesystemSessionStore
from autoclanker.bayes_layer.types import (
    CommitDecision,
    CompiledPriorBundle,
    PosteriorGraph,
    PosteriorSummary,
    QuerySuggestion,
    RankedCandidate,
    SessionManifest,
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


def _write_results_markdown(
    path: Path,
    *,
    manifest: SessionManifest,
    summary: PosteriorSummary,
    ranked_candidates: Sequence[RankedCandidate],
    queries: Sequence[QuerySuggestion],
    decision: CommitDecision | None,
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

    lines.extend(["", "## Artifacts"])
    lines.extend(f"- `{filename}`" for filename in artifact_filenames)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_session_report_bundle(
    *,
    store: FilesystemSessionStore,
    session_id: str,
    manifest: SessionManifest,
    compiled: CompiledPriorBundle,
    observations: Sequence[ValidEvalResult],
    summary: PosteriorSummary,
    ranked_candidates: Sequence[RankedCandidate] | None = None,
    queries: Sequence[QuerySuggestion] = (),
    decision: CommitDecision | None = None,
) -> dict[str, str]:
    artifacts = store.artifact_filenames
    ranked = tuple(ranked_candidates or summary.top_candidates)

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
        artifact_filenames=(
            artifacts.results_markdown,
            artifacts.convergence_plot,
            artifacts.candidate_rankings_plot,
            artifacts.prior_graph_plot,
            artifacts.posterior_graph_plot,
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
