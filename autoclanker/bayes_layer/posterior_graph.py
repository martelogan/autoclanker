from __future__ import annotations

from autoclanker.bayes_layer.types import (
    CompiledPriorBundle,
    ObjectivePosterior,
    PosteriorGraph,
    PosteriorGraphEdge,
)


def _pair_ref_to_nodes(pair_ref: str) -> tuple[str, str]:
    pair_body = pair_ref.removeprefix("pair:")
    left, right = pair_body.split("+", maxsplit=1)
    return left, right


def build_posterior_graph(
    objective_posterior: ObjectivePosterior,
    compiled_priors: CompiledPriorBundle,
) -> PosteriorGraph:
    edges: list[PosteriorGraphEdge] = []
    nodes: set[str] = set()
    for feature in objective_posterior.features:
        if feature.target_kind != "pair_effect":
            continue
        left, right = _pair_ref_to_nodes(feature.feature_name)
        nodes.update((left, right))
        edges.append(
            PosteriorGraphEdge(
                source=left,
                target=right,
                weight=feature.posterior_mean,
                relation="posterior_pair_effect",
            )
        )
    for spec in compiled_priors.linkage_hints:
        left, right = _pair_ref_to_nodes(spec.item.target_ref)
        nodes.update((left, right))
        edges.append(
            PosteriorGraphEdge(
                source=left,
                target=right,
                weight=spec.item.mean,
                relation="graph_directive",
            )
        )
    return PosteriorGraph(
        nodes=tuple(sorted(nodes)),
        edges=tuple(
            sorted(
                edges,
                key=lambda edge: (edge.source, edge.target, edge.relation),
            )
        ),
    )
