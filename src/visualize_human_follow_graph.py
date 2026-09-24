from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DISPLAY_NODE_COUNT = 50


def render_group(
    group_name: str,
    node_ids: set[int],
    follow_sources: list[int],
    follow_targets: list[int],
) -> None:
    group_edges = [
        (source, target)
        for source, target in zip(follow_sources, follow_targets)
        if source in node_ids and target in node_ids
    ]

    graph = nx.DiGraph()
    graph.add_nodes_from(node_ids)
    graph.add_edges_from(group_edges)
    ranked_nodes = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    sample_nodes = {node for node, _ in ranked_nodes[:DISPLAY_NODE_COUNT]}
    sampled_graph = graph.subgraph(sample_nodes).copy()

    positions = nx.spring_layout(sampled_graph, seed=42, k=1.8, iterations=100)
    degrees = dict(sampled_graph.degree())

    figure, axis = plt.subplots(figsize=(16, 12), facecolor="white")
    axis.set_facecolor("white")
    nx.draw_networkx_edges(
        sampled_graph,
        positions,
        ax=axis,
        edge_color="black",
        alpha=0.28,
        arrows=True,
        arrowsize=5,
        width=0.5,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        sampled_graph,
        positions,
        ax=axis,
        node_color="white",
        node_size=420,
        alpha=0.88,
        linewidths=0.25,
        edgecolors="black",
    )
    nx.draw_networkx_labels(
        sampled_graph,
        positions,
        ax=axis,
        labels={node: str(node) for node in sampled_graph},
        font_size=7,
        font_color="black",
    )
    axis.set_title(
        f"MGTAB {group_name.title()} Follow Directed Subgraph\n"
        f"Top {sampled_graph.number_of_nodes()} {group_name} accounts by degree",
        color="black",
        fontsize=16,
        fontweight="bold",
    )
    axis.text(
        0.01,
        0.01,
        f"{group_name.title()} accounts: {graph.number_of_nodes():,} | "
        f"{group_name.title()} follow edges: {graph.number_of_edges():,} | "
        f"Displayed edges: {sampled_graph.number_of_edges():,}",
        transform=axis.transAxes,
        color="black",
        fontsize=10,
    )
    axis.axis("off")
    output_path = PROJECT_DIR / f"mgtab_{group_name}_follow_digraph.png"
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"MGTAB {group_name} accounts: {graph.number_of_nodes():,}")
    print(f"MGTAB {group_name} follow edges: {graph.number_of_edges():,}")
    print(f"Sampled graph: {sampled_graph.number_of_nodes():,} nodes, "
          f"{sampled_graph.number_of_edges():,} edges")
    print(f"Saved visualization to {output_path}")


def main() -> None:
    edge_index = torch.load(DATA_DIR / "edge_index.pt", weights_only=False)
    edge_type = torch.load(DATA_DIR / "edge_type.pt", weights_only=False)
    labels_bot = torch.load(DATA_DIR / "labels_bot.pt", weights_only=False)

    follow_mask = edge_type == 0
    follow_sources = edge_index[0][follow_mask].tolist()
    follow_targets = edge_index[1][follow_mask].tolist()

    render_group(
        "human",
        set((labels_bot == 0).nonzero(as_tuple=True)[0].tolist()),
        follow_sources,
        follow_targets,
    )
    render_group(
        "bot",
        set((labels_bot == 1).nonzero(as_tuple=True)[0].tolist()),
        follow_sources,
        follow_targets,
    )
    render_group(
        "all",
        set(range(len(labels_bot))),
        follow_sources,
        follow_targets,
    )


if __name__ == "__main__":
    main()