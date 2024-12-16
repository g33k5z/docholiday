from langgraph.graph.state import CompiledStateGraph


def save_mermaid_graph(
    graph: CompiledStateGraph, filename: str = "diagrams/graph.mermaid"
) -> str:
    """Return the graph in Mermaid (crows feet) format."""
    m = graph.get_graph(xray=True).draw_mermaid()

    with open(filename, "w") as f:
        f.write(m)

    return m
