import networkx as nx


def build_causal_graph():
    """
    Ground-truth causal graph for the CAUSAL-X
    manufacturing environment.
    """

    graph = nx.DiGraph()

    edges = [
        ("competitor_price", "demand"),
        ("price", "demand"),
        ("marketing_spend", "demand"),

        ("demand", "sales"),
        ("production", "sales"),

        ("production", "closing_inventory"),
        ("sales", "closing_inventory"),

        ("avg_machine_health", "production"),

        ("supplier_delayed", "lead_time"),
        ("lead_time", "production"),

        ("price", "revenue"),
        ("sales", "revenue"),

        ("production", "production_cost"),
        ("closing_inventory", "inventory_cost"),
        ("sales", "logistics_cost"),

        ("revenue", "profit"),
        ("production_cost", "profit"),
        ("marketing_spend", "profit"),
        ("inventory_cost", "profit"),
        ("logistics_cost", "profit"),
        ("maintenance_cost", "profit"),
    ]

    graph.add_edges_from(edges)

    return graph


if __name__ == "__main__":

    graph = build_causal_graph()

    print("CAUSAL-X Causal Graph")
    print("=" * 40)

    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")

    print("\nCausal relationships:")

    for source, target in graph.edges:
        print(f"{source} -> {target}")