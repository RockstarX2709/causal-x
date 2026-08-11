from causal.counterfactual import CounterfactualEngine


engine = CounterfactualEngine(
    "data/raw/manufacturing_regional.csv"
)


scenarios = [
    ("A", -0.03),
    ("A", -0.05),
    ("A", -0.10),
    ("A", 0.05),
    ("B", -0.05),
    ("C", -0.05),
]


for product, change in scenarios:

    result = engine.summarize(
        product,
        change
    )

    print("=" * 60)

    print(
        f"Product {product}: "
        f"Price change {change * 100:.1f}%"
    )

    print(
        f"Demand change: "
        f"{result['demand_change_pct']:.2f}%"
    )

    print(
        f"Profit change: "
        f"{result['profit_change_pct']:.2f}%"
    )