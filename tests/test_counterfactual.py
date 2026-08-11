from causal.counterfactual import CounterfactualEngine


DATA_PATH = "data/raw/manufacturing_regional.csv"


def test_counterfactual_engine_runs():
    engine = CounterfactualEngine(DATA_PATH)

    result = engine.summarize(
        product="A",
        price_change_pct=-0.05
    )

    assert "demand_change_pct" in result
    assert "profit_change_pct" in result


def test_price_decrease_increases_demand():
    engine = CounterfactualEngine(DATA_PATH)

    result = engine.summarize(
        product="A",
        price_change_pct=-0.05
    )

    assert result["demand_change_pct"] > 0
