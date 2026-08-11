from causal.log_causal_effect import LogCausalEffectEstimator


DATA_PATH = "data/raw/manufacturing_regional.csv"


def test_causal_estimator_runs():
    estimator = LogCausalEffectEstimator(DATA_PATH)

    result = estimator.estimate_elasticity("A")

    assert "elasticity" in result
    assert "r_squared" in result
    assert result["elasticity"] < 0
    assert 0 <= result["r_squared"] <= 1


def test_price_intervention_returns_demand():
    estimator = LogCausalEffectEstimator(DATA_PATH)

    result = estimator.simulate_price_intervention(
        product="A",
        price_change=-0.05
    )

    assert result["baseline_demand"] > 0
    assert result["counterfactual_demand"] > 0
    assert result["demand_change_pct"] > 0
