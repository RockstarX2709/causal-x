from optimization.optimizer import DecisionOptimizer


def test_optimizer_returns_decision():
    optimizer = DecisionOptimizer(
        risk_aversion=0.25,
        simulations=1000,
        seed=42
    )

    result = optimizer.optimize(
        baseline_price=10_000,
        baseline_demand=1_500,
        baseline_cost=5_200,
        elasticity=-1.40,
        max_marketing_spend=500_000
    )

    best = result["best_decision"]

    assert best["price"] > 0
    assert best["expected_demand"] > 0
    assert best["expected_profit"] > 0
    assert "probability_of_loss" in best
    assert "CVaR_95" in best


def test_optimizer_returns_all_decisions():
    optimizer = DecisionOptimizer(
        risk_aversion=0.25,
        simulations=500,
        seed=42
    )

    result = optimizer.optimize(
        baseline_price=10_000,
        baseline_demand=1_500,
        baseline_cost=5_200,
        elasticity=-1.40,
        max_marketing_spend=500_000
    )

    decisions = result["all_decisions"]

    assert not decisions.empty
    assert len(decisions) > 0
