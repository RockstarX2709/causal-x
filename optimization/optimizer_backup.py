import numpy as np
import pandas as pd

from optimization.objective import (
    RiskAdjustedObjective
)

from optimization.constraints import (
    BusinessConstraints
)


class DecisionOptimizer:

    def __init__(
        self,
        risk_aversion=0.25,
        simulations=5000,
        seed=42
    ):

        self.objective = (
            RiskAdjustedObjective(
                risk_aversion=risk_aversion
            )
        )

        self.constraints = (
            BusinessConstraints()
        )

        self.simulations = simulations

        self.rng = np.random.default_rng(
            seed
        )

    # =========================================================
    # CAUSAL DEMAND MODEL
    # =========================================================

    def estimate_demand(
        self,
        baseline_demand,
        baseline_price,
        new_price,
        elasticity,
        marketing_spend,
        baseline_marketing=300_000
    ):

        # -----------------------------------------------------
        # PRICE EFFECT
        #
        # Demand_new / Demand_old
        # =
        # (Price_new / Price_old)^elasticity
        # -----------------------------------------------------

        price_ratio = (
            new_price
            / baseline_price
        )

        price_multiplier = (
            price_ratio
            ** elasticity
        )

        # -----------------------------------------------------
        # MARKETING EFFECT
        #
        # Diminishing returns
        # -----------------------------------------------------

        baseline_marketing_effect = (
            np.log1p(
                baseline_marketing
                / 100_000
            )
        )

        marketing_effect = (
            np.log1p(
                marketing_spend
                / 100_000
            )
        )

        marketing_multiplier = (
            1
            + 0.035
            * (
                marketing_effect
                - baseline_marketing_effect
            )
        )

        marketing_multiplier = max(
            marketing_multiplier,
            0.5
        )

        expected_demand = (
            baseline_demand
            * price_multiplier
            * marketing_multiplier
        )

        return max(
            expected_demand,
            0
        )

    # =========================================================
    # SIMULATE DECISION
    # =========================================================

    def simulate_decision(
        self,
        baseline_price,
        baseline_demand,
        baseline_cost,
        elasticity,
        price_change,
        marketing_spend,
        demand_noise=None,
        cost_noise=None,
    ):

        new_price = (
            baseline_price
            * (1 + price_change)
        )

        expected_demand = (
            self.estimate_demand(
                baseline_demand=
                    baseline_demand,

                baseline_price=
                    baseline_price,

                new_price=
                    new_price,

                elasticity=
                    elasticity,

                marketing_spend=
                    marketing_spend,
            )
        )

        # -----------------------------------------------------
        # DEMAND UNCERTAINTY
        # -----------------------------------------------------

        if demand_noise is None:

            demand_noise = (
                self.rng.normal(
                    1.0,
                    0.08,
                    self.simulations
                )
            )

        if cost_noise is None:

            cost_noise = (
                self.rng.normal(
                    1.0,
                    0.04,
                    self.simulations
                )
            )

        simulated_demand = (
            expected_demand
            * demand_noise
        )

        simulated_demand = np.maximum(
            simulated_demand,
            0
        )

        # -----------------------------------------------------
        # SALES
        # -----------------------------------------------------

        simulated_sales = (
            simulated_demand
        )

        # -----------------------------------------------------
        # REVENUE
        # -----------------------------------------------------

        revenue = (
            new_price
            * simulated_sales
        )

        # -----------------------------------------------------
        # COST
        # -----------------------------------------------------

        variable_cost = (
            baseline_cost
            * simulated_sales
            * cost_noise
        )

        marketing_cost = (
            marketing_spend
        )

        logistics_cost = (
            simulated_sales
            * 150
        )

        # -----------------------------------------------------
        # PROFIT
        # -----------------------------------------------------

        profits = (
            revenue
            - variable_cost
            - marketing_cost
            - logistics_cost
        )

        objective_result = (
            self.objective.calculate(
                profits
            )
        )

        return {
            "price":
                new_price,

            "expected_demand":
                expected_demand,

            "profits":
                profits,

            "expected_profit":
                objective_result[
                    "expected_profit"
                ],

            "CVaR_95":
                objective_result[
                    "cvar_95"
                ],

            "probability_of_loss":
                objective_result[
                    "probability_of_loss"
                ],

            "profit_p05":
                objective_result[
                    "profit_p05"
                ],

            "profit_tail_5":
                objective_result[
                    "profit_tail_5"
                ],

            "objective":
                objective_result[
                    "objective"
                ],
        }

    # =========================================================
    # OPTIMIZE
    # =========================================================

    def optimize(
        self,
        baseline_price,
        baseline_demand,
        baseline_cost,
        elasticity,
        max_marketing_spend=500_000,
    ):

        candidates = []

        # -----------------------------------------------------
        # COMMON RANDOM NUMBERS
        #
        # Every candidate decision is evaluated against the
        # same uncertainty scenarios. This makes comparisons
        # reproducible and statistically fair.
        # -----------------------------------------------------

        rng = np.random.default_rng(42)

        demand_noise = rng.normal(
            1.0,
            0.08,
            self.simulations
        )

        cost_noise = rng.normal(
            1.0,
            0.04,
            self.simulations
        )

        # -----------------------------------------------------
        # SEARCH SPACE
        # -----------------------------------------------------

        price_changes = np.arange(
            -0.10,
            0.101,
            0.01
        )

        max_marketing_spend = max(
            100_000,
            min(
                max_marketing_spend,
                1_000_000
            )
        )

        marketing_values = np.arange(
            100_000,
            max_marketing_spend + 1,
            50_000
        )

        # -----------------------------------------------------
        # GRID SEARCH
        # -----------------------------------------------------

        for price_change in (
            price_changes
        ):

            for marketing in (
                marketing_values
            ):

                new_price = (
                    baseline_price
                    * (1 + price_change)
                )

                if not self.constraints.valid_price(
                    new_price,
                    baseline_price
                ):
                    continue

                result = (
                    self.simulate_decision(
                        baseline_price=
                            baseline_price,

                        baseline_demand=
                            baseline_demand,

                        baseline_cost=
                            baseline_cost,

                        elasticity=
                            elasticity,

                        price_change=
                            price_change,

                        marketing_spend=
                            marketing,

                        demand_noise=
                            demand_noise,

                        cost_noise=
                            cost_noise,
                    )
                )

                candidates.append(
                    {
                        "price_change":
                            price_change,

                        "price":
                            result["price"],

                        "marketing_spend":
                            marketing,

                        "elasticity":
                            elasticity,

                        "expected_demand":
                            result[
                                "expected_demand"
                            ],

                        "expected_profit":
                            result[
                                "expected_profit"
                            ],

                        "CVaR_95":
                            result[
                                "CVaR_95"
                            ],

                        "probability_of_loss":
                            result[
                                "probability_of_loss"
                            ],

                        "profit_p05":
                            result[
                                "profit_p05"
                            ],

                        "profit_tail_5":
                            result[
                                "profit_tail_5"
                            ],

                        "objective":
                            result[
                                "objective"
                            ],
                    }
                )

        results = pd.DataFrame(
            candidates
        )

        if results.empty:

            raise RuntimeError(
                "No feasible decisions found."
            )

        # -----------------------------------------------------
        # BEST DECISION
        # -----------------------------------------------------

        results = results.sort_values(
            "objective",
            ascending=False
        ).reset_index(
            drop=True
        )

        best = (
            results.iloc[0]
            .to_dict()
        )

        return {
            "best_decision":
                best,

            "all_decisions":
                results,
        }


if __name__ == "__main__":

    # Example only.
    # In the real pipeline this value comes
    # from the causal estimator.

    optimizer = DecisionOptimizer(
        risk_aversion=0.25,
        simulations=5000,
        seed=42
    )

    result = optimizer.optimize(
        baseline_price=10_000,
        baseline_demand=1_500,
        baseline_cost=5_200,
        elasticity=-1.40,
    )

    best = result[
        "best_decision"
    ]

    print("=" * 70)
    print("CAUSAL-X CAUSAL OPTIMIZER")
    print("=" * 70)

    print(
        f"\nCausal elasticity: "
        f"{best['elasticity']:.3f}"
    )

    print(
        f"Price: "
        f"₹{best['price']:,.2f}"
    )

    print(
        f"Price change: "
        f"{best['price_change'] * 100:.1f}%"
    )

    print(
        f"Marketing spend: "
        f"₹{best['marketing_spend']:,.2f}"
    )

    print(
        f"Expected demand: "
        f"{best['expected_demand']:,.2f}"
    )

    print(
        f"Expected profit: "
        f"₹{best['expected_profit']:,.2f}"
    )

    print(
        f"CVaR 95%: "
        f"₹{best['CVaR_95']:,.2f}"
    )

    print(
        f"Risk-adjusted objective: "
        f"₹{best['objective']:,.2f}"
    )
