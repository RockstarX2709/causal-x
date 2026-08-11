from optimization.optimizer import (
    DecisionOptimizer
)


class OptimizerAgent:

    def __init__(
        self,
        risk_aversion=0.25
    ):

        self.optimizer = (
            DecisionOptimizer(
                risk_aversion=
                    risk_aversion,

                simulations=5000,

                seed=42
            )
        )

    def optimize(
        self,
        baseline_price,
        baseline_demand,
        baseline_cost,
        elasticity,
        max_marketing_spend=500_000
    ):

        result = (
            self.optimizer.optimize(
                baseline_price=
                    baseline_price,

                baseline_demand=
                    baseline_demand,

                baseline_cost=
                    baseline_cost,

                elasticity=
                    elasticity,

                max_marketing_spend=
                    max_marketing_spend,
            )
        )

        best = result[
            "best_decision"
        ]

        return {

            # =================================================
            # BEST DECISION
            # =================================================

            "price":
                best["price"],

            "price_change":
                best["price_change"],

            "marketing_spend":
                best[
                    "marketing_spend"
                ],

            "elasticity":
                best[
                    "elasticity"
                ],

            "expected_demand":
                best[
                    "expected_demand"
                ],

            "expected_profit":
                best[
                    "expected_profit"
                ],

            # =================================================
            # RISK
            # =================================================

            "cvar_95":
                best[
                    "CVaR_95"
                ],

            "probability_of_loss":
                best[
                    "probability_of_loss"
                ],

            "profit_p05":
                best[
                    "profit_p05"
                ],

            "profit_tail_5":
                best[
                    "profit_tail_5"
                ],

            # =================================================
            # OPTIMIZATION OBJECTIVE
            # =================================================

            "objective":
                best[
                    "objective"
                ],

            # =================================================
            # FULL DECISION LANDSCAPE
            #
            # Used by Scenario Explorer.
            # =================================================

            "all_decisions":
                result[
                    "all_decisions"
                ],
        }


if __name__ == "__main__":

    agent = OptimizerAgent()

    result = agent.optimize(
        baseline_price=10_000,
        baseline_demand=1_500,
        baseline_cost=5_200,
        elasticity=-1.40
    )

    print("=" * 65)
    print(
        "CAUSAL-X OPTIMIZER AGENT"
    )
    print("=" * 65)

    for key, value in result.items():

        if key == "all_decisions":

            print(
                f"{key}: "
                f"{value.shape}"
            )

        elif isinstance(
            value,
            float
        ):

            print(
                f"{key}: "
                f"{value:,.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )
