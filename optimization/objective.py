import numpy as np


class RiskAdjustedObjective:

    def __init__(
        self,
        risk_aversion=0.25
    ):

        self.risk_aversion = risk_aversion

    # =========================================================
    # LOWER-TAIL CVaR
    #
    # CVaR here represents the average profit in the worst
    # 5% of simulated outcomes.
    # =========================================================

    def calculate_cvar(
        self,
        profits,
        confidence=0.95
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        if len(profits) == 0:
            return 0.0

        threshold = np.quantile(
            profits,
            1.0 - confidence
        )

        tail = profits[
            profits <= threshold
        ]

        if len(tail) == 0:
            return float(threshold)

        return float(
            np.mean(tail)
        )

    # =========================================================
    # PROFIT LOWER TAIL
    # =========================================================

    def calculate_profit_tail(
        self,
        profits,
        tail_probability=0.05
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        if len(profits) == 0:
            return 0.0

        threshold = np.quantile(
            profits,
            tail_probability
        )

        tail = profits[
            profits <= threshold
        ]

        if len(tail) == 0:
            return float(threshold)

        return float(
            np.mean(tail)
        )

    # =========================================================
    # OBJECTIVE
    # =========================================================

    def calculate(
        self,
        profits
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        if len(profits) == 0:

            raise ValueError(
                "Profit simulation is empty."
            )

        expected_profit = float(
            np.mean(profits)
        )

        probability_of_loss = float(
            np.mean(
                profits < 0
            )
        )

        profit_p05 = float(
            np.quantile(
                profits,
                0.05
            )
        )

        profit_tail = (
            self.calculate_profit_tail(
                profits
            )
        )

        cvar = (
            self.calculate_cvar(
                profits
            )
        )

        # -----------------------------------------------------
        # TAIL SHORTFALL
        #
        # Measures how far the worst 5% tail falls below
        # the expected profit.
        #
        # This remains meaningful even when all profits
        # are positive.
        # -----------------------------------------------------

        tail_shortfall = max(
            expected_profit - cvar,
            0.0
        )

        # -----------------------------------------------------
        # RISK-ADJUSTED OBJECTIVE
        # -----------------------------------------------------

        objective = (
            expected_profit
            - (
                self.risk_aversion
                * tail_shortfall
            )
        )

        return {

            "expected_profit":
                expected_profit,

            "probability_of_loss":
                probability_of_loss,

            "profit_p05":
                profit_p05,

            "profit_tail_5":
                profit_tail,

            "cvar_95":
                cvar,

            "tail_shortfall":
                float(tail_shortfall),

            "objective":
                float(objective),
        }

    # =========================================================
    # COMPARE TWO DECISIONS
    # =========================================================

    def compare(
        self,
        baseline_profits,
        scenario_profits
    ):

        baseline = self.calculate(
            baseline_profits
        )

        scenario = self.calculate(
            scenario_profits
        )

        return {

            "baseline":
                baseline,

            "scenario":
                scenario,

            "profit_change":
                (
                    scenario[
                        "expected_profit"
                    ]
                    -
                    baseline[
                        "expected_profit"
                    ]
                ),

            "cvar_change":
                (
                    scenario[
                        "cvar_95"
                    ]
                    -
                    baseline[
                        "cvar_95"
                    ]
                ),

            "objective_change":
                (
                    scenario[
                        "objective"
                    ]
                    -
                    baseline[
                        "objective"
                    ]
                ),
        }


if __name__ == "__main__":

    objective = RiskAdjustedObjective(
        risk_aversion=0.25
    )

    rng = np.random.default_rng(
        42
    )

    profits = rng.normal(
        100_000_000,
        12_000_000,
        10_000
    )

    result = objective.calculate(
        profits
    )

    print("=" * 70)
    print(
        "CAUSAL-X RISK-ADJUSTED OBJECTIVE"
    )
    print("=" * 70)

    print(
        f"\nExpected profit: "
        f"₹{result['expected_profit']:,.2f}"
    )

    print(
        f"Probability of loss: "
        f"{result['probability_of_loss'] * 100:.2f}%"
    )

    print(
        f"5th percentile profit: "
        f"₹{result['profit_p05']:,.2f}"
    )

    print(
        f"Worst 5% average profit: "
        f"₹{result['profit_tail_5']:,.2f}"
    )

    print(
        f"Downside CVaR 95%: "
        f"₹{result['cvar_95']:,.2f}"
    )

    print(
        f"Tail shortfall: "
        f"₹{result['tail_shortfall']:,.2f}"
    )

    print(
        f"Risk-adjusted objective: "
        f"₹{result['objective']:,.2f}"
    )
