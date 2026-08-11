import numpy as np


class RiskModel:

    def __init__(
        self,
        simulations=10_000,
        seed=42
    ):

        self.simulations = simulations

        self.seed = seed

        self.rng = np.random.default_rng(
            seed
        )

    # =========================================================
    # MONTE CARLO PROFIT
    # =========================================================

    def simulate_profit(
        self,
        expected_profit,
        volatility=0.10
    ):

        expected_profit = float(
            expected_profit
        )

        scale = (
            abs(expected_profit)
            * volatility
        )

        simulated_profits = (
            self.rng.normal(
                loc=expected_profit,
                scale=scale,
                size=self.simulations
            )
        )

        return simulated_profits

    # =========================================================
    # PROFIT-BASED VaR
    #
    # VaR is reported as the 5th percentile profit.
    # Lower values represent worse outcomes.
    # =========================================================

    def value_at_risk(
        self,
        profits,
        confidence=0.95
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        lower_tail_probability = (
            1.0 - confidence
        )

        return float(
            np.quantile(
                profits,
                lower_tail_probability
            )
        )

    # =========================================================
    # DOWNSIDE CVaR
    #
    # Average actual loss in the worst 5% outcomes.
    #
    # If there are no losses:
    #
    #       CVaR = 0
    #
    # =========================================================

    def conditional_value_at_risk(
        self,
        profits,
        confidence=0.95
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        # -----------------------------------------------------
        # Actual losses only
        # -----------------------------------------------------

        losses = np.maximum(
            -profits,
            0.0
        )

        # -----------------------------------------------------
        # No actual losses
        # -----------------------------------------------------

        if np.all(
            losses == 0
        ):

            return 0.0

        # -----------------------------------------------------
        # Worst 5% loss threshold
        # -----------------------------------------------------

        threshold = np.quantile(
            losses,
            confidence
        )

        tail_losses = losses[
            losses >= threshold
        ]

        if len(tail_losses) == 0:

            return float(
                threshold
            )

        return float(
            np.mean(
                tail_losses
            )
        )

    # =========================================================
    # WORST 5% AVERAGE PROFIT
    # =========================================================

    def worst_5_percent_profit(
        self,
        profits
    ):

        profits = np.asarray(
            profits,
            dtype=float
        )

        threshold = np.quantile(
            profits,
            0.05
        )

        worst_tail = profits[
            profits <= threshold
        ]

        if len(worst_tail) == 0:

            return float(
                threshold
            )

        return float(
            np.mean(
                worst_tail
            )
        )

    # =========================================================
    # COMPLETE RISK REPORT
    # =========================================================

    def analyze(
        self,
        expected_profit,
        volatility=0.10
    ):

        profits = self.simulate_profit(
            expected_profit=
                expected_profit,

            volatility=
                volatility
        )

        expected_profit_realized = float(
            np.mean(
                profits
            )
        )

        profit_std = float(
            np.std(
                profits
            )
        )

        probability_of_loss = float(
            np.mean(
                profits < 0
            )
        )

        # -----------------------------------------------------
        # 5th percentile profit
        # -----------------------------------------------------

        profit_p05 = self.value_at_risk(
            profits
        )

        # -----------------------------------------------------
        # Average profit in worst 5%
        # -----------------------------------------------------

        worst_tail_profit = (
            self.worst_5_percent_profit(
                profits
            )
        )

        # -----------------------------------------------------
        # Downside CVaR
        # -----------------------------------------------------

        cvar = (
            self.conditional_value_at_risk(
                profits
            )
        )

        return {

            "expected_profit":
                expected_profit_realized,

            "profit_std":
                profit_std,

            # 5th percentile profit
            "VaR_95":
                profit_p05,

            # Positive downside loss measure
            "CVaR_95":
                cvar,

            "probability_of_loss":
                probability_of_loss,

            "profit_p05":
                profit_p05,

            "profit_tail_5":
                worst_tail_profit,

            "worst_case":
                float(
                    np.min(
                        profits
                    )
                ),

            "best_case":
                float(
                    np.max(
                        profits
                    )
                ),
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    model = RiskModel(
        simulations=10_000,
        seed=42
    )

    report = model.analyze(
        expected_profit=
            100_000_000,

        volatility=
            0.12
    )

    print("=" * 70)
    print(
        "CAUSAL-X RISK ENGINE"
    )
    print("=" * 70)

    print(
        f"\nExpected profit: "
        f"₹{report['expected_profit']:,.2f}"
    )

    print(
        f"Probability of loss: "
        f"{report['probability_of_loss'] * 100:.2f}%"
    )

    print(
        f"5th percentile profit: "
        f"₹{report['profit_p05']:,.2f}"
    )

    print(
        f"Worst 5% average profit: "
        f"₹{report['profit_tail_5']:,.2f}"
    )

    print(
        f"Downside CVaR 95%: "
        f"₹{report['CVaR_95']:,.2f}"
    )

    print(
        f"Worst case: "
        f"₹{report['worst_case']:,.2f}"
    )

    print(
        f"Best case: "
        f"₹{report['best_case']:,.2f}"
    )
