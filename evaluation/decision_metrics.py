import numpy as np


class DecisionMetrics:

    # =========================================================
    # REGRET
    # =========================================================

    @staticmethod
    def regret(
        optimal_profit,
        chosen_profit
    ):

        return (
            optimal_profit
            - chosen_profit
        )

    # =========================================================
    # REGRET PERCENTAGE
    # =========================================================

    @staticmethod
    def regret_percentage(
        optimal_profit,
        chosen_profit
    ):

        if optimal_profit == 0:
            return 0.0

        return (
            (
                optimal_profit
                - chosen_profit
            )
            / abs(optimal_profit)
        ) * 100

    # =========================================================
    # IMPROVEMENT
    # =========================================================

    @staticmethod
    def improvement(
        baseline_profit,
        optimized_profit
    ):

        if baseline_profit == 0:
            return 0.0

        return (
            (
                optimized_profit
                - baseline_profit
            )
            / abs(baseline_profit)
        ) * 100

    # =========================================================
    # RISK-ADJUSTED IMPROVEMENT
    # =========================================================

    @staticmethod
    def objective_improvement(
        baseline_objective,
        optimized_objective
    ):

        if baseline_objective == 0:
            return 0.0

        return (
            (
                optimized_objective
                - baseline_objective
            )
            / abs(baseline_objective)
        ) * 100

    # =========================================================
    # COMPLETE REPORT
    # =========================================================

    def evaluate(
        self,
        baseline_profit,
        chosen_profit,
        optimal_profit,
        baseline_objective=None,
        optimized_objective=None
    ):

        result = {
            "baseline_profit":
                baseline_profit,

            "chosen_profit":
                chosen_profit,

            "optimal_profit":
                optimal_profit,

            "regret":
                self.regret(
                    optimal_profit,
                    chosen_profit
                ),

            "regret_percentage":
                self.regret_percentage(
                    optimal_profit,
                    chosen_profit
                ),

            "profit_improvement":
                self.improvement(
                    baseline_profit,
                    chosen_profit
                ),
        }

        if (
            baseline_objective is not None
            and optimized_objective is not None
        ):

            result[
                "objective_improvement"
            ] = self.objective_improvement(
                baseline_objective,
                optimized_objective
            )

        return result


if __name__ == "__main__":

    metrics = DecisionMetrics()

    result = metrics.evaluate(
        baseline_profit=100_000_000,
        chosen_profit=112_000_000,
        optimal_profit=115_000_000,
    )

    print("=" * 65)
    print("CAUSAL-X DECISION METRICS")
    print("=" * 65)

    for key, value in result.items():

        if (
            "percentage"
            in key
            or "improvement"
            in key
        ):

            print(
                f"{key}: "
                f"{value:.2f}%"
            )

        else:

            print(
                f"{key}: "
                f"₹{value:,.2f}"
            )