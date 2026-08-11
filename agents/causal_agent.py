from causal.log_causal_effect import (
    LogCausalEffectEstimator
)

from causal.robustness import (
    CausalRobustness
)


class CausalAgent:

    def __init__(self, data_path):

        self.data_path = data_path

        self.estimator = (
            LogCausalEffectEstimator(
                data_path
            )
        )

        self.robustness = (
            CausalRobustness(
                data_path
            )
        )

    def analyze(
        self,
        product,
        price_change=-0.05
    ):

        # =====================================================
        # PRIMARY CAUSAL ESTIMATE
        # =====================================================

        result = (
            self.estimator
            .simulate_price_intervention(
                product=product,
                price_change=price_change
            )
        )

        # =====================================================
        # ROBUSTNESS
        # =====================================================

        robustness = (
            self.robustness.evaluate(
                product
            )
        )

        # =====================================================
        # COMBINE
        # =====================================================

        return {
            "product":
                product,

            "price_change":
                price_change,

            "estimated_elasticity":
                result[
                    "estimated_elasticity"
                ],

            "baseline_demand":
                result[
                    "baseline_demand"
                ],

            "counterfactual_demand":
                result[
                    "counterfactual_demand"
                ],

            "demand_change_pct":
                result[
                    "demand_change_pct"
                ],

            "r_squared":
                result[
                    "r_squared"
                ],

            # -------------------------------------------------
            # ROBUSTNESS
            # -------------------------------------------------

            "robustness":
                robustness[
                    "robustness"
                ],

            "bootstrap_ci_lower":
                robustness[
                    "ci_lower"
                ],

            "bootstrap_ci_upper":
                robustness[
                    "ci_upper"
                ],

            "bootstrap_std":
                robustness[
                    "bootstrap_std"
                ],

            "placebo_elasticity":
                robustness[
                    "placebo_elasticity"
                ],

            "sensitivity_with_controls":
                robustness[
                    "sensitivity_with_controls"
                ],

            "sensitivity_without_controls":
                robustness[
                    "sensitivity_without_controls"
                ],

            "sensitivity_difference":
                robustness[
                    "sensitivity_difference"
                ],
        }


if __name__ == "__main__":

    agent = CausalAgent(
        "data/raw/manufacturing_regional.csv"
    )

    for product in [
        "A",
        "B",
        "C"
    ]:

        result = agent.analyze(
            product,
            -0.05
        )

        print("=" * 70)

        print(
            f"PRODUCT {product}"
        )

        print(
            f"Elasticity: "
            f"{result['estimated_elasticity']:.4f}"
        )

        print(
            f"95% CI: "
            f"["
            f"{result['bootstrap_ci_lower']:.4f}, "
            f"{result['bootstrap_ci_upper']:.4f}"
            f"]"
        )

        print(
            f"Placebo: "
            f"{result['placebo_elasticity']:.4f}"
        )

        print(
            f"Robustness: "
            f"{result['robustness']}"
        )
