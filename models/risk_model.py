import numpy as np


class RiskModel:

    """
    CAUSAL-X Stress-Tested Risk Engine

    Simulates business uncertainty using multiple interacting
    shocks instead of assuming a single normal profit distribution.

    Shock sources:
        - demand uncertainty
        - production cost inflation
        - logistics cost inflation
        - supplier disruption
        - price-response uncertainty
    """

    def __init__(
        self,
        simulations=10_000,
        seed=42
    ):

        self.simulations = simulations

        self.rng = np.random.default_rng(
            seed
        )

    # =========================================================
    # BUSINESS SHOCK SIMULATION
    # =========================================================

    def simulate_profit(
        self,
        expected_profit,
        volatility=0.10
    ):

        base_profit = abs(
            expected_profit
        )

        # -----------------------------------------------------
        # 1. DEMAND SHOCK
        # -----------------------------------------------------

        demand_shock = self.rng.normal(
            loc=1.0,
            scale=volatility,
            size=self.simulations
        )

        # -----------------------------------------------------
        # 2. COST INFLATION
        # -----------------------------------------------------

        cost_shock = self.rng.normal(
            loc=1.0,
            scale=0.04,
            size=self.simulations
        )

        # -----------------------------------------------------
        # 3. LOGISTICS SHOCK
        # -----------------------------------------------------

        logistics_shock = self.rng.normal(
            loc=1.0,
            scale=0.05,
            size=self.simulations
        )

        # -----------------------------------------------------
        # 4. SUPPLIER DISRUPTION
        #
        # Small probability of a major negative event.
        # -----------------------------------------------------

        supplier_event = (
            self.rng.random(
                self.simulations
            ) < 0.03
        )

        supplier_multiplier = np.where(
            supplier_event,
            0.82,
            1.0
        )

        # -----------------------------------------------------
        # 5. PRICE-RESPONSE UNCERTAINTY
        # -----------------------------------------------------

        elasticity_noise = self.rng.normal(
            loc=1.0,
            scale=0.05,
            size=self.simulations
        )

        # -----------------------------------------------------
        # COMBINED BUSINESS EFFECT
        # -----------------------------------------------------

        revenue_multiplier = (
            demand_shock
            * elasticity_noise
            * supplier_multiplier
        )

        cost_multiplier = (
            cost_shock
            * logistics_shock
        )

        simulated_profits = (
            base_profit
            * revenue_multiplier
            - (
                base_profit
                * 0.35
                * (
                    cost_multiplier - 1.0
                )
            )
        )

        return simulated_profits

    # =========================================================
    # VALUE AT RISK
    # =========================================================

    def value_at_risk(
        self,
        profits,
        confidence=0.95
    ):

        profits = np.asarray(
            profits
        )

        return np.quantile(
            profits,
            1.0 - confidence
        )

    # =========================================================
    # CONDITIONAL VALUE AT RISK
    #
    # This returns the average profit in the worst tail.
    # Lower values indicate worse downside exposure.
    # =========================================================

    def conditional_value_at_risk(
        self,
        profits,
        confidence=0.95
    ):

        profits = np.asarray(
            profits
        )

        threshold = np.quantile(
            profits,
            1.0 - confidence
        )

        tail = profits[
            profits <= threshold
        ]

        if len(tail) == 0:

            return threshold

        return np.mean(
            tail
        )

    # =========================================================
    # PROBABILITY OF LOSS
    # =========================================================

    def probability_of_loss(
        self,
        profits
    ):

        return np.mean(
            profits < 0
        )

    # =========================================================
    # STRESS TEST
    # =========================================================

    def stress_test(
        self,
        expected_profit
    ):

        base = abs(
            expected_profit
        )

        scenarios = {

            "Normal":
                base,

            "Adverse":
                base
                * 0.75,

            "Severe":
                base
                * 0.55,
        }

        return scenarios

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

        var = self.value_at_risk(
            profits
        )

        cvar = (
            self.conditional_value_at_risk(
                profits
            )
        )

        probability_loss = (
            self.probability_of_loss(
                profits
            )
        )

        stress = self.stress_test(
            expected_profit
        )

        if probability_loss < 0.05:

            risk_level = "LOW"

        elif probability_loss < 0.15:

            risk_level = "MEDIUM"

        else:

            risk_level = "HIGH"

        return {

            "expected_profit":
                np.mean(profits),

            "profit_std":
                np.std(profits),

            "VaR_95":
                var,

            "CVaR_95":
                cvar,

            "probability_of_loss":
                probability_loss,

            "worst_case":
                np.min(profits),

            "best_case":
                np.max(profits),

            "risk_level":
                risk_level,

            "stress_normal":
                stress["Normal"],

            "stress_adverse":
                stress["Adverse"],

            "stress_severe":
                stress["Severe"],
        }


if __name__ == "__main__":

    model = RiskModel(
        simulations=10_000,
        seed=42
    )

    report = model.analyze(
        expected_profit=6_813_118,
        volatility=0.10
    )

    print("=" * 70)
    print(
        "CAUSAL-X STRESS-TESTED RISK ENGINE"
    )
    print("=" * 70)

    print(
        f"\nExpected profit: "
        f"₹{report['expected_profit']:,.2f}"
    )

    print(
        f"Profit standard deviation: "
        f"₹{report['profit_std']:,.2f}"
    )

    print(
        f"VaR 95%: "
        f"₹{report['VaR_95']:,.2f}"
    )

    print(
        f"CVaR 95%: "
        f"₹{report['CVaR_95']:,.2f}"
    )

    print(
        f"Probability of loss: "
        f"{report['probability_of_loss'] * 100:.2f}%"
    )

    print(
        f"Risk level: "
        f"{report['risk_level']}"
    )

    print("\nSTRESS TEST")

    print(
        f"Normal: "
        f"₹{report['stress_normal']:,.2f}"
    )

    print(
        f"Adverse: "
        f"₹{report['stress_adverse']:,.2f}"
    )

    print(
        f"Severe: "
        f"₹{report['stress_severe']:,.2f}"
    )

    print(
        "\n" + "=" * 70
    )
