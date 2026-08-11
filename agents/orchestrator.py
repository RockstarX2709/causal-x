import pandas as pd

from agents.analyst_agent import (
    AnalystAgent
)

from agents.causal_agent import (
    CausalAgent
)

from agents.forecast_agent import (
    ForecastAgent
)

from agents.optimizer_agent import (
    OptimizerAgent
)

from agents.risk_agent import (
    RiskAgent
)

from agents.auditor_agent import (
    AuditorAgent
)


class CausalXOrchestrator:

    def __init__(
        self,
        data_path
    ):

        self.data_path = data_path

        self.df = pd.read_csv(
            data_path
        )

        self.analyst = (
            AnalystAgent(
                data_path
            )
        )

        self.causal = (
            CausalAgent(
                data_path
            )
        )

        self.forecaster = (
            ForecastAgent(
                data_path
            )
        )

        self.optimizer = (
            OptimizerAgent()
        )

        self.risk = RiskAgent()

        self.auditor = (
            AuditorAgent()
        )

    # =========================================================
    # DECISION PIPELINE
    # =========================================================

    def solve(
        self,
        product,
        region,
        price_change=-0.05,
        marketing_budget=300_000
    ):

        # -----------------------------------------------------
        # 1. BUSINESS ANALYSIS
        # -----------------------------------------------------

        analysis = (
            self.analyst
            .analyze_product(
                product
            )
        )

        # -----------------------------------------------------
        # 2. CAUSAL ANALYSIS
        # -----------------------------------------------------

        causal = (
            self.causal.analyze(
                product=product,
                price_change=price_change
            )
        )

        learned_elasticity = (
            causal[
                "estimated_elasticity"
            ]
        )

        # -----------------------------------------------------
        # 3. FORECAST
        # -----------------------------------------------------

        forecast = (
            self.forecaster.forecast(
                product=product,
                region=region
            )
        )

        # -----------------------------------------------------
        # 4. BASELINE VALUES
        # -----------------------------------------------------

        baseline_price = (
            analysis[
                "average_price"
            ]
        )

        baseline_demand = (
            analysis[
                "average_demand"
            ]
        )

        product_df = self.df[
            self.df["product"]
            == product
        ]

        baseline_cost = (
            product_df[
                "production_cost"
            ].mean()
            / max(
                baseline_demand,
                1
            )
        )

        # -----------------------------------------------------
        # 5. CAUSAL OPTIMIZATION
        # -----------------------------------------------------

        optimization = (
            self.optimizer.optimize(

                baseline_price=
                    baseline_price,

                baseline_demand=
                    baseline_demand,

                baseline_cost=
                    baseline_cost,

                elasticity=
                    learned_elasticity,

                max_marketing_spend=
                    marketing_budget,
            )
        )

        # -----------------------------------------------------
        # 6. RISK
        # -----------------------------------------------------

        risk = (
            self.risk.analyze(
                expected_profit=
                    optimization[
                        "expected_profit"
                    ],

                volatility=0.10
            )
        )

        # -----------------------------------------------------
        # 7. AUDIT
        # -----------------------------------------------------

        audit = (
            self.auditor.audit(

                causal_result=
                    causal,

                forecast_result=
                    forecast,

                optimization_result=
                    optimization,
            )
        )

        # -----------------------------------------------------
        # 8. RETURN COMPLETE EVIDENCE
        # -----------------------------------------------------

        return {
            "business_analysis":
                analysis,

            "causal_analysis":
                causal,

            "forecast":
                forecast,

            "optimization":
                optimization,

            "risk":
                risk,

            "audit":
                audit,
        }


if __name__ == "__main__":

    orchestrator = CausalXOrchestrator(
        "data/raw/manufacturing_regional.csv"
    )

    result = orchestrator.solve(
        product="A",
        region="North"
    )

    print("=" * 75)
    print(
        "CAUSAL-X INTEGRATED DECISION SYSTEM"
    )
    print("=" * 75)

    print("\nCAUSAL MODEL")
    print("-" * 75)

    causal = result[
        "causal_analysis"
    ]

    print(
        f"Estimated elasticity: "
        f"{causal['estimated_elasticity']:.4f}"
    )

    print(
        f"R²: "
        f"{causal['r_squared']:.4f}"
    )

    print("\nOPTIMIZER")
    print("-" * 75)

    optimization = result[
        "optimization"
    ]

    print(
        f"Elasticity supplied to optimizer: "
        f"{optimization['elasticity']:.4f}"
    )

    print(
        f"Recommended price: "
        f"₹{optimization['price']:,.2f}"
    )

    print(
        f"Price change: "
        f"{optimization['price_change'] * 100:.2f}%"
    )

    print(
        f"Marketing spend: "
        f"₹{optimization['marketing_spend']:,.2f}"
    )

    print(
        f"Expected demand: "
        f"{optimization['expected_demand']:,.2f}"
    )

    print(
        f"Expected profit: "
        f"₹{optimization['expected_profit']:,.2f}"
    )

    print("\nRISK")
    print("-" * 75)

    risk = result[
        "risk"
    ]

    print(
        f"Risk level: "
        f"{risk['risk_level']}"
    )

    print(
        f"Probability of loss: "
        f"{risk['probability_of_loss'] * 100:.2f}%"
    )

    print(
        f"CVaR 95%: "
        f"₹{risk['CVaR_95']:,.2f}"
    )

    print("\nAUDIT")
    print("-" * 75)

    audit = result[
        "audit"
    ]

    print(
        f"Audit score: "
        f"{audit['audit_score']}/100"
    )

    print(
        f"Confidence: "
        f"{audit['confidence']}"
    )

    print(
        f"Approved: "
        f"{audit['approved']}"
    )

    if audit["warnings"]:

        print("\nWarnings:")

        for warning in (
            audit["warnings"]
        ):

            print(
                f"  • {warning}"
            )

    print(
        "\n" + "=" * 75
    )
