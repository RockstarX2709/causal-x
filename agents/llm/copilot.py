
import os

from dotenv import load_dotenv

from agents.llm.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

from agents.llm.schemas import (
    DecisionRecommendation,
)

from agents.llm.evidence import (
    EvidenceBuilder,
)


class DecisionCopilot:

    def __init__(self, model=None):

        load_dotenv()

        self.model = (
            model
            or os.getenv(
                "OPENAI_MODEL",
                "gpt-5.5"
            )
        )

        self.mode = os.getenv(
            "LLM_MODE",
            "mock"
        ).lower()

        self.client = None

        if self.mode == "openai":

            from openai import OpenAI

            self.client = OpenAI()

    # =========================================================
    # MOCK DECISION ENGINE
    # =========================================================

    def _mock_generate(
        self,
        question,
        business_analysis,
        causal_analysis,
        forecast,
        optimization,
        risk,
        audit,
    ):

        price_change = (
            optimization.get(
                "price_change",
                0
            )
        )

        price = optimization.get(
            "price",
            business_analysis.get(
                "average_price",
                0
            )
        )

        marketing = optimization.get(
            "marketing_spend",
            0
        )

        expected_demand = optimization.get(
            "expected_demand",
            0
        )

        expected_profit = optimization.get(
            "expected_profit",
            0
        )

        elasticity = causal_analysis.get(
            "estimated_elasticity",
            0
        )

        demand_change = causal_analysis.get(
            "demand_change_pct",
            0
        )

        forecast_mae = forecast.get(
            "MAE",
            0
        )

        risk_level = risk.get(
            "risk_level",
            "UNKNOWN"
        )

        probability_loss = risk.get(
            "probability_of_loss",
            0
        )

        audit_score = audit.get(
            "audit_score",
            0
        )

        confidence = audit.get(
            "confidence",
            "LOW"
        )

        warnings = audit.get(
            "warnings",
            []
        )

        # -----------------------------------------------------
        # RECOMMENDATION
        # -----------------------------------------------------

        if price_change < 0:

            recommendation = (
                f"Reduce the price to "
                f"₹{price:,.2f}, corresponding "
                f"to a {abs(price_change) * 100:.1f}% "
                f"price reduction, while allocating "
                f"₹{marketing:,.2f} to marketing."
            )

        elif price_change > 0:

            recommendation = (
                f"Increase the price to "
                f"₹{price:,.2f}, corresponding "
                f"to a {price_change * 100:.1f}% "
                f"price increase, while allocating "
                f"₹{marketing:,.2f} to marketing."
            )

        else:

            recommendation = (
                f"Maintain the current pricing "
                f"strategy and allocate "
                f"₹{marketing:,.2f} to marketing."
            )

        rationale = [
            (
                f"The causal model estimates a price "
                f"elasticity of {elasticity:.3f}."
            ),

            (
                f"The tested causal intervention "
                f"corresponds to an estimated "
                f"{demand_change:.2f}% demand change."
            ),

            (
                f"The optimizer estimates expected "
                f"demand of {expected_demand:,.2f}."
            ),

            (
                f"The risk-adjusted optimization "
                f"produces expected profit of "
                f"₹{expected_profit:,.2f}."
            ),
        ]

        causal_evidence = (
            f"The estimated price elasticity is "
            f"{elasticity:.3f}. Under the model's "
            f"identification assumptions, the tested "
            f"price intervention changes demand by "
            f"{demand_change:.2f}%."
        )

        forecast_evidence = (
            f"The demand forecasting model has a "
            f"backtest MAE of {forecast_mae:,.2f}. "
            f"The forecast should therefore be treated "
            f"as predictive evidence rather than causal "
            f"evidence."
        )

        risk_assessment = (
            f"Risk level is {risk_level}. "
            f"The simulated probability of loss is "
            f"{probability_loss * 100:.2f}%."
        )

        audit_assessment = (
            f"The recommendation received an audit "
            f"score of {audit_score}/100 with "
            f"{confidence} confidence."
        )

        caveats = [
            (
                "The recommendation is decision support "
                "and should not be executed automatically."
            ),

            (
                "Causal conclusions depend on the "
                "identification assumptions of the model."
            ),

            (
                "Forecasting and causal inference answer "
                "different questions and should not be "
                "treated as interchangeable."
            ),
        ]

        caveats.extend(
            warnings
        )

        return DecisionRecommendation(
            recommendation=
                recommendation,

            rationale=
                rationale,

            causal_evidence=
                causal_evidence,

            forecast_evidence=
                forecast_evidence,

            risk_assessment=
                risk_assessment,

            audit_assessment=
                audit_assessment,

            confidence=
                confidence,

            human_approval_required=
                True,

            caveats=
                caveats,
        )

    # =========================================================
    # OPENAI
    # =========================================================

    def _openai_generate(
        self,
        question,
        business_analysis,
        causal_analysis,
        forecast,
        optimization,
        risk,
        audit,
    ):

        evidence = EvidenceBuilder.build(
            business_analysis=
                business_analysis,

            causal_analysis=
                causal_analysis,

            forecast=
                forecast,

            optimization=
                optimization,

            risk=
                risk,

            audit=
                audit,
        )

        prompt = USER_PROMPT_TEMPLATE.format(
            question=question,

            business_analysis=
                business_analysis,

            causal_analysis=
                causal_analysis,

            forecast=
                forecast,

            optimization=
                optimization,

            risk=
                risk,

            audit=
                audit,
        )

        response = (
            self.client.responses.parse(
                model=self.model,

                instructions=
                    SYSTEM_PROMPT,

                input=prompt,

                text_format=
                    DecisionRecommendation,
            )
        )

        parsed = response.output_parsed

        if parsed is None:

            raise RuntimeError(
                "LLM returned no structured decision."
            )

        return parsed

    # =========================================================
    # PUBLIC API
    # =========================================================

    def generate(
        self,
        question,
        business_analysis,
        causal_analysis,
        forecast,
        optimization,
        risk,
        audit,
    ):

        if self.mode == "openai":

            try:

                return self._openai_generate(
                    question,
                    business_analysis,
                    causal_analysis,
                    forecast,
                    optimization,
                    risk,
                    audit,
                )

            except Exception as error:

                print(
                    "\n[CAUSAL-X] OpenAI API unavailable."
                )

                print(
                    f"[CAUSAL-X] Falling back to mock "
                    f"copilot: {type(error).__name__}"
                )

        return self._mock_generate(
            question,
            business_analysis,
            causal_analysis,
            forecast,
            optimization,
            risk,
            audit,
        )


if __name__ == "__main__":

    copilot = DecisionCopilot()

    result = copilot.generate(

        question=(
            "What should we do with Product A?"
        ),

        business_analysis={
            "product": "A",
            "average_price": 10000,
            "average_demand": 1500,
            "average_profit": 60_000_000,
        },

        causal_analysis={
            "estimated_elasticity": -1.4,
            "demand_change_pct": 7.2,
            "r_squared": 0.78,
        },

        forecast={
            "MAE": 800,
            "recent_actual_demand": 1500,
            "recent_predicted_demand": 1540,
        },

        optimization={
            "price": 9600,
            "price_change": -0.04,
            "marketing_spend": 350000,
            "expected_demand": 1620,
            "expected_profit": 65_000_000,
            "cvar_95": 8_000_000,
        },

        risk={
            "risk_level": "LOW",
            "probability_of_loss": 0.03,
        },

        audit={
            "audit_score": 91,
            "confidence": "HIGH",
            "approved": True,
            "warnings": [],
        },
    )

    print("=" * 70)
    print("CAUSAL-X DECISION COPILOT")
    print("=" * 70)

    print(
        f"\nMode: "
        f"{copilot.mode.upper()}"
    )

    print(
        f"\nRecommendation:\n"
        f"{result.recommendation}"
    )

    print("\nRationale:")

    for reason in result.rationale:

        print(
            f"  • {reason}"
        )

    print(
        f"\nCausal evidence:\n"
        f"{result.causal_evidence}"
    )

    print(
        f"\nForecast evidence:\n"
        f"{result.forecast_evidence}"
    )

    print(
        f"\nRisk assessment:\n"
        f"{result.risk_assessment}"
    )

    print(
        f"\nAudit assessment:\n"
        f"{result.audit_assessment}"
    )

    print(
        f"\nConfidence: "
        f"{result.confidence}"
    )

    print(
        f"Human approval required: "
        f"{result.human_approval_required}"
    )

    print("\nCaveats:")

    for caveat in result.caveats:

        print(
            f"  • {caveat}"
        )
