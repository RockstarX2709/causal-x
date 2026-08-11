from agents.orchestrator import (
    CausalXOrchestrator
)

from agents.llm.copilot import (
    DecisionCopilot
)


def main():

    data_path = (
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 75)
    print("CAUSAL-X END-TO-END DECISION COPILOT")
    print("=" * 75)

    # =========================================================
    # RUN QUANTITATIVE PIPELINE
    # =========================================================

    print("\n[1/2] Running quantitative engines...")

    orchestrator = CausalXOrchestrator(
        data_path
    )

    evidence = orchestrator.solve(
        product="A",
        region="North"
    )

    print("Quantitative pipeline complete.")

    # =========================================================
    # RUN COPILOT
    # =========================================================

    print("\n[2/2] Generating decision explanation...")

    copilot = DecisionCopilot()

    recommendation = copilot.generate(

        question=(
            "What should we do with "
            "Product A in the North region?"
        ),

        business_analysis=
            evidence[
                "business_analysis"
            ],

        causal_analysis=
            evidence[
                "causal_analysis"
            ],

        forecast=
            evidence[
                "forecast"
            ],

        optimization=
            evidence[
                "optimization"
            ],

        risk=
            evidence[
                "risk"
            ],

        audit=
            evidence[
                "audit"
            ],
    )

    # =========================================================
    # DISPLAY
    # =========================================================

    print("\n")
    print("=" * 75)
    print("CAUSAL-X EXECUTIVE DECISION")
    print("=" * 75)

    print("\nRECOMMENDATION")
    print("-" * 75)

    print(
        recommendation.recommendation
    )

    print("\nWHY?")

    for reason in (
        recommendation.rationale
    ):

        print(
            f"  • {reason}"
        )

    print("\nCAUSAL EVIDENCE")

    print(
        recommendation.causal_evidence
    )

    print("\nFORECAST EVIDENCE")

    print(
        recommendation.forecast_evidence
    )

    print("\nRISK ASSESSMENT")

    print(
        recommendation.risk_assessment
    )

    print("\nAUDIT ASSESSMENT")

    print(
        recommendation.audit_assessment
    )

    print(
        f"\nCONFIDENCE: "
        f"{recommendation.confidence}"
    )

    print(
        f"HUMAN APPROVAL REQUIRED: "
        f"{recommendation.human_approval_required}"
    )

    if recommendation.caveats:

        print("\nCAVEATS")

        for caveat in (
            recommendation.caveats
        ):

            print(
                f"  • {caveat}"
            )

    print("\n" + "=" * 75)
    print(
        "CAUSAL-X END-TO-END PIPELINE COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()
