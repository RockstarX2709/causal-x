class AuditorAgent:

    def audit(
        self,
        causal_result,
        forecast_result,
        optimization_result
    ):

        warnings = []

        score = 100

        # =====================================================
        # CAUSAL ROBUSTNESS
        # =====================================================

        robustness = causal_result.get(
            "robustness",
            "UNKNOWN"
        )

        if robustness == "HIGH":

            score += 0

        elif robustness == "MEDIUM":

            warnings.append(
                "Causal estimate has moderate robustness."
            )

            score -= 15

        elif robustness == "LOW":

            warnings.append(
                "Causal estimate has low robustness."
            )

            score -= 35

        else:

            warnings.append(
                "Causal robustness could not be verified."
            )

            score -= 25

        # =====================================================
        # CONFIDENCE INTERVAL
        # =====================================================

        ci_lower = causal_result.get(
            "bootstrap_ci_lower"
        )

        ci_upper = causal_result.get(
            "bootstrap_ci_upper"
        )

        if (
            ci_lower is not None
            and ci_upper is not None
        ):

            if (
                ci_lower < 0
                and ci_upper < 0
            ):

                pass

            else:

                warnings.append(
                    "Causal confidence interval "
                    "includes zero."
                )

                score -= 25

        # =====================================================
        # CAUSAL MODEL FIT
        # =====================================================

        causal_r2 = causal_result.get(
            "r_squared",
            0
        )

        if causal_r2 < 0.30:

            warnings.append(
                "Causal model fit is weak."
            )

            score -= 15

        elif causal_r2 < 0.50:

            warnings.append(
                "Causal model fit is moderate."
            )

            score -= 5

        # =====================================================
        # FORECAST QUALITY
        # =====================================================

        forecast_mae = forecast_result.get(
            "MAE",
            0
        )

        if forecast_mae > 5000:

            warnings.append(
                "Forecast error is high."
            )

            score -= 15

        # =====================================================
        # OPTIMIZATION
        # =====================================================

        expected_profit = (
            optimization_result.get(
                "expected_profit",
                0
            )
        )

        if expected_profit <= 0:

            warnings.append(
                "Optimization produced "
                "non-positive expected profit."
            )

            score -= 30

        # =====================================================
        # FINAL SCORE
        # =====================================================

        score = max(
            0,
            min(100, score)
        )

        if score >= 80:

            confidence = "HIGH"

        elif score >= 60:

            confidence = "MEDIUM"

        else:

            confidence = "LOW"

        return {
            "audit_score":
                score,

            "confidence":
                confidence,

            "warnings":
                warnings,

            "approved":
                score >= 60,
        }


if __name__ == "__main__":

    auditor = AuditorAgent()

    result = auditor.audit(

        causal_result={
            "robustness": "HIGH",
            "bootstrap_ci_lower": -2.7,
            "bootstrap_ci_upper": -2.0,
            "r_squared": 0.44,
        },

        forecast_result={
            "MAE": 1000
        },

        optimization_result={
            "expected_profit":
                100_000_000
        }
    )

    print("=" * 65)
    print("CAUSAL-X AUDITOR AGENT")
    print("=" * 65)

    print(
        f"Audit score: "
        f"{result['audit_score']}/100"
    )

    print(
        f"Confidence: "
        f"{result['confidence']}"
    )

    print(
        f"Approved: "
        f"{result['approved']}"
    )

    if result["warnings"]:

        print("\nWarnings:")

        for warning in result["warnings"]:

            print(
                f"  • {warning}"
            )
