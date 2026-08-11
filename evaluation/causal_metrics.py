import numpy as np


class CausalMetrics:

    TRUE_ELASTICITIES = {
        "A": -1.40,
        "B": -0.80,
        "C": -1.70,
    }

    # =========================================================
    # ABSOLUTE ERROR
    # =========================================================

    @staticmethod
    def absolute_error(
        estimated,
        actual
    ):

        return abs(
            estimated - actual
        )

    # =========================================================
    # RELATIVE ERROR
    # =========================================================

    @staticmethod
    def relative_error(
        estimated,
        actual
    ):

        if actual == 0:
            return np.nan

        return (
            abs(
                estimated - actual
            )
            / abs(actual)
        ) * 100

    # =========================================================
    # BIAS
    # =========================================================

    @staticmethod
    def bias(
        estimates,
        actuals
    ):

        estimates = np.asarray(
            estimates
        )

        actuals = np.asarray(
            actuals
        )

        return np.mean(
            estimates - actuals
        )

    # =========================================================
    # RMSE
    # =========================================================

    @staticmethod
    def rmse(
        estimates,
        actuals
    ):

        estimates = np.asarray(
            estimates
        )

        actuals = np.asarray(
            actuals
        )

        return np.sqrt(
            np.mean(
                (estimates - actuals) ** 2
            )
        )

    # =========================================================
    # EVALUATE PRODUCTS
    # =========================================================

    def evaluate_products(
        self,
        estimates
    ):

        results = []

        for product, estimated in (
            estimates.items()
        ):

            actual = (
                self.TRUE_ELASTICITIES[
                    product
                ]
            )

            results.append(
                {
                    "product": product,

                    "true_elasticity":
                        actual,

                    "estimated_elasticity":
                        estimated,

                    "absolute_error":
                        self.absolute_error(
                            estimated,
                            actual
                        ),

                    "relative_error_pct":
                        self.relative_error(
                            estimated,
                            actual
                        ),
                }
            )

        return results

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(
        self,
        estimates
    ):

        actuals = [
            self.TRUE_ELASTICITIES[
                product
            ]
            for product in estimates
        ]

        estimated = [
            estimates[product]
            for product in estimates
        ]

        return {
            "MAE":
                np.mean(
                    np.abs(
                        np.asarray(estimated)
                        - np.asarray(actuals)
                    )
                ),

            "RMSE":
                self.rmse(
                    estimated,
                    actuals
                ),

            "Bias":
                self.bias(
                    estimated,
                    actuals
                ),
        }


if __name__ == "__main__":

    metrics = CausalMetrics()

    estimates = {
        "A": -1.45,
        "B": -0.83,
        "C": -1.62,
    }

    results = metrics.evaluate_products(
        estimates
    )

    print("=" * 65)
    print("CAUSAL-X CAUSAL METRICS")
    print("=" * 65)

    for result in results:

        print(
            f"\nProduct {result['product']}"
        )

        print(
            f"True: "
            f"{result['true_elasticity']:.3f}"
        )

        print(
            f"Estimated: "
            f"{result['estimated_elasticity']:.3f}"
        )

        print(
            f"Absolute error: "
            f"{result['absolute_error']:.3f}"
        )

        print(
            f"Relative error: "
            f"{result['relative_error_pct']:.2f}%"
        )