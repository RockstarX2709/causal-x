import pandas as pd
import numpy as np

from causal.log_causal_effect import (
    LogCausalEffectEstimator
)

from models.demand_model import (
    DemandModel
)

from optimization.optimizer import (
    DecisionOptimizer
)

from evaluation.causal_metrics import (
    CausalMetrics
)

from evaluation.prediction_metrics import (
    PredictionMetrics
)


class CausalXBenchmark:

    def __init__(
        self,
        data_path
    ):

        self.data_path = data_path

        self.df = pd.read_csv(
            data_path
        )

    # =========================================================
    # CAUSAL BENCHMARK
    # =========================================================

    def causal_benchmark(self):

        estimator = (
            LogCausalEffectEstimator(
                self.data_path
            )
        )

        estimates = {}

        for product in [
            "A",
            "B",
            "C"
        ]:

            result = (
                estimator.estimate_elasticity(
                    product
                )
            )

            estimates[
                product
            ] = result[
                "elasticity"
            ]

        metrics = CausalMetrics()

        return {
            "estimates":
                estimates,

            "product_results":
                metrics.evaluate_products(
                    estimates
                ),

            "summary":
                metrics.summary(
                    estimates
                ),
        }

    # =========================================================
    # DEMAND BENCHMARK
    # =========================================================

    def demand_benchmark(self):

        # Chronological split
        df = self.df.sort_values(
            "day"
        ).reset_index(
            drop=True
        )

        split = int(
            len(df) * 0.80
        )

        train = df.iloc[
            :split
        ]

        test = df.iloc[
            split:
        ]

        model = DemandModel()

        model.fit(
            train
        )

        predictions = model.predict(
            test
        )

        metrics = PredictionMetrics()

        return {
            "metrics":
                metrics.evaluate(
                    test["demand"].values,
                    predictions
                )
        }

    # =========================================================
    # OPTIMIZATION BENCHMARK
    # =========================================================

    def optimization_benchmark(self):

        # Product A benchmark
        product_data = self.df[
            self.df["product"] == "A"
        ]

        baseline_price = (
            product_data["price"]
            .mean()
        )

        baseline_demand = (
            product_data["demand"]
            .mean()
        )

        baseline_cost = (
            product_data[
                "production_cost"
            ].mean()
            / baseline_demand
        )

        optimizer = DecisionOptimizer(
            risk_aversion=0.25,
            simulations=5000,
            seed=42
        )

        result = optimizer.optimize(
            baseline_price=
                baseline_price,

            baseline_demand=
                baseline_demand,

            baseline_cost=
                baseline_cost,
        )

        return result

    # =========================================================
    # COMPLETE BENCHMARK
    # =========================================================

    def run(self):

        print("=" * 75)
        print("CAUSAL-X SYSTEM BENCHMARK")
        print("=" * 75)

        # -----------------------------------------------------
        # CAUSAL
        # -----------------------------------------------------

        print(
            "\n[1/3] CAUSAL ESTIMATION"
        )

        causal = (
            self.causal_benchmark()
        )

        for result in (
            causal["product_results"]
        ):

            print(
                f"\nProduct "
                f"{result['product']}"
            )

            print(
                f"  True elasticity: "
                f"{result['true_elasticity']:.3f}"
            )

            print(
                f"  Estimated: "
                f"{result['estimated_elasticity']:.3f}"
            )

            print(
                f"  Error: "
                f"{result['relative_error_pct']:.2f}%"
            )

        summary = causal[
            "summary"
        ]

        print(
            f"\nCausal MAE: "
            f"{summary['MAE']:.4f}"
        )

        print(
            f"Causal RMSE: "
            f"{summary['RMSE']:.4f}"
        )

        # -----------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------

        print(
            "\n[2/3] DEMAND PREDICTION"
        )

        prediction = (
            self.demand_benchmark()
        )

        for key, value in (
            prediction[
                "metrics"
            ].items()
        ):

            print(
                f"  {key}: "
                f"{value:.4f}"
            )

        # -----------------------------------------------------
        # OPTIMIZATION
        # -----------------------------------------------------

        print(
            "\n[3/3] DECISION OPTIMIZATION"
        )

        optimization = (
            self.optimization_benchmark()
        )

        best = optimization[
            "best_decision"
        ]

        print(
            f"\n  Recommended price: "
            f"₹{best['price']:,.2f}"
        )

        print(
            f"  Price change: "
            f"{best['price_change'] * 100:.2f}%"
        )

        print(
            f"  Marketing: "
            f"₹{best['marketing_spend']:,.2f}"
        )

        print(
            f"  Expected demand: "
            f"{best['expected_demand']:,.2f}"
        )

        print(
            f"  Expected profit: "
            f"₹{best['expected_profit']:,.2f}"
        )

        print(
            f"  CVaR 95%: "
            f"₹{best['CVaR_95']:,.2f}"
        )

        print(
            f"  Risk-adjusted objective: "
            f"₹{best['objective']:,.2f}"
        )

        print(
            "\n" + "=" * 75
        )

        print(
            "BENCHMARK COMPLETE"
        )

        print(
            "=" * 75
        )

        return {
            "causal":
                causal,

            "prediction":
                prediction,

            "optimization":
                optimization,
        }


if __name__ == "__main__":

    benchmark = CausalXBenchmark(
        "data/raw/manufacturing_regional.csv"
    )

    benchmark.run()