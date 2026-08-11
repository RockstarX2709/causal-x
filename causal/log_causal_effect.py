import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class LogCausalEffectEstimator:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_product_data(self, product):

        df = self.df[
            self.df["product"] == product
        ].copy()

        if df.empty:
            raise ValueError(
                f"No data found for product {product}"
            )

        df = df.sort_values(
            "day"
        ).reset_index(drop=True)

        # -----------------------------------------------------
        # LOG TRANSFORMATIONS
        # -----------------------------------------------------

        df["log_demand"] = np.log(
            df["demand"].clip(lower=1)
        )

        df["log_price"] = np.log(
            df["price"].clip(lower=1)
        )

        df["log_competitor_price"] = np.log(
            df["competitor_price"].clip(lower=1)
        )

        df["log_marketing"] = np.log1p(
            df["marketing_spend"]
        )

        # -----------------------------------------------------
        # SEASONAL CONTROLS
        # -----------------------------------------------------

        df["sin_week"] = np.sin(
            2 * np.pi * df["day"] / 7
        )

        df["cos_week"] = np.cos(
            2 * np.pi * df["day"] / 7
        )

        df["sin_year"] = np.sin(
            2 * np.pi * df["day"] / 365
        )

        df["cos_year"] = np.cos(
            2 * np.pi * df["day"] / 365
        )

        # -----------------------------------------------------
        # REGION CONTROLS
        # -----------------------------------------------------

        region_dummies = pd.get_dummies(
            df["region"],
            prefix="region",
            drop_first=True,
            dtype=float
        )

        df = pd.concat(
            [df, region_dummies],
            axis=1
        )

        return df

    # =========================================================
    # FIT LOG-LOG MODEL
    # =========================================================

    def estimate_elasticity(self, product):

        df = self.prepare_product_data(
            product
        )

        features = [
            "log_price",
            "log_competitor_price",
            "log_marketing",
            "sin_week",
            "cos_week",
            "sin_year",
            "cos_year",
        ]

        region_features = [
            col
            for col in df.columns
            if col.startswith("region_")
        ]

        features.extend(
            region_features
        )

        X = df[features]

        y = df["log_demand"]

        model = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "regressor",
                    LinearRegression()
                ),
            ]
        )

        model.fit(X, y)

        regressor = model.named_steps[
            "regressor"
        ]

        scaler = model.named_steps[
            "scaler"
        ]

        price_index = features.index(
            "log_price"
        )

        # Convert standardized coefficient
        # back to original log-log coefficient.

        elasticity = (
            regressor.coef_[price_index]
            / scaler.scale_[price_index]
        )

        r_squared = model.score(
            X,
            y
        )

        return {
            "product": product,
            "elasticity": elasticity,
            "r_squared": r_squared,
            "model": model,
            "features": features,
            "data": df,
        }

    # =========================================================
    # COUNTERFACTUAL INTERVENTION
    # =========================================================

    def simulate_price_intervention(
        self,
        product,
        price_change=-0.05
    ):

        result = self.estimate_elasticity(
            product
        )

        df = result["data"]

        elasticity = result[
            "elasticity"
        ]

        baseline_demand = (
            df["demand"].mean()
        )

        # Structural counterfactual:
        #
        # D_cf / D =
        # (P_cf / P)^elasticity

        demand_multiplier = (
            (1 + price_change)
            ** elasticity
        )

        counterfactual_demand = (
            baseline_demand
            * demand_multiplier
        )

        demand_change_pct = (
            demand_multiplier - 1
        ) * 100

        return {
            "product": product,
            "price_change_pct":
                price_change * 100,

            "estimated_elasticity":
                elasticity,

            "baseline_demand":
                baseline_demand,

            "counterfactual_demand":
                counterfactual_demand,

            "demand_change_pct":
                demand_change_pct,

            "r_squared":
                result["r_squared"],
        }


# =============================================================
# EXPERIMENT
# =============================================================

if __name__ == "__main__":

    estimator = LogCausalEffectEstimator(
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 70)
    print("CAUSAL-X LOG-LOG CAUSAL ESTIMATION")
    print("=" * 70)

    true_elasticities = {
        "A": -1.40,
        "B": -0.80,
        "C": -1.70,
    }

    for product in ["A", "B", "C"]:

        result = (
            estimator.simulate_price_intervention(
                product=product,
                price_change=-0.05
            )
        )

        estimated = result[
            "estimated_elasticity"
        ]

        true_value = (
            true_elasticities[product]
        )

        absolute_error = abs(
            estimated - true_value
        )

        relative_error = (
            absolute_error
            / abs(true_value)
        ) * 100

        print("\n" + "-" * 70)

        print(
            f"Product: {product}"
        )

        print(
            f"True elasticity: "
            f"{true_value:.3f}"
        )

        print(
            f"Estimated elasticity: "
            f"{estimated:.3f}"
        )

        print(
            f"Absolute error: "
            f"{absolute_error:.3f}"
        )

        print(
            f"Relative error: "
            f"{relative_error:.2f}%"
        )

        print(
            f"Model R²: "
            f"{result['r_squared']:.4f}"
        )

        print(
            f"5% price reduction → "
            f"demand change: "
            f"{result['demand_change_pct']:.2f}%"
        )

    print("\n" + "=" * 70)