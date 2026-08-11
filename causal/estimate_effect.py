import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class CausalEffectEstimator:

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

        # Sort chronologically
        df = df.sort_values("day").reset_index(drop=True)

        # -----------------------------------------------------
        # TRANSFORMATIONS
        # -----------------------------------------------------

        df["price_ratio"] = (
            df["price"] / df["price"].mean()
        )

        df["competitor_ratio"] = (
            df["competitor_price"]
            / df["competitor_price"].mean()
        )

        df["marketing_log"] = np.log1p(
            df["marketing_spend"]
        )

        # -----------------------------------------------------
        # TIME CONTROLS
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
        # REGION ENCODING
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
    # ESTIMATE PRICE EFFECT
    # =========================================================

    def estimate_price_effect(self, product):

        df = self.prepare_product_data(product)

        # -----------------------------------------------------
        # FEATURES
        # -----------------------------------------------------

        features = [
            "price",
            "competitor_price",
            "marketing_log",
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

        features.extend(region_features)

        X = df[features]
        y = df["demand"]

        # -----------------------------------------------------
        # REMOVE MISSING VALUES
        # -----------------------------------------------------

        valid = (
            X.notna().all(axis=1)
            & y.notna()
        )

        X = X.loc[valid]
        y = y.loc[valid]

        df = df.loc[valid].copy()

        # -----------------------------------------------------
        # LINEAR MODEL
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # EXTRACT PRICE COEFFICIENT
        # -----------------------------------------------------

        regressor = model.named_steps[
            "regressor"
        ]

        scaler = model.named_steps[
            "scaler"
        ]

        price_index = features.index("price")

        coefficient = (
            regressor.coef_[price_index]
            / scaler.scale_[price_index]
        )

        # -----------------------------------------------------
        # MODEL R²
        # -----------------------------------------------------

        r_squared = model.score(X, y)

        return {
            "product": product,
            "price_coefficient": coefficient,
            "r_squared": r_squared,
            "model": model,
            "features": features,
            "data": df,
        }

    # =========================================================
    # ESTIMATE INTERVENTION
    # =========================================================

    def estimate_intervention(
        self,
        product,
        price_change=-0.05
    ):

        result = self.estimate_price_effect(
            product
        )

        df = result["data"]
        model = result["model"]
        features = result["features"]

        # -----------------------------------------------------
        # BASELINE WORLD
        # -----------------------------------------------------

        baseline = df.copy()

        # -----------------------------------------------------
        # COUNTERFACTUAL WORLD
        # -----------------------------------------------------

        intervention = df.copy()

        intervention["price"] = (
            intervention["price"]
            * (1 + price_change)
        )

        # -----------------------------------------------------
        # PREDICTIONS
        # -----------------------------------------------------

        X_baseline = baseline[features]

        X_intervention = intervention[features]

        baseline_prediction = model.predict(
            X_baseline
        )

        intervention_prediction = model.predict(
            X_intervention
        )

        # -----------------------------------------------------
        # INDIVIDUAL TREATMENT EFFECTS
        # -----------------------------------------------------

        individual_effect = (
            intervention_prediction
            - baseline_prediction
        )

        # Average Treatment Effect
        ate = np.mean(
            individual_effect
        )

        # Average baseline prediction
        baseline_mean = np.mean(
            baseline_prediction
        )

        # Average intervention prediction
        intervention_mean = np.mean(
            intervention_prediction
        )

        # -----------------------------------------------------
        # RELATIVE DEMAND EFFECT
        # -----------------------------------------------------

        relative_effect = (
            ate / baseline_mean
        ) * 100

        # -----------------------------------------------------
        # ELASTICITY
        # -----------------------------------------------------

        estimated_elasticity = (
            (
                intervention_mean
                - baseline_mean
            )
            / baseline_mean
        ) / price_change

        return {
            "product": product,
            "price_change": price_change,

            "baseline_demand":
                baseline_mean,

            "intervention_demand":
                intervention_mean,

            "absolute_effect":
                ate,

            "relative_effect":
                relative_effect,

            "estimated_elasticity":
                estimated_elasticity,

            "price_coefficient":
                result["price_coefficient"],

            "r_squared":
                result["r_squared"],

            "model":
                model,
        }


# =============================================================
# RUN EXPERIMENT
# =============================================================

if __name__ == "__main__":

    estimator = CausalEffectEstimator(
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 65)
    print("CAUSAL-X CAUSAL EFFECT ESTIMATION")
    print("=" * 65)

    print(
        "\nEstimating effect of a 5% price reduction..."
    )

    for product in ["A", "B", "C"]:

        result = (
            estimator.estimate_intervention(
                product=product,
                price_change=-0.05
            )
        )

        print("\n" + "-" * 65)

        print(
            f"Product {product}"
        )

        print(
            f"Price intervention: -5.00%"
        )

        print(
            f"Estimated baseline demand: "
            f"{result['baseline_demand']:,.2f}"
        )

        print(
            f"Estimated intervention demand: "
            f"{result['intervention_demand']:,.2f}"
        )

        print(
            f"Estimated absolute effect: "
            f"{result['absolute_effect']:,.2f}"
        )

        print(
            f"Estimated relative effect: "
            f"{result['relative_effect']:.2f}%"
        )

        print(
            f"Estimated elasticity: "
            f"{result['estimated_elasticity']:.3f}"
        )

        print(
            f"Model R²: "
            f"{result['r_squared']:.4f}"
        )

    print("\n" + "=" * 65)
    print("Experiment complete.")
    print("=" * 65)