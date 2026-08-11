import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


class CausalIdentifier:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

    # =========================================================
    # PREPARE DATA
    # =========================================================

    def prepare_data(self, product):

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
        # LOG VARIABLES
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
        # REGION
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
    # IDENTIFY CAUSAL EFFECT
    # =========================================================

    def identify_price_effect(
        self,
        product
    ):

        df = self.prepare_data(
            product
        )

        controls = [
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

        controls.extend(
            region_features
        )

        treatment = "log_price"

        outcome = "log_demand"

        # -----------------------------------------------------
        # IDENTIFICATION ASSUMPTION
        # -----------------------------------------------------

        assumptions = {
            "consistency": (
                "Observed demand corresponds to "
                "the defined pricing intervention."
            ),

            "exchangeability": (
                "Conditional on competitor price, "
                "marketing, time and region, "
                "price assignment is treated as "
                "independent of unobserved demand shocks."
            ),

            "positivity": (
                "There is sufficient variation "
                "in observed prices."
            ),

            "no_interference": (
                "One region's pricing intervention "
                "does not directly alter another "
                "region's potential demand."
            ),
        }

        # -----------------------------------------------------
        # IDENTIFIED ESTIMAND
        # -----------------------------------------------------

        estimand = {
            "treatment": treatment,
            "outcome": outcome,
            "controls": controls,
            "estimand": (
                "Average causal effect of "
                "log(price) on log(demand)"
            ),
        }

        return {
            "product": product,
            "treatment": treatment,
            "outcome": outcome,
            "controls": controls,
            "assumptions": assumptions,
            "estimand": estimand,
            "data": df,
        }

    # =========================================================
    # DISPLAY IDENTIFICATION
    # =========================================================

    def explain_identification(
        self,
        product
    ):

        result = self.identify_price_effect(
            product
        )

        print("=" * 70)
        print(
            "CAUSAL IDENTIFICATION"
        )
        print("=" * 70)

        print(
            f"\nProduct: {product}"
        )

        print(
            f"\nTreatment:"
            f"\n  {result['treatment']}"
        )

        print(
            f"\nOutcome:"
            f"\n  {result['outcome']}"
        )

        print("\nControls:")

        for control in result[
            "controls"
        ]:

            print(
                f"  • {control}"
            )

        print("\nIdentification assumptions:")

        for name, description in (
            result["assumptions"].items()
        ):

            print(
                f"\n{name.upper()}:"
            )

            print(
                f"  {description}"
            )

        print("\nIdentified estimand:")

        print(
            f"  {result['estimand']['estimand']}"
        )

        return result


if __name__ == "__main__":

    identifier = CausalIdentifier(
        "data/raw/manufacturing_regional.csv"
    )

    identifier.explain_identification(
        "A"
    )