import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression


class CausalRefuter:

    def __init__(self, data_path):

        self.df = pd.read_csv(
            data_path
        )

    # =========================================================
    # PREPARE
    # =========================================================

    def prepare_data(self, product):

        df = self.df[
            self.df["product"] == product
        ].copy()

        df = df.sort_values(
            "day"
        ).reset_index(drop=True)

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
    # BASE ESTIMATE
    # =========================================================

    def estimate(
        self,
        df
    ):

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

        features = [
            "log_price"
        ] + controls

        X = df[features]

        y = df["log_demand"]

        model = LinearRegression()

        model.fit(
            X,
            y
        )

        coefficient = model.coef_[0]

        return coefficient

    # =========================================================
    # PLACEBO TEST
    # =========================================================

    def placebo_test(
        self,
        product,
        seed=42
    ):

        df = self.prepare_data(
            product
        )

        rng = np.random.default_rng(
            seed
        )

        placebo = df[
            "log_price"
        ].sample(
            frac=1,
            random_state=seed
        ).reset_index(
            drop=True
        )

        df = df.reset_index(
            drop=True
        )

        df["placebo_price"] = placebo

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

        X = df[
            ["placebo_price"]
            + controls
        ]

        y = df[
            "log_demand"
        ]

        model = LinearRegression()

        model.fit(
            X,
            y
        )

        placebo_effect = (
            model.coef_[0]
        )

        return placebo_effect

    # =========================================================
    # RANDOM COMMON CAUSE TEST
    # =========================================================

    def random_common_cause_test(
        self,
        product,
        seed=42
    ):

        df = self.prepare_data(
            product
        )

        rng = np.random.default_rng(
            seed
        )

        df["random_cause"] = (
            rng.normal(
                0,
                1,
                len(df)
            )
        )

        controls = [
            "log_competitor_price",
            "log_marketing",
            "sin_week",
            "cos_week",
            "sin_year",
            "cos_year",
            "random_cause",
        ]

        region_features = [
            col
            for col in df.columns
            if col.startswith("region_")
        ]

        controls.extend(
            region_features
        )

        X = df[
            ["log_price"]
            + controls
        ]

        y = df[
            "log_demand"
        ]

        model = LinearRegression()

        model.fit(
            X,
            y
        )

        coefficient = (
            model.coef_[0]
        )

        return coefficient

    # =========================================================
    # SUBSET STABILITY
    # =========================================================

    def subset_stability_test(
        self,
        product,
        repetitions=10
    ):

        df = self.prepare_data(
            product
        )

        estimates = []

        rng = np.random.default_rng(
            42
        )

        for _ in range(
            repetitions
        ):

            sample_indices = rng.choice(
                len(df),
                size=int(
                    len(df) * 0.70
                ),
                replace=False
            )

            sample = df.iloc[
                sample_indices
            ]

            estimates.append(
                self.estimate(
                    sample
                )
            )

        return {
            "mean": np.mean(
                estimates
            ),
            "std": np.std(
                estimates
            ),
            "min": np.min(
                estimates
            ),
            "max": np.max(
                estimates
            ),
        }

    # =========================================================
    # FULL REFUTATION
    # =========================================================

    def run_refutation(
        self,
        product
    ):

        df = self.prepare_data(
            product
        )

        baseline = self.estimate(
            df
        )

        placebo = self.placebo_test(
            product
        )

        random_cause = (
            self.random_common_cause_test(
                product
            )
        )

        stability = (
            self.subset_stability_test(
                product
            )
        )

        return {
            "product": product,
            "baseline_effect": baseline,
            "placebo_effect": placebo,
            "random_common_cause_effect":
                random_cause,
            "subset_stability":
                stability,
        }


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":

    refuter = CausalRefuter(
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 70)
    print(
        "CAUSAL-X REFUTATION TESTS"
    )
    print("=" * 70)

    for product in [
        "A",
        "B",
        "C"
    ]:

        result = (
            refuter.run_refutation(
                product
            )
        )

        print("\n" + "-" * 70)

        print(
            f"Product {product}"
        )

        print(
            f"\nBaseline effect: "
            f"{result['baseline_effect']:.4f}"
        )

        print(
            f"Placebo effect: "
            f"{result['placebo_effect']:.4f}"
        )

        print(
            f"Random common cause: "
            f"{result['random_common_cause_effect']:.4f}"
        )

        stability = (
            result["subset_stability"]
        )

        print(
            f"\nSubset stability:"
        )

        print(
            f"  Mean: "
            f"{stability['mean']:.4f}"
        )

        print(
            f"  Std: "
            f"{stability['std']:.4f}"
        )

        print(
            f"  Min: "
            f"{stability['min']:.4f}"
        )

        print(
            f"  Max: "
            f"{stability['max']:.4f}"
        )

    print("\n" + "=" * 70)