import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class CausalRobustness:

    def __init__(self, data_path):

        self.df = pd.read_csv(
            data_path
        )

    # =========================================================
    # PREPARE PRODUCT DATA
    # =========================================================

    def prepare(
        self,
        product
    ):

        df = self.df[
            self.df["product"] == product
        ].copy()

        if len(df) < 30:

            raise ValueError(
                "Insufficient observations."
            )

        # -----------------------------------------------------
        # LOG TRANSFORM
        # -----------------------------------------------------

        df["log_demand"] = np.log(
            np.maximum(
                df["demand"],
                1e-6
            )
        )

        df["log_price"] = np.log(
            np.maximum(
                df["price"],
                1e-6
            )
        )

        df["log_competitor"] = np.log(
            np.maximum(
                df["competitor_price"],
                1e-6
            )
        )

        df["log_marketing"] = np.log1p(
            np.maximum(
                df["marketing_spend"],
                0
            )
        )

        # -----------------------------------------------------
        # TIME FEATURES
        # -----------------------------------------------------

        df["sin_week"] = np.sin(
            2
            * np.pi
            * df["day"]
            / 7
        )

        df["cos_week"] = np.cos(
            2
            * np.pi
            * df["day"]
            / 7
        )

        df["sin_year"] = np.sin(
            2
            * np.pi
            * df["day"]
            / 365
        )

        df["cos_year"] = np.cos(
            2
            * np.pi
            * df["day"]
            / 365
        )

        return df

    # =========================================================
    # FIT MODEL
    # =========================================================

    def fit_model(
        self,
        df,
        include_controls=True
    ):

        features = [
            "log_price"
        ]

        if include_controls:

            features += [
                "log_competitor",
                "log_marketing",
                "sin_week",
                "cos_week",
                "sin_year",
                "cos_year",
            ]

        X = df[
            features
        ]

        y = df[
            "log_demand"
        ]

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

        model.fit(
            X,
            y
        )

        regressor = (
            model.named_steps[
                "regressor"
            ]
        )

        scaler = (
            model.named_steps[
                "scaler"
            ]
        )

        price_index = (
            features.index(
                "log_price"
            )
        )

        elasticity = (
            regressor.coef_[
                price_index
            ]
            / scaler.scale_[
                price_index
            ]
        )

        predictions = (
            model.predict(X)
        )

        residuals = (
            y.values
            - predictions
        )

        r2 = (
            model.score(
                X,
                y
            )
        )

        return {
            "elasticity":
                elasticity,

            "r2":
                r2,

            "residuals":
                residuals,

            "predictions":
                predictions,

            "features":
                features,

            "model":
                model,
        }

    # =========================================================
    # BOOTSTRAP CONFIDENCE INTERVAL
    # =========================================================

    def bootstrap(
        self,
        product,
        n_bootstrap=500,
        seed=42
    ):

        df = self.prepare(
            product
        )

        rng = np.random.default_rng(
            seed
        )

        estimates = []

        n = len(df)

        for _ in range(
            n_bootstrap
        ):

            indices = (
                rng.integers(
                    0,
                    n,
                    n
                )
            )

            sample = df.iloc[
                indices
            ]

            try:

                result = (
                    self.fit_model(
                        sample
                    )
                )

                estimates.append(
                    result[
                        "elasticity"
                    ]
                )

            except Exception:

                continue

        estimates = np.asarray(
            estimates
        )

        if len(estimates) < 50:

            raise RuntimeError(
                "Bootstrap failed."
            )

        lower = np.percentile(
            estimates,
            2.5
        )

        upper = np.percentile(
            estimates,
            97.5
        )

        return {
            "mean":
                estimates.mean(),

            "std":
                estimates.std(),

            "lower":
                lower,

            "upper":
                upper,

            "samples":
                len(estimates),
        }

    # =========================================================
    # PLACEBO TEST
    # =========================================================

    def placebo_test(
        self,
        product,
        seed=42
    ):

        df = self.prepare(
            product
        )

        rng = np.random.default_rng(
            seed
        )

        shuffled = df[
            "log_price"
        ].values.copy()

        rng.shuffle(
            shuffled
        )

        placebo = df.copy()

        placebo[
            "log_price"
        ] = shuffled

        result = (
            self.fit_model(
                placebo
            )
        )

        return {
            "placebo_elasticity":
                result[
                    "elasticity"
                ],

            "actual_direction":
                "negative",

            "placebo_direction":
                (
                    "negative"
                    if result[
                        "elasticity"
                    ] < 0
                    else "positive"
                ),
        }

    # =========================================================
    # CONTROL SENSITIVITY
    # =========================================================

    def sensitivity(
        self,
        product
    ):

        df = self.prepare(
            product
        )

        full = self.fit_model(
            df,
            include_controls=True
        )

        simple = self.fit_model(
            df,
            include_controls=False
        )

        return {
            "with_controls":
                full[
                    "elasticity"
                ],

            "without_controls":
                simple[
                    "elasticity"
                ],

            "difference":
                (
                    full[
                        "elasticity"
                    ]
                    -
                    simple[
                        "elasticity"
                    ]
                ),
        }

    # =========================================================
    # COMPLETE ROBUSTNESS REPORT
    # =========================================================

    def evaluate(
        self,
        product
    ):

        baseline = self.fit_model(
            self.prepare(product)
        )

        bootstrap = self.bootstrap(
            product
        )

        placebo = self.placebo_test(
            product
        )

        sensitivity = self.sensitivity(
            product
        )

        ci_lower = (
            bootstrap["lower"]
        )

        ci_upper = (
            bootstrap["upper"]
        )

        # -----------------------------------------------------
        # ROBUSTNESS RULES
        # -----------------------------------------------------

        confidence_interval_excludes_zero = (
            ci_lower < 0
            and ci_upper < 0
        )

        placebo_is_weaker = (
            abs(
                placebo[
                    "placebo_elasticity"
                ]
            )
            <
            abs(
                baseline[
                    "elasticity"
                ]
            )
        )

        sensitivity_difference = abs(
            sensitivity[
                "difference"
            ]
        )

        if (
            confidence_interval_excludes_zero
            and placebo_is_weaker
            and sensitivity_difference < 0.5
        ):

            robustness = "HIGH"

        elif confidence_interval_excludes_zero:

            robustness = "MEDIUM"

        else:

            robustness = "LOW"

        return {
            "product":
                product,

            "elasticity":
                baseline[
                    "elasticity"
                ],

            "r2":
                baseline["r2"],

            "bootstrap_mean":
                bootstrap["mean"],

            "ci_lower":
                ci_lower,

            "ci_upper":
                ci_upper,

            "bootstrap_std":
                bootstrap["std"],

            "placebo_elasticity":
                placebo[
                    "placebo_elasticity"
                ],

            "sensitivity_with_controls":
                sensitivity[
                    "with_controls"
                ],

            "sensitivity_without_controls":
                sensitivity[
                    "without_controls"
                ],

            "sensitivity_difference":
                sensitivity[
                    "difference"
                ],

            "robustness":
                robustness,
        }


if __name__ == "__main__":

    evaluator = CausalRobustness(
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 75)
    print("CAUSAL-X CAUSAL ROBUSTNESS ANALYSIS")
    print("=" * 75)

    for product in [
        "A",
        "B",
        "C"
    ]:

        result = evaluator.evaluate(
            product
        )

        print(
            f"\nPRODUCT {product}"
        )

        print(
            f"Elasticity: "
            f"{result['elasticity']:.4f}"
        )

        print(
            f"R²: "
            f"{result['r2']:.4f}"
        )

        print(
            f"Bootstrap 95% CI: "
            f"["
            f"{result['ci_lower']:.4f}, "
            f"{result['ci_upper']:.4f}"
            f"]"
        )

        print(
            f"Placebo elasticity: "
            f"{result['placebo_elasticity']:.4f}"
        )

        print(
            f"With controls: "
            f"{result['sensitivity_with_controls']:.4f}"
        )

        print(
            f"Without controls: "
            f"{result['sensitivity_without_controls']:.4f}"
        )

        print(
            f"Robustness: "
            f"{result['robustness']}"
        )

    print(
        "\n" + "=" * 75
    )
