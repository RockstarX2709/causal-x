import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class PredictionMetrics:

    # =========================================================
    # MAE
    # =========================================================

    @staticmethod
    def mae(
        actual,
        predicted
    ):

        return mean_absolute_error(
            actual,
            predicted
        )

    # =========================================================
    # RMSE
    # =========================================================

    @staticmethod
    def rmse(
        actual,
        predicted
    ):

        return np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        )

    # =========================================================
    # MAPE
    # =========================================================

    @staticmethod
    def mape(
        actual,
        predicted
    ):

        actual = np.asarray(
            actual
        )

        predicted = np.asarray(
            predicted
        )

        mask = actual != 0

        return np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        ) * 100

    # =========================================================
    # R2
    # =========================================================

    @staticmethod
    def r2(
        actual,
        predicted
    ):

        return r2_score(
            actual,
            predicted
        )

    # =========================================================
    # COMPLETE REPORT
    # =========================================================

    def evaluate(
        self,
        actual,
        predicted
    ):

        return {
            "MAE":
                self.mae(
                    actual,
                    predicted
                ),

            "RMSE":
                self.rmse(
                    actual,
                    predicted
                ),

            "MAPE":
                self.mape(
                    actual,
                    predicted
                ),

            "R2":
                self.r2(
                    actual,
                    predicted
                ),
        }


if __name__ == "__main__":

    actual = np.array(
        [100, 120, 130, 150, 160]
    )

    predicted = np.array(
        [102, 117, 128, 147, 163]
    )

    metrics = PredictionMetrics()

    result = metrics.evaluate(
        actual,
        predicted
    )

    print("=" * 65)
    print("CAUSAL-X PREDICTION METRICS")
    print("=" * 65)

    for key, value in result.items():

        print(
            f"{key}: {value:.4f}"
        )