import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class DemandForecaster:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        self.features = [
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_7",
            "rolling_14",
            "rolling_30",
            "sin_week",
            "cos_week",
            "sin_year",
            "cos_year",
        ]

        self.fitted = False

    # =========================================================
    # CREATE TIME SERIES FEATURES
    # =========================================================

    def create_features(
        self,
        df,
        product,
        region
    ):

        data = df[
            (df["product"] == product)
            & (df["region"] == region)
        ].copy()

        data = data.sort_values(
            "day"
        ).reset_index(drop=True)

        data["lag_1"] = (
            data["demand"].shift(1)
        )

        data["lag_7"] = (
            data["demand"].shift(7)
        )

        data["lag_14"] = (
            data["demand"].shift(14)
        )

        data["rolling_7"] = (
            data["demand"]
            .rolling(7)
            .mean()
        )

        data["rolling_14"] = (
            data["demand"]
            .rolling(14)
            .mean()
        )

        data["rolling_30"] = (
            data["demand"]
            .rolling(30)
            .mean()
        )

        data["sin_week"] = np.sin(
            2 * np.pi * data["day"] / 7
        )

        data["cos_week"] = np.cos(
            2 * np.pi * data["day"] / 7
        )

        data["sin_year"] = np.sin(
            2 * np.pi * data["day"] / 365
        )

        data["cos_year"] = np.cos(
            2 * np.pi * data["day"] / 365
        )

        data = data.dropna()

        return data

    # =========================================================
    # TRAIN
    # =========================================================

    def fit(
        self,
        df,
        product,
        region
    ):

        data = self.create_features(
            df,
            product,
            region
        )

        X = data[
            self.features
        ]

        y = data[
            "demand"
        ]

        self.model.fit(
            X,
            y
        )

        self.fitted = True

        return self

    # =========================================================
    # BACKTEST
    # =========================================================

    def backtest(
        self,
        df,
        product,
        region,
        test_fraction=0.20
    ):

        data = self.create_features(
            df,
            product,
            region
        )

        split = int(
            len(data)
            * (1 - test_fraction)
        )

        train = data.iloc[
            :split
        ]

        test = data.iloc[
            split:
        ]

        self.model.fit(
            train[self.features],
            train["demand"]
        )

        predictions = self.model.predict(
            test[self.features]
        )

        mae = mean_absolute_error(
            test["demand"],
            predictions
        )

        return {
            "MAE": mae,
            "actual": test["demand"].values,
            "predicted": predictions,
        }


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/manufacturing_regional.csv"
    )

    forecaster = DemandForecaster()

    result = forecaster.backtest(
        df,
        product="A",
        region="North"
    )

    print("=" * 65)
    print("CAUSAL-X DEMAND FORECASTER")
    print("=" * 65)

    print(
        f"Product: A"
    )

    print(
        f"Region: North"
    )

    print(
        f"Backtest MAE: "
        f"{result['MAE']:,.2f}"
    )