import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DemandModel:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        self.features = None
        self.is_fitted = False

    # =========================================================
    # FEATURE ENGINEERING
    # =========================================================

    def prepare_features(self, df):

        data = df.copy()

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

        # Price relative to product baseline
        data["price_ratio"] = (
            data["price"]
            / data.groupby("product")["price"]
            .transform("mean")
        )

        data["competitor_ratio"] = (
            data["competitor_price"]
            / data.groupby("product")["competitor_price"]
            .transform("mean")
        )

        # Marketing intensity
        data["log_marketing"] = np.log1p(
            data["marketing_spend"]
        )

        # Encode categorical variables
        data = pd.get_dummies(
            data,
            columns=[
                "product",
                "region",
                "supplier"
            ],
            dtype=float
        )

        return data

    # =========================================================
    # TRAIN
    # =========================================================

    def fit(self, df):

        data = self.prepare_features(df)

        target = "demand"

        excluded = [
            "date",
            "demand",
            "sales",
            "revenue",
            "profit",
            "closing_inventory",
            "opening_inventory",
            "production_cost",
            "inventory_cost",
            "logistics_cost",
            "maintenance_cost",
        ]

        features = [
            col
            for col in data.columns
            if col not in excluded
        ]

        X = data[features]
        y = data[target]

        self.features = features

        self.model.fit(
            X,
            y
        )

        self.is_fitted = True

        return self

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, df):

        if not self.is_fitted:
            raise RuntimeError(
                "DemandModel must be fitted first."
            )

        data = self.prepare_features(df)

        # Ensure identical columns
        for feature in self.features:

            if feature not in data.columns:
                data[feature] = 0

        X = data[
            self.features
        ]

        return self.model.predict(X)

    # =========================================================
    # EVALUATE
    # =========================================================

    def evaluate(self, df):

        predictions = self.predict(df)

        actual = df["demand"].values

        mae = mean_absolute_error(
            actual,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predictions
            )
        )

        r2 = r2_score(
            actual,
            predictions
        )

        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/manufacturing_regional.csv"
    )

    # Chronological split
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

    model.fit(train)

    metrics = model.evaluate(
        test
    )

    print("=" * 65)
    print("CAUSAL-X DEMAND MODEL")
    print("=" * 65)

    print(
        f"MAE:  {metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:,.2f}"
    )

    print(
        f"R²:   {metrics['R2']:.4f}"
    )