import pandas as pd

from models.forecasting import (
    DemandForecaster
)


class ForecastAgent:

    def __init__(self, data_path):

        self.data_path = data_path

        self.df = pd.read_csv(
            data_path
        )

    def forecast(
        self,
        product,
        region
    ):

        forecaster = DemandForecaster()

        result = forecaster.backtest(
            self.df,
            product,
            region
        )

        return {
            "product":
                product,

            "region":
                region,

            "MAE":
                result["MAE"],

            "recent_actual_demand":
                result[
                    "actual"
                ][-1],

            "recent_predicted_demand":
                result[
                    "predicted"
                ][-1],

            "mean_actual_demand":
                result[
                    "actual"
                ].mean(),

            "mean_predicted_demand":
                result[
                    "predicted"
                ].mean(),
        }


if __name__ == "__main__":

    agent = ForecastAgent(
        "data/raw/manufacturing_regional.csv"
    )

    result = agent.forecast(
        "A",
        "North"
    )

    print("=" * 65)
    print("CAUSAL-X FORECAST AGENT")
    print("=" * 65)

    for key, value in result.items():

        print(
            f"{key}: {value:.4f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )