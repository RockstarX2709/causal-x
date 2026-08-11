import pandas as pd


class RevenueModel:

    def calculate(
        self,
        price,
        sales
    ):

        return price * sales

    def calculate_dataframe(
        self,
        df
    ):

        result = df.copy()

        result["predicted_revenue"] = (
            result["price"]
            * result["sales"]
        )

        return result

    def summarize(
        self,
        df
    ):

        result = self.calculate_dataframe(
            df
        )

        return {
            "total_revenue":
                result[
                    "predicted_revenue"
                ].sum(),

            "average_revenue":
                result[
                    "predicted_revenue"
                ].mean(),

            "max_revenue":
                result[
                    "predicted_revenue"
                ].max(),

            "min_revenue":
                result[
                    "predicted_revenue"
                ].min(),
        }


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/manufacturing_regional.csv"
    )

    model = RevenueModel()

    summary = model.summarize(
        df
    )

    print("=" * 65)
    print("CAUSAL-X REVENUE MODEL")
    print("=" * 65)

    for key, value in summary.items():

        print(
            f"{key}: ₹{value:,.2f}"
        )
        