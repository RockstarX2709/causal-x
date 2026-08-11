import pandas as pd


class ProfitModel:

    def calculate(
        self,
        revenue,
        production_cost,
        marketing_cost,
        inventory_cost,
        logistics_cost,
        maintenance_cost
    ):

        return (
            revenue
            - production_cost
            - marketing_cost
            - inventory_cost
            - logistics_cost
            - maintenance_cost
        )

    def calculate_dataframe(
        self,
        df
    ):

        result = df.copy()

        result["calculated_profit"] = (
            result["revenue"]
            - result["production_cost"]
            - result["marketing_spend"] / 3
            - result["inventory_cost"]
            - result["logistics_cost"]
            - result["maintenance_cost"] / 3
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
            "total_profit":
                result[
                    "calculated_profit"
                ].sum(),

            "average_profit":
                result[
                    "calculated_profit"
                ].mean(),

            "maximum_profit":
                result[
                    "calculated_profit"
                ].max(),

            "minimum_profit":
                result[
                    "calculated_profit"
                ].min(),
        }


if __name__ == "__main__":

    df = pd.read_csv(
        "data/raw/manufacturing_regional.csv"
    )

    model = ProfitModel()

    summary = model.summarize(
        df
    )

    print("=" * 65)
    print("CAUSAL-X PROFIT MODEL")
    print("=" * 65)

    for key, value in summary.items():

        print(
            f"{key}: ₹{value:,.2f}"
        )