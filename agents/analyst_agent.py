import pandas as pd
import numpy as np


class AnalystAgent:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def analyze_product(self, product):

        df = self.df[
            self.df["product"] == product
        ].copy()

        if df.empty:
            raise ValueError(
                f"Product {product} not found."
            )

        return {
            "product": product,

            "observations":
                len(df),

            "average_price":
                df["price"].mean(),

            "average_demand":
                df["demand"].mean(),

            "average_sales":
                df["sales"].mean(),

            "average_revenue":
                df["revenue"].mean(),

            "average_profit":
                df["profit"].mean(),

            "average_machine_health":
                df[
                    "avg_machine_health"
                ].mean(),

            "stockout_rate":
                df["stockout"].mean(),

            "supplier_delay_rate":
                df[
                    "supplier_delayed"
                ].mean(),

            "average_lead_time":
                df["lead_time"].mean(),
        }

    def identify_risks(self, product):

        df = self.df[
            self.df["product"] == product
        ]

        risks = []

        stockout_rate = (
            df["stockout"].mean()
        )

        delay_rate = (
            df["supplier_delayed"].mean()
        )

        machine_health = (
            df["avg_machine_health"].mean()
        )

        if stockout_rate > 0.10:
            risks.append(
                "High stockout frequency"
            )

        if delay_rate > 0.15:
            risks.append(
                "Supplier reliability risk"
            )

        if machine_health < 0.80:
            risks.append(
                "Low machine health"
            )

        if not risks:
            risks.append(
                "No major operational risk detected"
            )

        return risks


if __name__ == "__main__":

    agent = AnalystAgent(
        "data/raw/manufacturing_regional.csv"
    )

    result = agent.analyze_product("A")

    print("=" * 65)
    print("CAUSAL-X ANALYST AGENT")
    print("=" * 65)

    for key, value in result.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.4f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print("\nRisks:")

    for risk in agent.identify_risks("A"):
        print(f"  • {risk}")