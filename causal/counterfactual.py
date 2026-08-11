import pandas as pd
import numpy as np


class CounterfactualEngine:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

        self.elasticities = {
            "A": -1.40,
            "B": -0.80,
            "C": -1.70,
        }

        self.production_costs = {
            "A": 5200,
            "B": 13500,
            "C": 2600,
        }

    # ---------------------------------------------------------
    # COUNTERFACTUAL DEMAND
    # ---------------------------------------------------------

    def counterfactual_demand(
        self,
        row,
        new_price
    ):

        product = row["product"]

        old_price = row["price"]

        elasticity = self.elasticities[
            product
        ]

        # Structural price intervention:
        #
        # D_cf / D_observed
        # =
        # (P_cf / P_observed)^elasticity

        price_ratio = (
            new_price / old_price
        )

        demand_multiplier = (
            price_ratio ** elasticity
        )

        cf_demand = (
            row["demand"]
            * demand_multiplier
        )

        return max(
            0,
            cf_demand
        )

    # ---------------------------------------------------------
    # COUNTERFACTUAL ECONOMICS
    # ---------------------------------------------------------

    def simulate_price_change(
        self,
        product,
        price_change_pct
    ):

        subset = self.df[
            self.df["product"] == product
        ].copy()

        if subset.empty:
            raise ValueError(
                f"Product {product} not found."
            )

        results = []

        for _, row in subset.iterrows():

            old_price = row["price"]

            new_price = (
                old_price
                * (1 + price_change_pct)
            )

            # -------------------------------
            # COUNTERFACTUAL DEMAND
            # -------------------------------

            cf_demand = (
                self.counterfactual_demand(
                    row,
                    new_price
                )
            )

            # -------------------------------
            # PRODUCTION CAPACITY
            # -------------------------------

            production_capacity = (
                row["production"]
                / max(
                    row["avg_machine_health"],
                    0.01
                )
            )

            # Assume production can respond
            # to additional demand up to capacity.

            cf_production = min(
                max(
                    row["production"],
                    cf_demand
                ),
                production_capacity
            )

            # -------------------------------
            # INVENTORY
            # -------------------------------

            cf_sales = min(
                cf_demand,
                row["opening_inventory"]
                + cf_production
            )

            cf_inventory = (
                row["opening_inventory"]
                + cf_production
                - cf_sales
            )

            # -------------------------------
            # REVENUE
            # -------------------------------

            cf_revenue = (
                cf_sales
                * new_price
            )

            # -------------------------------
            # COSTS
            # -------------------------------

            cf_production_cost = (
                cf_production
                * self.production_costs[
                    product
                ]
            )

            cf_inventory_cost = (
                cf_inventory * 12
            )

            cf_logistics_cost = (
                cf_sales * 150
            )

            # Keep marketing and maintenance
            # constant for a pure price intervention.

            cf_marketing_cost = (
                row["marketing_spend"]
                / 3
            )

            cf_maintenance_cost = (
                row["maintenance_cost"]
                / 3
            )

            # -------------------------------
            # PROFIT
            # -------------------------------

            cf_profit = (
                cf_revenue
                - cf_production_cost
                - cf_marketing_cost
                - cf_inventory_cost
                - cf_logistics_cost
                - cf_maintenance_cost
            )

            profit_change = (
                cf_profit
                - row["profit"]
            )

            results.append(
                {
                    "date": row["date"],
                    "product": product,
                    "region": row["region"],

                    "baseline_price":
                        old_price,

                    "counterfactual_price":
                        new_price,

                    "baseline_demand":
                        row["demand"],

                    "counterfactual_demand":
                        cf_demand,

                    "baseline_sales":
                        row["sales"],

                    "counterfactual_sales":
                        cf_sales,

                    "baseline_revenue":
                        row["revenue"],

                    "counterfactual_revenue":
                        cf_revenue,

                    "baseline_profit":
                        row["profit"],

                    "counterfactual_profit":
                        cf_profit,

                    "profit_change":
                        profit_change,
                }
            )

        return pd.DataFrame(results)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    def summarize(
        self,
        product,
        price_change_pct
    ):

        result = self.simulate_price_change(
            product,
            price_change_pct
        )

        summary = {

            "product": product,

            "price_change_pct":
                price_change_pct * 100,

            "baseline_demand":
                result[
                    "baseline_demand"
                ].sum(),

            "counterfactual_demand":
                result[
                    "counterfactual_demand"
                ].sum(),

            "demand_change_pct":
                (
                    result[
                        "counterfactual_demand"
                    ].sum()
                    /
                    result[
                        "baseline_demand"
                    ].sum()
                    - 1
                )
                * 100,

            "baseline_revenue":
                result[
                    "baseline_revenue"
                ].sum(),

            "counterfactual_revenue":
                result[
                    "counterfactual_revenue"
                ].sum(),

            "baseline_profit":
                result[
                    "baseline_profit"
                ].sum(),

            "counterfactual_profit":
                result[
                    "counterfactual_profit"
                ].sum(),

            "profit_change":
                result[
                    "profit_change"
                ].sum(),
        }

        summary[
            "profit_change_pct"
        ] = (
            summary[
                "profit_change"
            ]
            /
            summary[
                "baseline_profit"
            ]
        ) * 100

        return summary


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    engine = CounterfactualEngine(
        "data/raw/manufacturing_regional.csv"
    )

    print("=" * 60)
    print("CAUSAL-X COUNTERFACTUAL ENGINE")
    print("=" * 60)

    # Test a 5% price reduction for Product A

    summary = engine.summarize(
        product="A",
        price_change_pct=-0.05
    )

    print("\nScenario:")
    print(
        "Product A price reduced by 5%"
    )

    print("\nResults:")

    for key, value in summary.items():

        if isinstance(value, float):

            if "pct" in key:
                print(
                    f"{key}: {value:.2f}%"
                )

            else:
                print(
                    f"{key}: ₹{value:,.2f}"
                )

        else:
            print(
                f"{key}: {value}"
            )