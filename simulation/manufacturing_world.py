import numpy as np
import pandas as pd
from pathlib import Path


class ManufacturingWorld:

    def __init__(self, days=730, seed=42):

        self.days = days
        self.rng = np.random.default_rng(seed)

        self.products = {
            "A": {
                "base_demand": 1400,
                "price": 10000,
                "elasticity": -1.40,
                "production_cost": 5200,
            },
            "B": {
                "base_demand": 850,
                "price": 25000,
                "elasticity": -0.80,
                "production_cost": 13500,
            },
            "C": {
                "base_demand": 2300,
                "price": 5000,
                "elasticity": -1.70,
                "production_cost": 2600,
            },
        }

        self.regions = {
            "North": 1.00,
            "South": 1.12,
            "East": 0.88,
        }

        self.suppliers = {
            "S1": {
                "cost": 500,
                "lead_time": 5,
                "reliability": 0.96,
            },
            "S2": {
                "cost": 470,
                "lead_time": 8,
                "reliability": 0.89,
            },
            "S3": {
                "cost": 540,
                "lead_time": 3,
                "reliability": 0.99,
            },
        }

        self.machines = {
            "M1": {"capacity": 4500, "health": 0.95},
            "M2": {"capacity": 5000, "health": 0.92},
            "M3": {"capacity": 4000, "health": 0.97},
            "M4": {"capacity": 5500, "health": 0.90},
            "M5": {"capacity": 3500, "health": 0.94},
        }

    # ---------------------------------------------------------
    # SEASONALITY
    # ---------------------------------------------------------

    def seasonality(self, day):

        weekly = 1 + 0.05 * np.sin(
            2 * np.pi * day / 7
        )

        yearly = 1 + 0.15 * np.sin(
            2 * np.pi * day / 365
        )

        return weekly * yearly

    # ---------------------------------------------------------
    # COMPETITOR PRICE
    # ---------------------------------------------------------

    def competitor_price(self, product, day):

        base = self.products[product]["price"]

        trend = 1 + 0.0001 * day

        noise = self.rng.normal(0, 0.02)

        return base * trend * (1 + noise)

    # ---------------------------------------------------------
    # MARKETING
    # ---------------------------------------------------------

    def marketing_spend(self, product):

        return self.rng.uniform(
            100_000,
            500_000
        )

    def marketing_effect(self, spend):

        return 1 + 0.035 * np.log1p(
            spend / 100_000
        )

    # ---------------------------------------------------------
    # OUR PRICE
    # ---------------------------------------------------------

    def our_price(self, product, day):

        base = self.products[product]["price"]

        promotion = self.rng.choice(
            [1.00, 0.97, 0.95],
            p=[0.80, 0.15, 0.05],
        )

        return base * promotion

    # ---------------------------------------------------------
    # DEMAND
    # ---------------------------------------------------------

    def generate_demand(
        self,
        product,
        region,
        price,
        competitor_price,
        marketing,
        day,
    ):

        config = self.products[product]

        base = config["base_demand"]

        seasonal = self.seasonality(day)

        regional = self.regions[region]

        # TRUE CAUSAL PRICE EFFECT
        relative_price = (
            price / config["price"]
        )

        price_effect = (
            relative_price
            ** config["elasticity"]
        )

        # TRUE CAUSAL COMPETITOR EFFECT
        competitor_effect = (
            1
            + 0.80
            * (
                (competitor_price - price)
                / price
            )
        )

        # TRUE CAUSAL MARKETING EFFECT
        marketing_effect = (
            self.marketing_effect(
                marketing
            )
        )

        noise = self.rng.normal(
            1.0,
            0.08
        )

        demand = (
            base
            * seasonal
            * regional
            * price_effect
            * competitor_effect
            * marketing_effect
            * noise
        )

        return max(0, demand)

    # ---------------------------------------------------------
    # MACHINE HEALTH
    # ---------------------------------------------------------

    def update_machine_health(self, health):

        degradation = self.rng.uniform(
            0.0001,
            0.001
        )

        noise = self.rng.normal(
            0,
            0.0002
        )

        new_health = (
            health
            - degradation
            + noise
        )

        return np.clip(
            new_health,
            0.50,
            1.00
        )

    # ---------------------------------------------------------
    # SUPPLIER EVENT
    # ---------------------------------------------------------

    def supplier_event(self, supplier):

        reliability = self.suppliers[
            supplier
        ]["reliability"]

        return (
            self.rng.random()
            > reliability
        )

    # ---------------------------------------------------------
    # GENERATE WORLD
    # ---------------------------------------------------------

    def generate(self):

        rows = []

        machine_health = {
            machine: info["health"]
            for machine, info
            in self.machines.items()
        }

        # Inventory is maintained per product + region
        inventory = {}

        for product in self.products:

            for region in self.regions:

                inventory[
                    (product, region)
                ] = (
                    self.products[
                        product
                    ]["base_demand"]
                    * self.regions[region]
                    * 2
                )

        for day in range(self.days):

            date = (
                pd.Timestamp("2024-01-01")
                + pd.Timedelta(days=day)
            )

            # -------------------------------------------------
            # UPDATE MACHINE HEALTH
            # -------------------------------------------------

            for machine in machine_health:

                machine_health[machine] = (
                    self.update_machine_health(
                        machine_health[machine]
                    )
                )

            # -------------------------------------------------
            # SUPPLIER
            # -------------------------------------------------

            supplier = self.rng.choice(
                list(self.suppliers.keys())
            )

            supplier_info = self.suppliers[
                supplier
            ]

            supplier_delayed = (
                self.supplier_event(
                    supplier
                )
            )

            lead_time = (
                supplier_info["lead_time"]
            )

            if supplier_delayed:

                lead_time += self.rng.integers(
                    3,
                    10
                )

            # -------------------------------------------------
            # PRODUCT + REGION
            # -------------------------------------------------

            for product in self.products:

                price = self.our_price(
                    product,
                    day
                )

                competitor_price = (
                    self.competitor_price(
                        product,
                        day
                    )
                )

                marketing = (
                    self.marketing_spend(
                        product
                    )
                )

                for region in self.regions:

                    # -----------------------------------------
                    # DEMAND
                    # -----------------------------------------

                    demand = (
                        self.generate_demand(
                            product=product,
                            region=region,
                            price=price,
                            competitor_price=competitor_price,
                            marketing=marketing,
                            day=day,
                        )
                    )

                    # -----------------------------------------
                    # PRODUCTION CAPACITY
                    # -----------------------------------------

                    available_capacity = sum(
                        info["capacity"]
                        * machine_health[machine]
                        for machine, info
                        in self.machines.items()
                    )

                    # Allocate production capacity
                    regional_capacity = (
                        available_capacity
                        * self.regions[region]
                        / sum(
                            self.regions.values()
                        )
                    )

                    production = min(
                        regional_capacity,
                        demand
                        * self.rng.uniform(
                            0.90,
                            1.10
                        )
                    )

                    # -----------------------------------------
                    # INVENTORY
                    # -----------------------------------------

                    key = (
                        product,
                        region
                    )

                    opening_inventory = (
                        inventory[key]
                    )

                    sales = min(
                        demand,
                        opening_inventory
                        + production
                    )

                    closing_inventory = (
                        opening_inventory
                        + production
                        - sales
                    )

                    stockout = int(
                        sales < demand
                    )

                    inventory[key] = (
                        closing_inventory
                    )

                    # -----------------------------------------
                    # COSTS
                    # -----------------------------------------

                    production_cost = (
                        production
                        * self.products[
                            product
                        ]["production_cost"]
                    )

                    inventory_cost = (
                        closing_inventory
                        * 12
                    )

                    logistics_cost = (
                        sales
                        * 150
                    )

                    maintenance_cost = (
                        sum(
                            max(
                                0,
                                0.95
                                - machine_health[m]
                            )
                            for m in machine_health
                        )
                        * 100_000
                    )

                    # -----------------------------------------
                    # REVENUE / PROFIT
                    # -----------------------------------------

                    revenue = (
                        sales * price
                    )

                    profit = (
                        revenue
                        - production_cost
                        - marketing / 3
                        - inventory_cost
                        - logistics_cost
                        - maintenance_cost / 3
                    )

                    rows.append(
                        {
                            "date": date,
                            "day": day,
                            "product": product,
                            "region": region,

                            "price": price,
                            "competitor_price": competitor_price,

                            "marketing_spend": marketing,

                            "demand": demand,
                            "production": production,

                            "opening_inventory":
                                opening_inventory,

                            "closing_inventory":
                                closing_inventory,

                            "sales": sales,

                            "stockout": stockout,

                            "supplier": supplier,
                            "supplier_delayed":
                                int(
                                    supplier_delayed
                                ),

                            "lead_time": lead_time,

                            "revenue": revenue,

                            "production_cost":
                                production_cost,

                            "inventory_cost":
                                inventory_cost,

                            "logistics_cost":
                                logistics_cost,

                            "maintenance_cost":
                                maintenance_cost,

                            "profit": profit,

                            "avg_machine_health":
                                np.mean(
                                    list(
                                        machine_health.values()
                                    )
                                ),
                        }
                    )

        return pd.DataFrame(rows)


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    world = ManufacturingWorld(
        days=730,
        seed=42
    )

    df = world.generate()

    output_dir = Path(
        "data/raw"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "manufacturing_regional.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    print("=" * 60)
    print("CAUSAL-X SYNTHETIC MANUFACTURING WORLD")
    print("=" * 60)

    print(
        f"Rows generated: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min()} → "
        f"{df['date'].max()}"
    )

    print(
        f"Products: "
        f"{df['product'].unique().tolist()}"
    )

    print(
        f"Regions: "
        f"{df['region'].unique().tolist()}"
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    print("\nFirst 5 rows:")
    print(df.head())